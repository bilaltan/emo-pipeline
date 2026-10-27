"""The store Stage 3b's pyarrow workers read the Phase 2 tables from: one code path for local runs and AWS.

The table URL picks the store: s3:// and s3a:// open pyarrow's S3FileSystem (as on EMR), anything else the local disk.
Nothing else changes between the two.

Metering (emu, run_phase3b's s3_emu, from parse_emu) wraps that store so every read call on a data file is counted as
one GET with its bytes. pyarrow issues the same calls against S3 (footer, then column-chunk ranges, each one ranged
GET), so the counts are pyarrow's own; HEAD and LIST requests are not counted. With a latency and bandwidth each GET
also waits latency + bytes / bandwidth, which emulates S3 on a local store (Track B's s3_emu_bench model: the latency
and bandwidth are assumptions, not measurements of S3). On a real S3 store only counting is allowed, since its GETs
already pay real latency. Metered reads pass through Python, one read at a time per file, so a timing run on S3
reads without metering and a separate run counts.
"""
import threading
import time

STATS = {"gets": 0, "bytes": 0}   # this process; Stage 3b takes the difference around each unit's read
_LOCK = threading.Lock()
_FS = {}


def is_s3(uri):
    return uri.startswith(("s3://", "s3a://"))


def to_path(uri):
    """uri as a path in its store: bucket/key on S3, a local path otherwise."""
    if is_s3(uri):
        return uri.split("://", 1)[1].rstrip("/")
    if uri.startswith("file:"):
        from urllib.parse import unquote, urlparse
        return unquote(urlparse(uri).path).rstrip("/")
    return uri.rstrip("/")


def parse_emu(spec):
    """None / '' -> None (plain reads); 'count' -> (0, 0), count only; 'LAT_MS,MBPS' -> emulated S3, e.g. '20,80'."""
    if not spec:
        return None
    if spec == "count":
        return (0.0, 0.0)
    lat, mbps = (float(x) for x in spec.split(","))
    if lat < 0 or mbps <= 0:
        raise ValueError(f"s3_emu needs latency >= 0 ms and bandwidth > 0 MB/s, got {spec!r}")
    return (lat, mbps)


def check(uri, emu):
    if emu and any(emu) and is_s3(uri):
        raise ValueError(f"s3_emu={emu} on a real S3 store ({uri}) would add latency on top of S3's; use 'count'")


def filesystem(uri, emu=None):
    """pyarrow filesystem of the store holding uri, metered when emu is set (one per store and setting per process)."""
    check(uri, emu)
    key = (is_s3(uri), emu)
    if key not in _FS:
        import pyarrow.fs as pafs
        base = pafs.S3FileSystem(region="us-east-1") if is_s3(uri) else pafs.LocalFileSystem()
        _FS[key] = base if emu is None else _metered(base, *emu)
    return _FS[key]


def snapshot(uri):
    """A Delta table's current version read from its _delta_log, without Spark: (pyarrow filesystem, data file
    paths in that store, version). The store keeps every overwritten version's files until VACUUM, so listing
    the directory would repeat rows. Starts from the last checkpoint when there is one, then replays the commits."""
    import json
    import pyarrow.fs as pafs
    import pyarrow.parquet as pq
    from urllib.parse import unquote
    fs, root = filesystem(uri), to_path(uri)
    log = f"{root}/_delta_log"
    names = sorted(i.base_name for i in fs.get_file_info(pafs.FileSelector(log)))
    live, start = set(), -1
    if "_last_checkpoint" in names:
        with fs.open_input_stream(f"{log}/_last_checkpoint") as f:
            start = int(json.loads(f.read())["version"])
        for n in names:
            if n.startswith(f"{start:020d}.checkpoint") and n.endswith(".parquet"):
                adds = pq.read_table(f"{log}/{n}", filesystem=fs, columns=["add"]).column("add").to_pylist()
                live.update(unquote(a["path"]) for a in adds if a)
    commits = [n for n in names if n.endswith(".json") and n[:20].isdigit() and int(n[:20]) > start]
    for n in commits:
        with fs.open_input_stream(f"{log}/{n}") as f:
            for line in f.read().decode().splitlines():
                a = json.loads(line)
                if "add" in a:
                    live.add(unquote(a["add"]["path"]))
                elif "remove" in a:
                    live.discard(unquote(a["remove"]["path"]))
    version = max([int(n[:20]) for n in commits] + [start])
    return fs, sorted(f"{root}/{p}" for p in live), version


class _MeteredFile:
    """A store file whose every read call is one GET: counted, and delayed by latency + bytes / bandwidth."""

    def __init__(self, f, lat_s, bps):
        self.f, self.lat, self.bps, self.closed = f, lat_s, bps, False

    def read(self, n=-1):
        data = self.f.read(None if n is None or n < 0 else n)
        with _LOCK:
            STATS["gets"] += 1
            STATS["bytes"] += len(data)
        if self.bps:
            time.sleep(self.lat + len(data) / self.bps)   # releases the GIL: GETs of other threads overlap
        return data

    def seek(self, off, whence=0):
        return self.f.seek(off, whence)

    def tell(self):
        return self.f.tell()

    def close(self):
        self.f.close()
        self.closed = True

    def readable(self):
        return True

    def seekable(self):
        return True

    def writable(self):
        return False


def _metered(base, lat_ms, mbps):
    import pyarrow as pa
    import pyarrow.fs as pafs
    lat_s, bps = lat_ms / 1e3, mbps * 1e6

    class Handler(pafs.FileSystemHandler):
        def get_type_name(self):
            return "metered-" + base.type_name

        def normalize_path(self, path):
            return path

        def equals(self, other):
            return self is other

        def get_file_info(self, paths):
            return base.get_file_info(paths)

        def get_file_info_selector(self, selector):
            return base.get_file_info(selector)

        def open_input_file(self, path):
            return pa.PythonFile(_MeteredFile(base.open_input_file(path), lat_s, bps), mode="r")

        def open_input_stream(self, path):
            return self.open_input_file(path)

        def _ro(self, *a, **k):
            raise OSError("the metered store is read-only")
        create_dir = delete_dir = delete_dir_contents = delete_root_dir_contents = delete_file = _ro
        move = copy_file = open_output_stream = open_append_stream = _ro

    return pafs.PyFileSystem(Handler())
