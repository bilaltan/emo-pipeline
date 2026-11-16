#!/usr/bin/env python3
"""
DistDGL step 3: start one machine's processes, as DGL's tools/launch.py does over ssh: its graph server(s), then
its trainers (DGL clients and torch.distributed ranks). Machine m serves partition m. Every machine runs this with
the same machine list, in the same order; run() returns when this machine's trainers exit.

On EMR, launch_spark.py calls run() inside one Spark barrier task per worker (no ssh between nodes). Locally,
test_two_machines.sh calls it in two network namespaces, each with its own address and /dev/shm.

    python experiments/distdgl/machine.py --machine 0 --ips 10.0.0.1,10.0.0.2 --part-config /tmp/dd/g.json ...
"""
import argparse
import os
import subprocess
import time
import sys

TRAIN_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py")


def run(machine, ips, part_config, workdir, trainers=1, servers=1, omp=4, port=30050, master_port=29500,
        ifname=None, train_args=(), python=sys.executable):
    os.makedirs(workdir, exist_ok=True)
    ip_config = os.path.join(workdir, "ip_config.txt")
    with open(ip_config, "w") as f:
        f.write("".join(f"{ip} {port}\n" for ip in ips))
    common = {"DGL_NUM_SAMPLER": "0", "DGL_NUM_CLIENT": str(len(ips) * trainers), "DGL_CONF_PATH": part_config,
              "DGL_IP_CONFIG": ip_config, "DGL_NUM_SERVER": str(servers), "DGL_GRAPH_FORMAT": "csc",
              "PYTHONUNBUFFERED": "1"}
    if ifname:   # gloo and DGL pick the interface from the hostname otherwise
        common.update(GLOO_SOCKET_IFNAME=ifname, TP_SOCKET_IFNAME=ifname)
    cmd = [python, TRAIN_PY, "--ip-config", ip_config, "--part-config", part_config, *train_args]
    procs = []
    for s in range(servers):
        env = dict(os.environ, **common, DGL_ROLE="server", DGL_SERVER_ID=str(machine * servers + s),
                   DGL_KEEP_ALIVE="0", OMP_NUM_THREADS="1")
        procs.append(subprocess.Popen(cmd, env=env))
    for t in range(trainers):
        env = dict(os.environ, **common, DGL_DIST_MODE="distributed", DGL_ROLE="client", OMP_NUM_THREADS=str(omp),
                   MASTER_ADDR=ips[0], MASTER_PORT=str(master_port), RANK=str(machine * trainers + t),
                   WORLD_SIZE=str(len(ips) * trainers), LOCAL_RANK=str(t))
        procs.append(subprocess.Popen(cmd, env=env))
    # Servers exit once every client has left. A failed process leaves the others waiting on it forever, so the
    # first non-zero exit stops this machine (the other machines' trainers then fail on the lost connection).
    while any(p.poll() is None for p in procs):
        if any(p.poll() not in (None, 0) for p in procs):
            for p in procs:
                if p.poll() is None:
                    p.kill()
            break
        time.sleep(2)
    rcs = [p.wait() for p in procs]
    if any(rcs):
        raise RuntimeError(f"machine {machine}: exit codes {rcs} (servers first, then trainers)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", type=int, required=True)
    ap.add_argument("--ips", required=True, help="every machine's address, comma-separated, machine 0 first")
    ap.add_argument("--part-config", required=True)
    ap.add_argument("--workdir", default="/tmp/distdgl")
    ap.add_argument("--trainers", type=int, default=1, help="trainer processes per machine")
    ap.add_argument("--omp", type=int, default=4, help="threads per trainer")
    ap.add_argument("--ifname", default=None)
    args, train_args = ap.parse_known_args()
    run(args.machine, args.ips.split(","), args.part_config, args.workdir, trainers=args.trainers, omp=args.omp,
        ifname=args.ifname, train_args=train_args)


if __name__ == "__main__":
    main()
