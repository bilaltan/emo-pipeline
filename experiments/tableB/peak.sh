#!/usr/bin/env bash
# Runs a command and appends its peak memory to OUT: GPU = nvidia-smi memory.used every second minus the reading
# before the start (the GPU must be this job's alone); host = summed RSS of the command's session (every process
# it starts: Spark's JVM, its Python workers), and apart: Spark's JVM (RSS) and the rest, the Python driver and
# workers, as RSS and as PSS (pages shared by the forked workers and their libraries counted once, not per process).
# Each field is its own peak over the run. OUT_trace/LABEL.tsv keeps every sample (epoch s, GPU, total, JVM, rest RSS,
# rest PSS; MB), so a peak can be put in its phase: Stage 3 is where the GPU is in use.
#   bash experiments/tableB/peak.sh OUT LABEL cmd...
out=$1; label=$2; shift 2
smi() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
# -> total RSS, JVM RSS, rest RSS, rest PSS (MB) of session $1; a process that exits mid-read only costs that sample
host() {
  local f=() q
  for q in $(ps -o pid= --sid $1); do f+=(/proc/$q/comm /proc/$q/status /proc/$q/smaps_rollup); done
  awk 'FILENAME ~ /comm$/ {j = ($0 == "java")}
       FILENAME ~ /status$/ && /^VmRSS:/ {t += $2; if (j) jv += $2; else p += $2}
       FILENAME ~ /smaps_rollup$/ && /^Pss:/ {if (!j) pp += $2}
       END {printf "%d %d %d %d\n", t / 1024, jv / 1024, p / 1024, pp / 1024}' "${f[@]}" 2>/dev/null
}
gmax=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
trace=${out%.jsonl}_trace/${label// /_}.tsv; mkdir -p "$(dirname "$trace")"; : > "$trace"
g0=$(smi); pg=$g0; ph=0; pj=0; pp=0; ps_=0; t0=$(date +%s)
setsid "$@" &   # no job control here, so setsid needs no fork: the session id is $!
pid=$!
while kill -0 $pid 2>/dev/null; do
  g=$(smi); [ "${g:-0}" -gt "$pg" ] && pg=$g
  # a full card spills into host memory (sysmem fallback): slow training, and it can run the host out of memory
  if [ "${g:-0}" -ge $(( gmax - 1024 )) ]; then echo "peak.sh: GPU full (${g} of ${gmax} MB), stopping $label"; pkill -9 -s $pid; fi
  read -r h j p s <<< "$(host $pid)"
  [ "${h:-0}" -gt "$ph" ] && ph=$h; [ "${j:-0}" -gt "$pj" ] && pj=$j
  [ "${p:-0}" -gt "$pp" ] && pp=$p; [ "${s:-0}" -gt "$ps_" ] && ps_=$s
  echo "$(date +%s) $(( ${g:-0} - g0 )) ${h:-0} ${j:-0} ${p:-0} ${s:-0}" >> "$trace"
  sleep 1
done
wait $pid; rc=$?
printf '{"time": "%s", "label": "%s", "rc": %d, "wall_s": %d, "gpu_idle_mb": %d, "peak_gpu_mb": %d, "peak_host_mb": %d, "peak_host_jvm_mb": %d, "peak_host_py_mb": %d, "peak_host_py_pss_mb": %d}\n' \
  "$(date '+%F %T')" "$label" $rc $(( $(date +%s) - t0 )) "$g0" $(( pg - g0 )) "$ph" "$pj" "$pp" "$ps_" >> "$out"
exit $rc
