#!/usr/bin/env bash
# Runs a command and appends its peak memory to OUT: GPU = nvidia-smi memory.used every second minus the reading
# before the start (the GPU is this job's under gpu_lock); host = summed RSS of the command's session (every process
# it starts: Spark's JVM, its Python workers).
#   bash experiments/tableB/peak.sh OUT LABEL cmd...
out=$1; label=$2; shift 2
smi() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
g0=$(smi); pg=$g0; ph=0; t0=$(date +%s)
setsid "$@" &   # no job control here, so setsid needs no fork: the session id is $!
pid=$!
while kill -0 $pid 2>/dev/null; do
  g=$(smi); [ "${g:-0}" -gt "$pg" ] && pg=$g
  h=$(ps -o rss= --sid $pid | awk '{s += $1} END {print int(s / 1024)}'); [ "${h:-0}" -gt "$ph" ] && ph=$h
  sleep 1
done
wait $pid; rc=$?
printf '{"time": "%s", "label": "%s", "rc": %d, "wall_s": %d, "gpu_idle_mb": %d, "peak_gpu_mb": %d, "peak_host_mb": %d}\n' \
  "$(date '+%F %T')" "$label" $rc $(( $(date +%s) - t0 )) "$g0" $(( pg - g0 )) "$ph" >> "$out"
exit $rc
