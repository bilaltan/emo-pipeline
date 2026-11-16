#!/usr/bin/env bash
# DistDGL on two "machines" on one host, before any AWS run: two network namespaces joined by a veth pair, each with
# its own address and /dev/shm, so DGL's machine ids, shared-memory partitions and gloo work as on two hosts.
# Partitions the Track A cache in 2 (partition.py), then starts machine.py in each namespace. Needs root:
#
#   wsl -d Ubuntu-24.04 -u root -- bash experiments/distdgl/test_two_machines.sh <track A cache> <tag> [epochs]
set -e
CACHE=${1:?Track A cache directory}; TAG=${2:-test}; EPOCHS=${3:-}
U=${DD_USER:-btan4}
PY=/home/$U/mamba/envs/distdgl/bin/python
REPO=$(cd "$(dirname "$0")/../.." && pwd)
W=/tmp/distdgl_$TAG
AS_U="runuser -u $U -- env HOME=/home/$U"
rm -rf $W && mkdir -p $W && chown $U $W
$AS_U $PY $REPO/experiments/distdgl/partition.py --cache $CACHE --parts 2 --out $W/parts

ip netns del dd0 2>/dev/null || true; ip netns del dd1 2>/dev/null || true
ip netns add dd0; ip netns add dd1
trap 'ip netns del dd0; ip netns del dd1' EXIT
ip link add dd0v type veth peer name dd1v
for i in 0 1; do
    ip link set dd${i}v netns dd$i
    ip netns exec dd$i ip addr add 10.77.0.$((i + 1))/24 dev dd${i}v
    ip netns exec dd$i ip link set dd${i}v up
    ip netns exec dd$i ip link set lo up
done
pids=()
for i in 0 1; do
    # ip netns exec gives each machine its own mount namespace, so this /dev/shm is private to it
    ip netns exec dd$i sh -c "mount -t tmpfs tmpfs /dev/shm && exec $AS_U $PY $REPO/experiments/distdgl/machine.py \
        --machine $i --ips 10.77.0.1,10.77.0.2 --part-config $W/parts/g.json --workdir $W/m$i --ifname dd${i}v \
        --trainers 1 --omp 4 --tag $TAG --out $REPO/results/icde/distdgl_local.jsonl ${EPOCHS:+--epochs $EPOCHS}" \
        > $W/m$i.log 2>&1 &
    pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait $p || rc=$?; done
grep -h "epoch\|RESULT\|Error\|error" $W/m0.log | tail -20
echo "two-machine DistDGL rc=$rc (logs $W/m0.log, $W/m1.log)"
exit $rc
