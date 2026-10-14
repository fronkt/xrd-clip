#!/usr/bin/env bash
# Frank's extra checks X2 + X3 (tasks/todo.md C8) on a rented CPU-rich box:
#   X2  critic2 re-rank with answer-blind zero-shifted peak lists (fit_zs.peaks), the 36 shortlists holding the answer
#   X3  critic2 refits of the 10 zero-padded pressure-series patterns (fit_x3.peaks): shortlists, control, injection
#   cd /workspace/xrd; setsid nohup bash scripts/box/critic2_x23_box.sh > /workspace/box.log 2>&1 < /dev/null &
# CSD-derived inputs (candidate CIFs) arrive only by a manual-mode upload and leave the box only as DIFF values per
# refcode pair. The local pull loop is armed in the launch command. Ends in a self-stop either way: right after the
# pull (PULLED), 6 h after RUN_DONE with no pull, or a watchdog stop after 30 min with no worker process.
set -u
cd /workspace/xrd || exit 1

selfstop() {
  local id key
  id=$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^CONTAINER_ID=//p')
  key=$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^CONTAINER_API_KEY=//p')
  echo "[box] SELF-STOP $(date -u +%FT%TZ) ($1)"
  curl -s -X PUT -H "Authorization: Bearer $key" -d '{"state":"stopped"}' "https://console.vast.ai/api/v0/instances/$id/"
}

(
  idle=0
  while true; do
    sleep 60
    # process NAME critic2 (this script's own name contains "critic2", so an args match would never go idle)
    if ps -eo comm= | grep -qx critic2 || ps -eo args= | grep -q "[2]9_critic2"; then idle=0; else idle=$((idle + 1)); fi
    [ -f /workspace/RUN_DONE ] && exit 0
    if [ $idle -ge 30 ]; then selfstop "watchdog: no worker for 30 min"; exit 0; fi
  done
) &

echo "[box] start $(date -u +%FT%TZ)"
md5sum -c --quiet MANIFEST.md5 || { echo "[box] MD5 MISMATCH"; selfstop "md5 mismatch"; exit 1; }
quota=$(awk '{ if ($1 == "max") print 0; else printf "%d", $1 / $2 }' /sys/fs/cgroup/cpu.max 2>/dev/null || echo 0)
[ "${quota:-0}" -gt 0 ] || quota=$(nproc)
echo "[box] cgroup quota ${quota} cores; nproc $(nproc); $(free -g | awk '/Mem/ {print $2 " GB RAM"}')"

# critic2 1.4 (the version validated locally on Windows), checksum from the release's SHA256SUMS
mkdir -p /workspace/critic2 && cd /workspace/critic2
curl -sSLO https://github.com/aoterodelaroza/critic2/releases/download/1.4/critic2-linux-x86_64.tar.gz
echo "2020d490ffba8d3d340f0a47f26eebf722391b959265d5c5c29de17c29a754f9  critic2-linux-x86_64.tar.gz" | sha256sum -c \
  || { echo "[box] critic2 checksum FAIL"; selfstop "critic2 checksum"; exit 1; }
tar -xzf critic2-linux-x86_64.tar.gz
export CRITIC2=$(find /workspace/critic2 -type f -name critic2 -perm -u+x | head -1)
echo "[box] critic2 at $CRITIC2"; printf "end\n" > t.cri; "$CRITIC2" t.cri | grep -E "version|features"
cd /workspace/xrd

export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
C2=$(( quota - 6 )); [ $C2 -lt 8 ] && C2=8
A=$(( C2 * 78 / 100 )); B=$(( C2 - A ))
JOBS=cache/critic2/x23_jobs.json
X2=$(python -c "import json; print(','.join(json.load(open('$JOBS'))['x2_pids']))")
X3=$(python -c "import json; print(','.join(json.load(open('$JOBS'))['x3_pids']))")
mkdir -p logs
echo "[box] X2 on $A processes, X3 on $B (+ 4 for the pairs)"
python scripts/29_critic2_rerank.py run --k 50 --timeout 14400 --procs "$A" --peaks fit_zs.peaks --suffix _zs \
  --pids "$X2" > logs/x2_run.log 2>&1 &
P1=$!
python scripts/29_critic2_rerank.py run --k 50 --timeout 14400 --procs "$B" --peaks fit_x3.peaks --suffix _x3 \
  --pids "$X3" > logs/x3_run.log 2>&1 &
P2=$!
python scripts/29_critic2_rerank.py pairs --pairs_file cache/critic2/x3_control_pairs.json --kind control \
  --peaks fit_x3.peaks --suffix _x3 --procs 2 --timeout 14400 > logs/x3_control.log 2>&1 &
P3=$!
python scripts/29_critic2_rerank.py pairs --pairs_file cache/critic2/x3_inject_pairs.json --kind inject \
  --peaks fit_x3.peaks --suffix _x3 --procs 2 --timeout 14400 > logs/x3_inject.log 2>&1 &
P4=$!
for p in $P1 $P2 $P3 $P4; do wait $p; echo "[box] worker $p exit $?"; done
echo "[box] all workers done $(date -u +%FT%TZ); X2 $(wc -l < cache/critic2/diffs_zs.jsonl), X3 $(wc -l < cache/critic2/diffs_x3.jsonl)"
tar -cf /workspace/out.tar logs cache/critic2/diffs_zs.jsonl cache/critic2/diffs_x3.jsonl cache/critic2/control_x3.jsonl \
  cache/critic2/inject_x3.jsonl
md5sum /workspace/out.tar > /workspace/out.tar.md5
touch /workspace/RUN_DONE
echo "[box] RUN_DONE $(du -h /workspace/out.tar | cut -f1)"
for i in $(seq 1 360); do
  [ -f /workspace/PULLED ] && { selfstop "pulled"; exit 0; }
  sleep 60
done
selfstop "6 h after RUN_DONE"
