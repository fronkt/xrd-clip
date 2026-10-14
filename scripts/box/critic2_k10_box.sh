#!/usr/bin/env bash
# R3 critic2 re-rank, K = 10 with a 4-h cap per comparison, on a rented CPU-rich box (10-13: the global VC search
# costs ~V^2 in the candidate's cell volume, too slow for the laptop; see 29's --k note).
#   cd /workspace/xrd; setsid nohup bash scripts/box/critic2_k10_box.sh > /workspace/box.log 2>&1 < /dev/null &
# CSD-derived inputs (candidate CIFs) arrive only by a manual-mode upload and leave the box only as DIFF values per
# refcode pair and peak lists of the open opXRD patterns. The local pull loop is armed in the launch command.
# Ends in a self-stop either way: right after the pull (PULLED), 6 h after RUN_DONE with no pull, or a watchdog
# stop after 30 min with no worker process.
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
C2=$(( quota - 2 )); [ $C2 -lt 4 ] && C2=4
mkdir -p logs
missing=$(for d in cache/critic2/pattern_*/; do [ -f "$d/fit.peaks" ] || basename "$d"; done | paste -sd, -)
echo "[box] patterns without a peak fit: ${missing:-none}"
if [ -z "$missing" ]; then
  python scripts/29_critic2_rerank.py run --k 10 --timeout 14400 --procs "$C2" > logs/c2_run.log 2>&1
  echo "EXIT $?" >> logs/c2_run.log
else
  # the main run lists its jobs at start, before the late fits exist, so it leaves those patterns out; a second
  # run restricted to them starts once their fits land
  python scripts/29_critic2_rerank.py fit --procs 2 --fit_timeout 14400 > logs/c2_fit.log 2>&1 &
  FIT=$!
  python scripts/29_critic2_rerank.py run --k 10 --timeout 14400 --procs $(( C2 - 2 )) > logs/c2_run.log 2>&1 &
  RUN=$!
  wait $FIT
  echo "[box] late fits done $(date -u +%FT%TZ)"; cat logs/c2_fit.log | grep -v cached
  python scripts/29_critic2_rerank.py run --k 10 --timeout 14400 --procs 10 --pids "$missing" > logs/c2_run_late.log 2>&1
  echo "EXIT $?" >> logs/c2_run_late.log
  wait $RUN
  echo "EXIT $?" >> logs/c2_run.log
fi
echo "[box] all workers done $(date -u +%FT%TZ); $(wc -l < cache/critic2/diffs.jsonl) comparisons in diffs.jsonl"
tar -cf /workspace/out.tar logs cache/critic2/diffs.jsonl $(ls cache/critic2/*/fit.peaks 2>/dev/null)
md5sum /workspace/out.tar > /workspace/out.tar.md5
touch /workspace/RUN_DONE
echo "[box] RUN_DONE $(du -h /workspace/out.tar | cut -f1)"
for i in $(seq 1 360); do
  [ -f /workspace/PULLED ] && { selfstop "pulled"; exit 0; }
  sleep 60
done
selfstop "6 h after RUN_DONE"
