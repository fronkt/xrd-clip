#!/usr/bin/env bash
# Box A follow-up (10-13): the R1 ln q v2 CSLS score (21 score) ran out of GPU memory on the 24 GB 4090 while the
# R4c retrain held ~17 GB and the sweep workers the rest. 21 score is not resumable, so rerun it alone once the main
# queue is done (RUN_DONE), into its own work dir; the main out.tar is untouched.
#   setsid nohup bash scripts/box/rev_gpu_retry.sh > /workspace/retry.log 2>&1 < /dev/null &
# Ends in a self-stop: 2 h after RETRY_DONE without RETRY_PULLED; the score itself is capped at 3 h.
set -u
cd /workspace/xrd || exit 1

selfstop() {
  local id key
  id=$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^CONTAINER_ID=//p')
  key=$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^CONTAINER_API_KEY=//p')
  echo "[retry] SELF-STOP $(date -u +%FT%TZ) ($1)"
  curl -s -X PUT -H "Authorization: Bearer $key" -d '{"state":"stopped"}' "https://console.vast.ai/api/v0/instances/$id/"
}

echo "[retry] waiting for RUN_DONE $(date -u +%FT%TZ)"
until [ -f /workspace/RUN_DONE ]; do sleep 60; done
echo "[retry] start $(date -u +%FT%TZ); GPU in use: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
export PYTHONPATH=src NUMPY_MADVISE_HUGEPAGE=0 MALLOC_MMAP_MAX_=0 MALLOC_TRIM_THRESHOLD_=68719476736
ulimit -n 65536
mkdir -p cache/real_bench_lnq2r logs
cp cache/real_bench/hub_ctx.npz cache/real_bench_lnq2r/
OMP_NUM_THREADS=1 LNQ_VERSION=2 timeout 3h python scripts/21_hubness.py score --device cuda --workers 8 \
  --work cache/real_bench_lnq2r < /dev/null > logs/hub_lnq2_retry.log 2>&1
echo "EXIT $?" >> logs/hub_lnq2_retry.log
tar -cf /workspace/out_retry.tar logs/hub_lnq2_retry.log cache/real_bench_lnq2r/full
md5sum /workspace/out_retry.tar > /workspace/out_retry.tar.md5
touch /workspace/RETRY_DONE
echo "[retry] RETRY_DONE $(du -h /workspace/out_retry.tar | cut -f1) $(date -u +%FT%TZ)"
for i in $(seq 1 120); do
  [ -f /workspace/RETRY_PULLED ] && { echo "[retry] pulled"; exit 0; }
  sleep 60
done
selfstop "2 h after RETRY_DONE"
