#!/usr/bin/env bash
# Revision GPU box (tasks/todo.md Step 3): R4c realism retrain, R1 ln q v2 CSLS rescore (21 score), R2 ablations
# (28 score/eval, classical only), the G0 sweep rerun (ln q v2, crc32 seeds), and the retrained model's library
# embeddings. CSD-derived inputs (stick shards, checkpoint) arrive only by a manual-mode upload.
#   setsid nohup bash scripts/box/rev_gpu_box.sh > /workspace/box.log 2>&1 &
# Ends in a self-stop: 2 h after RUN_DONE (time to pull), or a watchdog stop after 30 min with no worker.
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
    if ps -eo args | grep -qE "[x]rdclip.train|[2]1_hubness|[2]8_rescore|[1]2_strong_baselines|[e]mbed_lib"; then idle=0; else idle=$((idle + 1)); fi
    [ -f /workspace/RUN_DONE ] && exit 0
    if [ $idle -ge 30 ]; then selfstop "watchdog: no worker for 30 min"; exit 0; fi
  done
) &

echo "[box] start $(date -u +%FT%TZ)"
md5sum -c MANIFEST.md5 > /dev/null || { echo "[box] MD5 MISMATCH"; selfstop "md5 mismatch"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; assert torch.cuda.is_available(); print('[box] torch', torch.__version__)" || { selfstop "no cuda"; exit 1; }
pip install -q scipy pymatgen > /dev/null 2>&1 || true
quota=$(awk '{ if ($1 == "max") print 0; else printf "%d", $1 / $2 }' /sys/fs/cgroup/cpu.max 2>/dev/null || echo 0)
[ "${quota:-0}" -gt 0 ] || quota=$(nproc)
echo "[box] cgroup quota ${quota} cores"
ulimit -n 65536
export PYTHONPATH=src NUMPY_MADVISE_HUGEPAGE=0 MALLOC_MMAP_MAX_=0 MALLOC_TRIM_THRESHOLD_=68719476736
mkdir -p logs results
TRUTH="--truth data/opxrd_truth.json --patterns data/opxrd_cnrs_organic.json"

# 1. R4c realism retrain: the original run's command plus --realism (configs/train_5090_polymorph.md)
(
  OMP_NUM_THREADS=4 python -m xrdclip.train --sticks cache/sticks --train_refs refcodes/scaffold_train.txt \
    --val_refs refcodes/scaffold_val.txt --poly_families refcodes/polymorph_families.json --batch_size 512 \
    --epochs 60 --lr 3e-4 --weight_decay 0.05 --embed_dim 256 --workers 24 --realism \
    --out runs/xrdclip_realism > logs/train.log 2>&1
  echo "EXIT $?" >> logs/train.log
  # library structure embeddings with the retrained model (19's per-shard embedder, 6 shards + extras in parallel)
  OMP_NUM_THREADS=1 python - > logs/embed_lib.log 2>&1 <<'EOF'
import os, sys
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, "scripts")
from importlib import import_module
bench = import_module("19_real_bench")
os.makedirs("cache/real_bench_realism", exist_ok=True)
jobs = [(p, n, "runs/xrdclip_realism/best.pt", 256,
         f"cache/real_bench_realism/emb_{os.path.basename(os.path.dirname(p))}_{os.path.basename(p)[:-4]}_{n}.npy")
        for p, n, _ in bench.shard_table(None)]
with ProcessPoolExecutor(len(jobs)) as ex:
    for out, dt in ex.map(bench._embed_shard, jobs):
        print("[embed_lib]", out, round(dt), "s", flush=True)
EOF
  echo "EXIT $?" >> logs/embed_lib.log
) &

# 2. R1: CSLS etc. for ln q v2 = the whole hub-pass score with LNQ_VERSION=2 (other methods reproduce)
(
  OMP_NUM_THREADS=1 LNQ_VERSION=2 python scripts/21_hubness.py score --device cuda --workers 8 \
    --work cache/real_bench < /dev/null > logs/hub_lnq2.log 2>&1
  echo "EXIT $?" >> logs/hub_lnq2.log
  # 3. R2 ablations (classical only; paired differences are computed locally against the r2 ranks)
  for tag in r2_noB r2_nolp r2_noka2 r2_nomask; do
    OMP_NUM_THREADS=1 python scripts/28_rescore.py score --tag $tag --device cuda --workers 8 --threads 4 $TRUTH \
      > logs/abl_$tag.log 2>&1
    OMP_NUM_THREADS=4 python scripts/28_rescore.py eval --tag $tag --no_model --base "" $TRUTH \
      --out results/rescore_$tag.json >> logs/abl_$tag.log 2>&1
    echo "EXIT $?" >> logs/abl_$tag.log
  done
) &

# 4. G0 sweep rerun with the ORIGINAL checkpoint (ln q v2 + the v1 column, crc32 seeds), 6 workers
sweep() { NTHREADS=2 OMP_NUM_THREADS=2 python scripts/12_strong_baselines.py --no_vc --lnq_version 2 --only "$1" \
            --out results/sbv2_$2.json --ranks_out results/sbv2_$2_ranks.npz > logs/sbv2_$2.log 2>&1
          echo "EXIT $?" >> logs/sbv2_$2.log; }
( sweep clean,zero_offset w1; sweep sign w8 ) &
sweep strain w2 &
( sweep domain_nm w3; sweep counts w5 ) &
sweep march w4 &
sweep thermal w6 &
sweep field w7 &
wait
echo "[box] all workers done $(date -u +%FT%TZ)"
mv cache/real_bench/full cache/real_bench/full_lnq2 2>/dev/null
tar -cf /workspace/out.tar logs results runs/xrdclip_realism cache/real_bench_realism cache/real_bench/full_lnq2 \
  $(ls -d cache/real_bench/rescore_r2_* 2>/dev/null | sed 's|$|/ranks.npz|')
md5sum /workspace/out.tar > /workspace/out.tar.md5
touch /workspace/RUN_DONE
echo "[box] RUN_DONE $(du -h /workspace/out.tar | cut -f1)"
for i in $(seq 1 120); do
  [ -f /workspace/PULLED ] && { echo "[box] pulled"; exit 0; }
  sleep 60
done
selfstop "2 h after RUN_DONE"
