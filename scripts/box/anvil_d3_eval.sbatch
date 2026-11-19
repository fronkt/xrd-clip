#!/bin/bash
#SBATCH -A che260157-gpu
#SBATCH -p gpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus-per-node=1
#SBATCH -t 08:00:00
#SBATCH -o /anvil/scratch/x-fcai3/xrdclip_ops/%x_%j.out
# Resubmission D3 evaluation inputs (tasks/todo.md Section D, E2). For each retrained checkpoint: library structure
# + pattern embeddings on the GPU (scripts/44, GPU/CPU parity checked), then the synthetic polymorph sweep with the
# BFDH texture conditions (scripts/12, the sbv2 flags). The real-pattern evaluations (28 eval, 30 eval) run locally
# on the pulled embeddings. Every step skips finished outputs, so a resubmitted job continues.
# Inputs are read from the project fileset; every output goes to user scratch ($OUT), because the project fileset
# refused new files on 11-18 (ENOSPC at 91.7 % of quota, cause not established).
#   sbatch -J xc-d3eval scripts/box/anvil_d3_eval.sbatch
set -u
P=/anvil/projects/x-che260157
export OUT=/anvil/scratch/x-fcai3/xrdclip_ops/d3eval
export PATH=$P/pxrd_env/bin:$PATH
export PYTHONPATH=$P/xrd-clip/src
cd "$P/xrd-clip"
echo "bundle $(cat BUNDLE_COMMIT)  host $(hostname)  out $OUT  start $(date -u +%FT%TZ)"
python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))" || exit 2
export NUMPY_MADVISE_HUGEPAGE=0 MALLOC_MMAP_MAX_=0 MALLOC_TRIM_THRESHOLD_=68719476736
mkdir -p $OUT/logs_d3 $OUT/results $OUT/cache || exit 2
RUNS="d3_fix d3_sym_s0 d3_sym_s1"
# not GROUPS: bash reserves it (the user's group ids), ignores the assignment and expands it to the primary gid
CONDS="clean zero_offset sign strain domain_nm counts march= march_bfdh thermal field"
fail=0

for r in $RUNS; do
  [ -f runs/$r/DONE ] || { echo "no finished run $r"; exit 3; }
  OMP_NUM_THREADS=8 python -W ignore scripts/44_d3_embed.py --ckpt runs/$r/best.pt --emb_dir $OUT/cache/real_bench_$r \
    > $OUT/logs_d3/embed_$r.log 2>&1
  rc=$?; echo "embed $r rc=$rc $(date -u +%FT%TZ)"; [ $rc -eq 0 ] || fail=1
done

sweep() {   # <run> <condition prefix>; 12 rewrites its json after every condition, so completion = the .done marker
  local tag=${2%=} rc o=$OUT/results/sbv3_$1_${2%=}
  [ -f $o.done ] && return 0
  NTHREADS=4 OMP_NUM_THREADS=4 python -W ignore scripts/12_strong_baselines.py --ckpt runs/$1/best.pt --no_vc \
    --lnq_version 2 --only "$2" --out $o.json --ranks_out ${o}_ranks.npz > $OUT/logs_d3/sbv3_$1_$tag.log 2>&1
  rc=$?; [ $rc -eq 0 ] && touch $o.done
  echo "sweep $1 $tag rc=$rc $(date -u +%FT%TZ)"
}
export -f sweep
for r in $RUNS; do
  printf '%s\n' $CONDS | xargs -P 7 -I{} bash -c "sweep $r {}"
done

# complete = every run has a result for every condition of 12's conditions() (43), not N marker files
python -W ignore - $OUT/results $RUNS <<'PY' || fail=1
import json, os, sys
from importlib import import_module
sys.path.insert(0, "scripts")
want = set(import_module("12_strong_baselines").conditions())
res, bad = sys.argv[1], 0
for r in sys.argv[2:]:
    have = set()
    for f in os.listdir(res):
        if f.startswith(f"sbv3_{r}_") and f.endswith(".json"):
            have |= set(json.load(open(os.path.join(res, f)))["conditions"])
    print(f"sweep {r}: {len(have & want)} / {len(want)} conditions", flush=True)
    bad += have != want
sys.exit(bad)
PY
cd $OUT && tar -cf d3_eval_out.tar logs_d3 results cache && md5sum d3_eval_out.tar > d3_eval_out.tar.md5 || fail=1
echo "END fail=$fail $(du -h $OUT/d3_eval_out.tar | cut -f1) $(date -u +%FT%TZ)"
exit $fail
