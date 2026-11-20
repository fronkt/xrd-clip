#!/bin/bash
#SBATCH -A che260157
#SBATCH -p shared
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=30G
#SBATCH -t 02:00:00
#SBATCH -o /anvil/scratch/x-fcai3/xrdclip_ops/%x_%j.out
# CPU twin of anvil_poly_sweep.sbatch (gpu queue stalled 11-19): classical scores are CPU numpy in 12 either way; the
# model column runs on CPU. Resubmission D3: the submitted model (xrdclip_poly) through the same synthetic sweep as the D3 retrains (scripts/12,
# the sbv2 flags, 43 conditions incl. the 8 BFDH texture ones), on the same platform (Anvil A100), so the sweep
# table compares every model on identical draws. The checkpoint sits on user scratch (project fileset ENOSPC 11-18).
#   scp runs/xrdclip_poly/best.pt anvil:/anvil/scratch/x-fcai3/xrdclip_ops/d3eval/xrdclip_poly/best.pt
#   sbatch -J xc-polysweep scripts/box/anvil_poly_sweep.sbatch
set -u
P=/anvil/projects/x-che260157
export OUT=/anvil/scratch/x-fcai3/xrdclip_ops/d3eval_cpu
export PATH=$P/pxrd_env/bin:$PATH
export PYTHONPATH=$P/xrd-clip/src
cd "$P/xrd-clip"
echo "bundle $(cat BUNDLE_COMMIT)  host $(hostname)  out $OUT  start $(date -u +%FT%TZ)"
python -c "import torch; print(torch.__version__, 'cpu only')" || exit 2
export NUMPY_MADVISE_HUGEPAGE=0 MALLOC_MMAP_MAX_=0 MALLOC_TRIM_THRESHOLD_=68719476736
export CKPT=/anvil/scratch/x-fcai3/xrdclip_ops/d3eval/xrdclip_poly/best.pt
[ -f $CKPT ] || { echo "no checkpoint $CKPT"; exit 3; }
md5sum $CKPT
mkdir -p $OUT/logs_d3 $OUT/results || exit 2
# not GROUPS: bash reserves it (the user's group ids), ignores the assignment and expands it to the primary gid
CONDS="clean zero_offset sign strain domain_nm counts march= march_bfdh thermal field"
fail=0

sweep() {   # <condition prefix>; completion = the .done marker
  local tag=${1%=} rc o=$OUT/results/sbv3_poly_${1%=}
  [ -f $o.done ] && return 0
  NTHREADS=4 OMP_NUM_THREADS=4 python -W ignore scripts/12_strong_baselines.py --ckpt $CKPT --no_vc \
    --lnq_version 2 --only "$1" --out $o.json --ranks_out ${o}_ranks.npz > $OUT/logs_d3/sbv3_poly_$tag.log 2>&1
  rc=$?; [ $rc -eq 0 ] && touch $o.done
  echo "sweep poly $tag rc=$rc $(date -u +%FT%TZ)"
}
export -f sweep
printf '%s\n' $CONDS | xargs -P 4 -I{} bash -c "sweep {}"

# complete = a result for every condition of 12's conditions() (43), not N marker files
python -W ignore - $OUT/results poly <<'PY' || fail=1
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
cd $OUT && tar -cf poly_sweep_out.tar results/sbv3_poly_* logs_d3/sbv3_poly_* && md5sum poly_sweep_out.tar > poly_sweep_out.tar.md5 || fail=1
echo "END fail=$fail $(date -u +%FT%TZ)"
exit $fail
