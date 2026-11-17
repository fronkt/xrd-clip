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
#   sbatch -J xc-d3eval anvil_d3_eval.sbatch
set -u
P=/anvil/projects/x-che260157
export PATH=$P/pxrd_env/bin:$PATH
export PYTHONPATH=$P/xrd-clip/src
cd "$P/xrd-clip"
echo "bundle $(cat BUNDLE_COMMIT)  host $(hostname)  start $(date -u +%FT%TZ)"
python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))" || exit 2
export NUMPY_MADVISE_HUGEPAGE=0 MALLOC_MMAP_MAX_=0 MALLOC_TRIM_THRESHOLD_=68719476736
mkdir -p logs_d3 results
RUNS="d3_fix d3_sym_s0 d3_sym_s1"
GROUPS="clean zero_offset sign strain domain_nm counts march= march_bfdh thermal field"
fail=0

for r in $RUNS; do
  [ -f runs/$r/DONE ] || { echo "no finished run $r"; exit 3; }
  OMP_NUM_THREADS=8 python -W ignore scripts/44_d3_embed.py --ckpt runs/$r/best.pt --emb_dir cache/real_bench_$r \
    > logs_d3/embed_$r.log 2>&1
  rc=$?; echo "embed $r rc=$rc $(date -u +%FT%TZ)"; [ $rc -eq 0 ] || fail=1
done

sweep() {   # <run> <condition prefix>; 12 rewrites its json after every condition, so completion = the .done marker
  local tag=${2%=} rc
  [ -f results/sbv3_$1_$tag.done ] && return 0
  NTHREADS=4 OMP_NUM_THREADS=4 python -W ignore scripts/12_strong_baselines.py --ckpt runs/$1/best.pt --no_vc \
    --lnq_version 2 --only "$2" --out results/sbv3_$1_$tag.json --ranks_out results/sbv3_$1_${tag}_ranks.npz \
    > logs_d3/sbv3_$1_$tag.log 2>&1
  rc=$?; [ $rc -eq 0 ] && touch results/sbv3_$1_$tag.done
  echo "sweep $1 $tag rc=$rc $(date -u +%FT%TZ)"
}
export -f sweep
for r in $RUNS; do
  printf '%s\n' $GROUPS | xargs -P 7 -I{} bash -c "sweep $r {}"
done

n_ok=$(ls results/sbv3_*.done 2>/dev/null | wc -l)
echo "sweep outputs $n_ok / $(( $(echo $RUNS | wc -w) * $(echo $GROUPS | wc -w) ))"
[ "$n_ok" -eq $(( $(echo $RUNS | wc -w) * $(echo $GROUPS | wc -w) )) ] || fail=1
tar -cf d3_eval_out.tar logs_d3 results/sbv3_* $(for r in $RUNS; do echo cache/real_bench_$r; done)
md5sum d3_eval_out.tar > d3_eval_out.tar.md5
echo "END fail=$fail $(du -h d3_eval_out.tar | cut -f1) $(date -u +%FT%TZ)"
exit $fail
