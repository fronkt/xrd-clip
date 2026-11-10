#!/bin/bash
#SBATCH -A che260157-gpu
#SBATCH -p gpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus-per-node=1
#SBATCH -t 24:00:00
#SBATCH -o /anvil/scratch/x-fcai3/xrdclip_ops/%x_%j.out
# Resubmission D3 retrains (tasks/todo.md Section D). One run per job; resubmit the same job to continue (state.pt
# every epoch). Submit with:  sbatch -J xc-<run> --export=ALL,RUN=<run>,FLAGS="<train flags>" anvil_train.sbatch
set -u
P=/anvil/projects/x-che260157
export PATH=$P/pxrd_env/bin:$PATH
export PYTHONPATH=$P/xrd-clip/src
cd "$P/xrd-clip"
echo "bundle $(cat BUNDLE_COMMIT)  run $RUN  flags $FLAGS  host $(hostname)  start $(date -u +%FT%TZ)"
python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))" || exit 2
[ -f runs/$RUN/DONE ] && { echo "already done"; exit 0; }
python -m xrdclip.train --sticks cache/sticks --out runs/$RUN --epochs 60 --batch_size 512 --workers 28 \
    --poly_families refcodes/polymorph_families.json --val_mode degraded --resume $FLAGS
rc=$?
echo "train rc=$rc $(date -u +%FT%TZ)"
[ $rc -eq 0 ] && [ -f runs/$RUN/last.pt ] && touch runs/$RUN/DONE
exit $rc
