#!/usr/bin/env bash
#SBATCH --job-name=mouse-semseg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
# Set --partition/--account/--output at submission for the target cluster.
set -euo pipefail
: "${MOUSEGEOFORGE_ROOT:?Set repository path on shared filesystem}"
: "${SEMSEG_PYTHON:?Set Python interpreter in the cluster environment}"
: "${SEMSEG_ITEMS:?Set the verified mouse-semseg-v1 manifest path}"
: "${SEMSEG_RUN:?Set a new data-disk output directory}"
cd "$MOUSEGEOFORGE_ROOT"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
SEMSEG_NPROC="${SEMSEG_NPROC:-8}"
SEMSEG_ARM="${SEMSEG_ARM:-M}"
if [[ "$SEMSEG_ARM" == F || "$SEMSEG_ARM" == U ]]; then
  SEMSEG_STEPS="${SEMSEG_STEPS:-6000}"
else
  SEMSEG_STEPS="${SEMSEG_STEPS:-18000}"
fi
# One torchrun launcher per node, not eight competing srun launchers.
exec "$SEMSEG_PYTHON" -m torch.distributed.run --standalone --nnodes=1 \
  --nproc-per-node="$SEMSEG_NPROC" tools/train_mouse_semseg_20261024.py train \
  --root "$MOUSEGEOFORGE_ROOT" --items "$SEMSEG_ITEMS" --run-dir "$SEMSEG_RUN" \
  --arm "$SEMSEG_ARM" --steps "$SEMSEG_STEPS" --global-batch=8 --micro-batch=1 \
  --device=cuda --precision=bf16 --workers=2 --allow-unregistered-fixed-budget "$@"
