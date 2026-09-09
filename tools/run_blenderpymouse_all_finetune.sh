#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/haozong/projects/MouseGeoForge"
cd "$PROJECT_ROOT"

MODE="background"
if [[ "${1:-}" == "--foreground" ]]; then
  MODE="foreground"
  shift
fi

RUN_DIR="${1:-output/blenderpymouse_vggt_omega_all/run_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_DIR/checkpoints"

LOG="$RUN_DIR/train_stdout.log"
PID_FILE="$RUN_DIR/pid.txt"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-vggt-omega/vggt_omega_1b_512.pt}"
LR="${LR:-1e-5}"
EPOCHS="${EPOCHS:-5}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-250}"
VAL_EVERY="${VAL_EVERY:-250}"
VAL_BATCHES="${VAL_BATCHES:-2}"
REAL_VAL_FRAME_COUNT="${REAL_VAL_FRAME_COUNT:-41}"
REAL_VAL_FRAME_STEP="${REAL_VAL_FRAME_STEP:-10}"
REAL_VAL_BATCHES="${REAL_VAL_BATCHES:-16}"
REAL_VAL_ROOTS="${REAL_VAL_ROOTS:-water-reaching-dataset/E_L2_2024-11-18_1_Trial_1,water-reaching-dataset/FB_L2_2024-03-16_1_Trial_3,water-reaching-dataset/FB_R2_2024-03-15_1_Trial_1,water-reaching-dataset/FG_L2_2024-03-05_1_Trial_4,water-reaching-dataset/FG_R3_2024-04-04_1_Trial_2,water-reaching-dataset/FJ_L2_2024-06-24_1_Trial_7,water-reaching-dataset/FS_L2_2024-06-25_1_Trial_2,water-reaching-dataset/FU_R2L2_2024-06-27_1_Trial_1}"

CMD=(
  .venv-vggt-omega-eval/bin/python3 -u
  vggt-omega/training/finetune_blenderpymouse_water_reaching.py \
  --blendedmvs-root dataset/BlenderPyMouse \
  --checkpoint "$INIT_CHECKPOINT" \
  --output "$RUN_DIR/vggt_omega_blenderpymouse_all_light_heads_latest.pt" \
  --log-csv "$RUN_DIR/train_log.csv" \
  --checkpoint-dir "$RUN_DIR/checkpoints" \
  --checkpoint-every "$CHECKPOINT_EVERY" \
  --image-size 256 \
  --num-views 3 \
  --batch-size 1 \
  --num-workers 2 \
  --epochs "$EPOCHS" \
  --max-steps 0 \
  --samples-per-scene 2 \
  --train-mode light_heads \
  --lr "$LR" \
  --depth-weight 0.2 \
  --depth-gradient-weight 0.05 \
  --val-every "$VAL_EVERY" \
  --val-batches "$VAL_BATCHES" \
  --real-val-roots "$REAL_VAL_ROOTS" \
  --real-val-frame-count "$REAL_VAL_FRAME_COUNT" \
  --real-val-frame-step "$REAL_VAL_FRAME_STEP" \
  --real-val-batches "$REAL_VAL_BATCHES"
)

if [[ "$MODE" == "foreground" ]]; then
  echo "$$" > "$PID_FILE"
  exec "${CMD[@]}" > "$LOG" 2>&1
fi

nohup "${CMD[@]}" > "$LOG" 2>&1 &

echo "$!" > "$PID_FILE"
echo "RUN_DIR=$RUN_DIR"
echo "PID=$(cat "$PID_FILE")"
echo "LOG=$LOG"
