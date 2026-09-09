#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/haozong/projects/MouseGeoForge"
cd "$PROJECT_ROOT"

MODE="background"
if [[ "${1:-}" == "--foreground" ]]; then
  MODE="foreground"
  shift
fi

CHECKPOINT="output/blenderpymouse_vggt_omega_all/run_20260716_base_lr5e6_e50/vggt_omega_1b_512_blenderpymouse_best_real_full.pt"
EVAL_ROOT="water-reaching-dataset/vggt-omega-eval-bestreal-e14-global-full-fixedrel-pairwise-top50-min2"
ANIPOSE_ROOT="water-reaching-dataset/vggt-omega-anipose-bestreal-e14-fixedrel-aniposelib"
LOG="$EVAL_ROOT/pipeline.log"
PID_FILE="$EVAL_ROOT/pipeline.pid"

mkdir -p "$EVAL_ROOT"

run_pipeline() {
  .venv-vggt-omega-eval/bin/python3 -u \
    water-reaching-dataset/run_vggt_omega_dataset_eval.py \
    --dataset-root water-reaching-dataset \
    --output-root "$EVAL_ROOT" \
    --checkpoint "$CHECKPOINT" \
    --frame-start 0 \
    --frame-step 1 \
    --batch-size 8 \
    --min-views 2 \
    --alignment-mode global \
    --camera-stabilization-mode avg-relative-extrinsic \
    --score-top-fraction 0.5 \
    --camera-outlier-mode pairwise \
    --camera-outlier-z-threshold 3.5

  .venv-vggt-omega-eval/bin/python3 -u \
    water-reaching-dataset/evaluate_vggt_calibration_anipose_triangulation.py \
    --source-root "$EVAL_ROOT" \
    --output-root "$ANIPOSE_ROOT" \
    --calibration-mode fixedrel \
    --triangulation-backend aniposelib \
    --min-views 2 \
    --alignment-mode global
}

if [[ "$MODE" == "foreground" ]]; then
  echo "$$" > "$PID_FILE"
  run_pipeline
  exit 0
fi

nohup bash "$0" --foreground > "$LOG" 2>&1 &
echo "$!" > "$PID_FILE"
echo "PID=$(cat "$PID_FILE")"
echo "LOG=$LOG"
echo "EVAL_ROOT=$EVAL_ROOT"
echo "ANIPOSE_ROOT=$ANIPOSE_ROOT"
