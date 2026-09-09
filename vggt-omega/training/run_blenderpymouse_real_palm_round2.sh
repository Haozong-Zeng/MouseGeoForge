#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/haozong/projects/MouseGeoForge"
RUN_DIR="${PROJECT_ROOT}/output/blenderpymouse_vggt_omega_all/run_20260718_real_palm_dense_lr5e7_e20_v2"
TRAIN_SCRIPT="${PROJECT_ROOT}/vggt-omega/training/finetune_blenderpymouse_water_reaching.py"

TRAIN_TRIALS=(
  E_L2_2024-11-18_1_Trial_1
  FB_L2_2024-03-12_1_Trial_1
  FB_L2_2024-03-12_1_Trial_4
  FB_L2_2024-03-16_1_Trial_3
  FB_L3_2024-03-12_1_Trial_3
  FB_R2_2024-03-15_1_Trial_1
  FB_R2_2024-03-16_1_Trial_1
  FB_R3_2024-03-14_1_Trial_3
  FB_R3_2024-03-16_1_Trial_5
  FG_L2_2024-03-05_1_Trial_4
  FG_L2_2024-03-06_1_Trial_63
  FG_R3_2024-03-06_1_Trial_2
  FJ_L2_2024-06-24_1_Trial_7
  FJ_L2_2024-06-29_1_Trial_2
  FS_L2_2024-06-25_1_Trial_2
  FU_R2L2_2024-06-27_1_Trial_1
)

VAL_TRIALS=(
  E_L2_2024-11-21_1_Trial_1
  FB_L2_2024-03-16_1_Trial_8
  FG_R3_2024-04-04_1_Trial_2
  FJ_L2_2024-07-29_1_Trial_3
  FS_L2_2024-06-29_1_Trial_51
  FU_R2L2_2024-06-29_1_Trial_31
)

join_trials() {
  local joined=""
  local trial
  for trial in "$@"; do
    [[ -n "${joined}" ]] && joined+=","
    joined+="${PROJECT_ROOT}/water-reaching-dataset/${trial}"
  done
  printf '%s' "${joined}"
}

TRAIN_ROOTS="$(join_trials "${TRAIN_TRIALS[@]}")"
VAL_ROOTS="$(join_trials "${VAL_TRIALS[@]}")"
mkdir -p "${RUN_DIR}/checkpoints"

exec "${PROJECT_ROOT}/.venv-vggt-omega-eval/bin/python" -u "${TRAIN_SCRIPT}" \
  --blendedmvs-root "${PROJECT_ROOT}/dataset/BlenderPyMouse" \
  --checkpoint "${PROJECT_ROOT}/vggt-omega/vggt_omega_1b_512.pt" \
  --output "${RUN_DIR}/vggt_omega_real_palm_dense_latest.pt" \
  --checkpoint-dir "${RUN_DIR}/checkpoints" \
  --checkpoint-every 0 \
  --log-csv "${RUN_DIR}/train_log.csv" \
  --image-size 256 \
  --num-views 3 \
  --batch-size 1 \
  --num-workers 2 \
  --epochs 20 \
  --max-steps 0 \
  --samples-per-scene 1 \
  --val-scenes 1 \
  --val-batches 4 \
  --val-every 500 \
  --real-val-roots "${VAL_ROOTS}" \
  --real-val-frame-step 20 \
  --real-val-frame-count 32 \
  --real-val-batches 4 \
  --real-train-roots "${TRAIN_ROOTS}" \
  --real-depth-val-roots "${VAL_ROOTS}" \
  --real-train-frame-step 5 \
  --real-train-frame-count 0 \
  --real-depth-val-frame-step 40 \
  --real-depth-val-frame-count 0 \
  --real-depth-likelihood-threshold 0.7 \
  --real-sparse-batch-size 8 \
  --real-sparse-every 2 \
  --real-sparse-weight 2.0 \
  --real-palm-point-weight 4.0 \
  --real-anchor-point-weight 1.0 \
  --real-depth-val-batches 0 \
  --train-mode dense \
  --lr 5e-7 \
  --weight-decay 0.01 \
  --camera-weight 0.0 \
  --depth-weight 0.2 \
  --depth-gradient-weight 0.05 \
  --scale-mode auto \
  --grad-clip 1.0 \
  --log-every 10 \
  --seed 42 \
  "$@"
