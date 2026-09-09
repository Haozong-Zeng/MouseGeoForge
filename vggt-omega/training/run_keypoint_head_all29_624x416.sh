#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/home/haozong/projects/MouseGeoForge
DATA_ROOT="${PROJECT_ROOT}/water-reaching-dataset"
PYTHON="${PROJECT_ROOT}/.venv-vggt-omega-eval/bin/python"
TRAIN_SCRIPT="${PROJECT_ROOT}/vggt-omega/training/finetune_blendedmvs.py"
RUN_NAME="${RUN_NAME:-run_20260728_keypoint_head_all29_frozen_624x416_e5}"
RUN_DIR="${PROJECT_ROOT}/output/blenderpymouse_vggt_omega_all/${RUN_NAME}"
REAL_SCALE_MAP="${PROJECT_ROOT}/output/coordinate_alignment/vggt_original_trial_rig_scales_624x416.json"
BASE_CHECKPOINT="${PROJECT_ROOT}/output/blenderpymouse_vggt_omega_all/run_20260728_geometry_coupled_aligned_624x416_official_e10/vggt_omega_1b_512_geometryaligned_step6000_full.pt"
EPOCHS="${EPOCHS:-5}"
MAX_STEPS="${MAX_STEPS:-0}"
DRY_RUN="${DRY_RUN:-none}"

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

for index in "${!TRAIN_TRIALS[@]}"; do
  TRAIN_TRIALS[$index]="${DATA_ROOT}/${TRAIN_TRIALS[$index]}"
done
for index in "${!VAL_TRIALS[@]}"; do
  VAL_TRIALS[$index]="${DATA_ROOT}/${VAL_TRIALS[$index]}"
done
TRAIN_ROOTS=$(IFS=,; echo "${TRAIN_TRIALS[*]}")
VAL_ROOTS=$(IFS=,; echo "${VAL_TRIALS[*]}")

mkdir -p "${RUN_DIR}/checkpoints"
echo "$$" > "${RUN_DIR}/train.pid"

exec "${PYTHON}" "${TRAIN_SCRIPT}" \
  --blendedmvs-root "${PROJECT_ROOT}/dataset/BlenderPyMouse" \
  --checkpoint "${BASE_CHECKPOINT}" \
  --output "${RUN_DIR}/vggt_omega_keypoint_head_all29_latest.pt" \
  --checkpoint-dir "${RUN_DIR}/checkpoints" \
  --checkpoint-every 1000 \
  --log-csv "${RUN_DIR}/train_log.csv" \
  --image-width 624 \
  --image-height 416 \
  --num-views 3 \
  --batch-size 1 \
  --num-workers 2 \
  --epochs "${EPOCHS}" \
  --max-steps "${MAX_STEPS}" \
  --samples-per-scene 1 \
  --val-scenes 16 \
  --val-batches 16 \
  --val-every 250 \
  --scene-path-includes "*PyMouseStudio-*" \
  --enable-keypoint-head \
  --no-forward-depth-head \
  --keypoint-head-subset all \
  --keypoint-head-weight 1 \
  --real-keypoint-head-weight 0.25 \
  --keypoint-head-heatmap-weight 1 \
  --keypoint-head-visibility-weight 0.1 \
  --keypoint-head-center3d-weight 5 \
  --keypoint-head-fixed-rig-weight 1 \
  --keypoint-head-predicted-rig-weight 0.25 \
  --keypoint-head-heatmap-sigma 6 \
  --train-mode keypoint_head \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --lr-schedule warmup_cosine \
  --warmup-ratio 0.05 \
  --min-lr-ratio 0.02 \
  --camera-weight 0 \
  --depth-weight 0 \
  --point-weight 0 \
  --match-weight 0 \
  --keypoint-region-weight 0 \
  --keypoint-sparse-depth-weight 0 \
  --keypoint-reprojection-weight 0 \
  --real-geometry-train-roots "${TRAIN_ROOTS}" \
  --real-geometry-val-roots "${VAL_ROOTS}" \
  --real-geometry-cameras A,B,C \
  --real-geometry-train-frame-step 10 \
  --real-geometry-val-frame-step 20 \
  --real-geometry-val-batches 36 \
  --real-geometry-batch-size 1 \
  --real-geometry-every 2 \
  --real-geometry-likelihood-threshold 0.7 \
  --real-geometry-scale-mode vggt_rig \
  --real-geometry-scale-map "${REAL_SCALE_MAP}" \
  --real-geometry-weight 0 \
  --real-geometry-camera-weight 0 \
  --grad-clip 1 \
  --log-every 10 \
  --seed 42 \
  --scale-mode mean_point_distance \
  --canonicalize-camera \
  --no-save-full \
  --dry-run "${DRY_RUN}"
