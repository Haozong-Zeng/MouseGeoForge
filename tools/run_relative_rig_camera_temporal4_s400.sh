#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/haozong/projects/MouseGeoForge
OUTPUT="${ROOT}/output/blenderpymouse_vggt_omega_all/run_20260728_relative_rig_camera_temporal4_conf75_rigfilter_s400_lr5e7"

mkdir -p "${OUTPUT}"
exec env PYTHONUNBUFFERED=1 \
  "${ROOT}/.venv-vggt-omega-eval/bin/python" \
  "${ROOT}/tools/launch_relative_rig_camera_training.py" \
  --template-checkpoint \
  "${ROOT}/output/blenderpymouse_vggt_omega_all/run_20260728_relative_rig_camera_head_624x416_s300_lr2e6/checkpoints/vggt_omega_relative_rig_camera_step000300.pt" \
  --base-checkpoint \
  "${ROOT}/output/blenderpymouse_vggt_omega_all/run_20260728_geometry_coupled_aligned_624x416_official_e10/vggt_omega_1b_512_geometryaligned_step6000_full.pt" \
  --output-dir "${OUTPUT}" \
  --max-steps 400 \
  --lr 5e-7 \
  --center-weight 1.0 \
  --baseline-weight 1.0 \
  --synthetic-relative-rig-weight 0.5 \
  --real-relative-rig-weight 4.0 \
  --temporal-frames 4 \
  --shared-rig-weight 1.0 \
  --rig-min-confidence 0.99 \
  --rig-min-coverage 0.70 \
  --rig-keep-fraction 0.75 \
  --rig-max-rotation-error-deg 30 \
  --rig-max-center-error 0.40 \
  --rig-outlier-rotation-deg 15 \
  --rig-outlier-center 0.35 \
  --validation-every 25 \
  --checkpoint-every 25 \
  >"${OUTPUT}/train_console.log" 2>&1
