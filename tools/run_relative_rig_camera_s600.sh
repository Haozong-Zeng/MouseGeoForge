#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/haozong/projects/MouseGeoForge
OUTPUT="${ROOT}/output/blenderpymouse_vggt_omega_all/run_20260728_relative_rig_camera_head_624x416_s600_lr1e6_baseline2"

mkdir -p "${OUTPUT}"
exec env PYTHONUNBUFFERED=1 \
  "${ROOT}/.venv-vggt-omega-eval/bin/python" \
  "${ROOT}/tools/launch_relative_rig_camera_training.py" \
  --template-checkpoint \
  "${ROOT}/output/blenderpymouse_vggt_omega_all/run_20260728_relative_rig_camera_head_624x416_s300_lr2e6/checkpoints/vggt_omega_relative_rig_camera_step000300.pt" \
  --base-checkpoint \
  "${ROOT}/output/blenderpymouse_vggt_omega_all/run_20260728_geometry_coupled_aligned_624x416_official_e10/vggt_omega_1b_512_geometryaligned_step6000_full.pt" \
  --output-dir "${OUTPUT}" \
  --max-steps 600 \
  --lr 1e-6 \
  --center-weight 1.5 \
  --baseline-weight 2.0 \
  --validation-every 50 \
  --checkpoint-every 50 \
  >"${OUTPUT}/train_console.log" 2>&1
