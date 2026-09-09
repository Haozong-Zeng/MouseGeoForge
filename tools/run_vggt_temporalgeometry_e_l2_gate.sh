#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/haozong/projects/MouseGeoForge
PYTHON="${ROOT}/.venv-vggt-omega-eval/bin/python"
DATASET="${ROOT}/water-reaching-dataset"
CHECKPOINT="${ROOT}/output/blenderpymouse_vggt_omega_all/run_20260729_relative_rig_camera_temporal4_geometry_s300_lr2e7/checkpoints/vggt_omega_relative_rig_camera_best_real_geometry.pt"
TAG="temporalgeometry275-624x416-e-l2"
PERFRAME="${DATASET}/vggt-omega-eval-${TAG}-global-full-perframe-min2"
FIXEDREL="${DATASET}/vggt-omega-eval-${TAG}-global-full-fixedrel-pairwise-top50-min2"
ANIPOSE_GUI="${DATASET}/vggt-omega-anipose-${TAG}-gui-prior-aniposelib"
ANIPOSE_FIXED="${DATASET}/vggt-omega-anipose-${TAG}-fixedrel-aniposelib"
REPORT="${ROOT}/output/vggt_temporalrig375-624x416_four_method_scoring/temporalgeometry_e_l2_gate"

"${PYTHON}" -u "${DATASET}/run_vggt_omega_dataset_eval.py" \
  --dataset-root "${DATASET}" \
  --output-root "${PERFRAME}" \
  --checkpoint "${CHECKPOINT}" \
  --frame-start 0 \
  --frame-step 1 \
  --batch-size 48 \
  --min-views 2 \
  --alignment-mode global \
  --camera-stabilization-mode per-frame \
  --trial-name E_L2_2024-11-18_1_Trial_1 \
  --trial-name E_L2_2024-11-21_1_Trial_1 \
  --resume

"${PYTHON}" -u "${DATASET}/postprocess_vggt_omega_fixed_rig.py" \
  --source-root "${PERFRAME}" \
  --output-root "${FIXEDREL}" \
  --alignment-mode global \
  --aggregate weighted-mean \
  --min-views 2 \
  --score-top-fraction 0.5 \
  --camera-outlier-mode pairwise \
  --camera-outlier-z-threshold 3.5

"${PYTHON}" -u "${DATASET}/evaluate_vggt_calibration_anipose_triangulation.py" \
  --source-root "${PERFRAME}" \
  --output-root "${ANIPOSE_GUI}" \
  --calibration-mode gui-prior \
  --likelihood-threshold 0.7 \
  --min-views 2 \
  --alignment-mode global \
  --triangulation-backend aniposelib \
  --overwrite

"${PYTHON}" -u "${DATASET}/evaluate_vggt_calibration_anipose_triangulation.py" \
  --source-root "${FIXEDREL}" \
  --output-root "${ANIPOSE_FIXED}" \
  --calibration-mode fixedrel \
  --likelihood-threshold 0.7 \
  --min-views 2 \
  --alignment-mode global \
  --triangulation-backend aniposelib \
  --overwrite

"${PYTHON}" -u "${DATASET}/make_vggt_omega_four_method_report.py" \
  --method "VGGT calibration -> Aniposelib" \
  "${DATASET}/vggt-omega-anipose-geometryaligned6000-624x416-gui-prior-aniposelib" \
  "${ANIPOSE_GUI}" \
  --method "VGGT calibration -> Aniposelib fixedrel" \
  "${DATASET}/vggt-omega-anipose-geometryaligned6000-624x416-fixedrel-aniposelib" \
  "${ANIPOSE_FIXED}" \
  --method "VGGT depth fixedrel pairwise" \
  "${DATASET}/vggt-omega-eval-geometryaligned6000-624x416-global-full-fixedrel-pairwise-top50-min2" \
  "${FIXEDREL}" \
  --method "VGGT depth per-frame" \
  "${DATASET}/vggt-omega-eval-geometryaligned6000-624x416-global-full-perframe-min2" \
  "${PERFRAME}" \
  --candidate-label TemporalGeometry275 \
  --output-dir "${REPORT}"
