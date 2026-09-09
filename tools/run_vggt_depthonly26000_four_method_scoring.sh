#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/haozong/projects/MouseGeoForge"
PYTHON="${PROJECT_ROOT}/.venv-vggt-omega-eval/bin/python"
DATASET_ROOT="${PROJECT_ROOT}/water-reaching-dataset"
CHECKPOINT="${PROJECT_ROOT}/output/blenderpymouse_vggt_omega_all/run_20260727_step500_style_transfer_allsynthetic_dense_kp_e20/vggt_omega_1b_512_step26000_style_transfer_allsynthetic_best_real_palm_full.pt"

PERFRAME_ROOT="${DATASET_ROOT}/vggt-omega-eval-depthonly26000-current-b48-global-full-perframe-min2"
FIXEDREL_ROOT="${DATASET_ROOT}/vggt-omega-eval-depthonly26000-current-b48-global-full-fixedrel-pairwise-top50-min2"
ANIPOSE_GUI_ROOT="${DATASET_ROOT}/vggt-omega-anipose-depthonly26000-current-b48-gui-prior-aniposelib"
ANIPOSE_FIXED_ROOT="${DATASET_ROOT}/vggt-omega-anipose-depthonly26000-current-b48-fixedrel-aniposelib"
LOG_ROOT="${PROJECT_ROOT}/output/vggt_depthonly26000_four_method_scoring"

mkdir -p "${LOG_ROOT}"
echo "$$" > "${LOG_ROOT}/pid.txt"

run_stage() {
  local name="$1"
  shift
  printf '[%s] stage_start=%s\n' "$(date --iso-8601=seconds)" "${name}" | tee -a "${LOG_ROOT}/pipeline.log"
  "$@" 2>&1 | tee "${LOG_ROOT}/${name}.log"
  printf '[%s] stage_done=%s\n' "$(date --iso-8601=seconds)" "${name}" | tee -a "${LOG_ROOT}/pipeline.log"
}

run_stage depthonly_perframe \
  "${PYTHON}" -u "${DATASET_ROOT}/run_vggt_omega_dataset_eval.py" \
  --dataset-root "${DATASET_ROOT}" \
  --output-root "${PERFRAME_ROOT}" \
  --checkpoint "${CHECKPOINT}" \
  --frame-start 0 \
  --frame-step 1 \
  --batch-size 48 \
  --min-views 2 \
  --alignment-mode global \
  --camera-stabilization-mode per-frame \
  --resume

run_stage depthonly_fixedrel_pairwise \
  "${PYTHON}" -u "${DATASET_ROOT}/postprocess_vggt_omega_fixed_rig.py" \
  --source-root "${PERFRAME_ROOT}" \
  --output-root "${FIXEDREL_ROOT}" \
  --alignment-mode global \
  --aggregate weighted-mean \
  --min-views 2 \
  --score-top-fraction 0.5 \
  --camera-outlier-mode pairwise \
  --camera-outlier-z-threshold 3.5

run_stage depthonly_anipose_gui_prior \
  "${PYTHON}" -u "${DATASET_ROOT}/evaluate_vggt_calibration_anipose_triangulation.py" \
  --source-root "${PERFRAME_ROOT}" \
  --output-root "${ANIPOSE_GUI_ROOT}" \
  --calibration-mode gui-prior \
  --likelihood-threshold 0.7 \
  --min-views 2 \
  --alignment-mode global \
  --triangulation-backend aniposelib \
  --overwrite

run_stage depthonly_anipose_fixedrel \
  "${PYTHON}" -u "${DATASET_ROOT}/evaluate_vggt_calibration_anipose_triangulation.py" \
  --source-root "${FIXEDREL_ROOT}" \
  --output-root "${ANIPOSE_FIXED_ROOT}" \
  --calibration-mode fixedrel \
  --likelihood-threshold 0.7 \
  --min-views 2 \
  --alignment-mode global \
  --triangulation-backend aniposelib \
  --overwrite

printf '[%s] pipeline_done\n' "$(date --iso-8601=seconds)" | tee -a "${LOG_ROOT}/pipeline.log"
