#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/haozong/projects/MouseGeoForge"
PYTHON="${PROJECT_ROOT}/.venv-vggt-omega-eval/bin/python"
DATASET_ROOT="${PROJECT_ROOT}/water-reaching-dataset"
CHECKPOINT="${PROJECT_ROOT}/output/blenderpymouse_vggt_omega_all/run_20260728_relative_rig_camera_temporal4_conf75_rigfilter_s400_lr5e7/checkpoints/vggt_omega_relative_rig_camera_best_real_relative_rig.pt"

TAG="temporalrig375-624x416"
PERFRAME_ROOT="${DATASET_ROOT}/vggt-omega-eval-${TAG}-global-full-perframe-min2"
FIXEDREL_ROOT="${DATASET_ROOT}/vggt-omega-eval-${TAG}-global-full-fixedrel-pairwise-top50-min2"
ANIPOSE_GUI_ROOT="${DATASET_ROOT}/vggt-omega-anipose-${TAG}-gui-prior-aniposelib"
ANIPOSE_FIXED_ROOT="${DATASET_ROOT}/vggt-omega-anipose-${TAG}-fixedrel-aniposelib"
BASELINE_PERFRAME="${DATASET_ROOT}/vggt-omega-eval-original-current-b48-global-full-perframe-min2"
BASELINE_FIXEDREL="${DATASET_ROOT}/vggt-omega-eval-original-current-b48-global-full-fixedrel-pairwise-top50-min2"
BASELINE_ANIPOSE_GUI="${DATASET_ROOT}/vggt-omega-anipose-original-current-b48-gui-prior-aniposelib"
BASELINE_ANIPOSE_FIXED="${DATASET_ROOT}/vggt-omega-anipose-original-current-b48-fixedrel-aniposelib"
LOG_ROOT="${PROJECT_ROOT}/output/vggt_${TAG}_four_method_scoring"
REPORT_ROOT="${LOG_ROOT}/report"

mkdir -p "${LOG_ROOT}"
echo "$$" > "${LOG_ROOT}/pid.txt"

run_stage() {
  local name="$1"
  shift
  printf '[%s] stage_start=%s\n' "$(date --iso-8601=seconds)" "${name}" | tee -a "${LOG_ROOT}/pipeline.log"
  "$@" 2>&1 | tee "${LOG_ROOT}/${name}.log"
  printf '[%s] stage_done=%s\n' "$(date --iso-8601=seconds)" "${name}" | tee -a "${LOG_ROOT}/pipeline.log"
}

run_stage candidate_perframe \
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

run_stage candidate_fixedrel_pairwise \
  "${PYTHON}" -u "${DATASET_ROOT}/postprocess_vggt_omega_fixed_rig.py" \
  --source-root "${PERFRAME_ROOT}" \
  --output-root "${FIXEDREL_ROOT}" \
  --alignment-mode global \
  --aggregate weighted-mean \
  --min-views 2 \
  --score-top-fraction 0.5 \
  --camera-outlier-mode pairwise \
  --camera-outlier-z-threshold 3.5

run_stage candidate_anipose_gui_prior \
  "${PYTHON}" -u "${DATASET_ROOT}/evaluate_vggt_calibration_anipose_triangulation.py" \
  --source-root "${PERFRAME_ROOT}" \
  --output-root "${ANIPOSE_GUI_ROOT}" \
  --calibration-mode gui-prior \
  --likelihood-threshold 0.7 \
  --min-views 2 \
  --alignment-mode global \
  --triangulation-backend aniposelib \
  --overwrite

run_stage candidate_anipose_fixedrel \
  "${PYTHON}" -u "${DATASET_ROOT}/evaluate_vggt_calibration_anipose_triangulation.py" \
  --source-root "${FIXEDREL_ROOT}" \
  --output-root "${ANIPOSE_FIXED_ROOT}" \
  --calibration-mode fixedrel \
  --likelihood-threshold 0.7 \
  --min-views 2 \
  --alignment-mode global \
  --triangulation-backend aniposelib \
  --overwrite

run_stage temporal_rig_variance \
  "${PYTHON}" -u "${PROJECT_ROOT}/tools/analyze_vggt_rig_temporal_variance.py" \
  --model "Original" "${BASELINE_PERFRAME}" "#457b9d" \
  --model "TemporalRig375" "${PERFRAME_ROOT}" "#e76f51" \
  --output-dir "${LOG_ROOT}/temporal_rig_variance"

run_stage comparison_report \
  "${PYTHON}" -u "${DATASET_ROOT}/make_vggt_omega_four_method_report.py" \
  --method "VGGT calibration -> Aniposelib" "${BASELINE_ANIPOSE_GUI}" "${ANIPOSE_GUI_ROOT}" \
  --method "VGGT calibration -> Aniposelib fixedrel" "${BASELINE_ANIPOSE_FIXED}" "${ANIPOSE_FIXED_ROOT}" \
  --method "VGGT depth fixedrel pairwise" "${BASELINE_FIXEDREL}" "${FIXEDREL_ROOT}" \
  --method "VGGT depth per-frame" "${BASELINE_PERFRAME}" "${PERFRAME_ROOT}" \
  --candidate-label "TemporalRig375" \
  --output-dir "${REPORT_ROOT}"

printf '[%s] pipeline_done\n' "$(date --iso-8601=seconds)" | tee -a "${LOG_ROOT}/pipeline.log"
