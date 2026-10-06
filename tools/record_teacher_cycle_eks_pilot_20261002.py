"""Persist measured checkpoint, preprocessing incident and EKS decision for next heartbeat."""
import datetime as dt,json,shutil
from pathlib import Path
from run_teacher_cycle_dlc_20261001 import write_state
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/teacher_temporal_cycle_20261001'
ensemble=json.loads((OUT/'lp_ensemble_17_31_73/summary.json').read_text())
eks=json.loads((OUT/'official_eks_17_31_73/summary.json').read_text())
parity={str(s):json.loads((OUT/f'continuous_lp_seed{s}_kl_exactnorm/manifest.json').read_text())[0]['labeled_batch_parity_max_px'] for s in [17,31,73]}
next_actions=[
    'Do NOT add more teacher epochs now. Three independent LP-KL seeds and official EKS pilot are complete.',
    'Build uncertainty-weighted FRAMEWISE reconstruction control before attributing EKS changes to temporal smoothing: identical observations, per-coordinate seed variance, fixed noise floor and robust reprojection; never optimize cameras.',
    'Assess conservative observation gates and quality versus coverage, true leave-camera-out, at least two supported nondegenerate rays, cheirality and missing/imputed flags. Include losses of dorsal coverage and shared confident failures.',
    'Use the same pending-free development clips and an additional predeclared development clip if safe; no locked holdout adaptation. Keep all generated 3D diagnostic until independent quality criteria are met.',
    'Consider official EKS variance inflation or process-noise sensitivity only as a bounded follow-up after the weighted-framewise control, not endless smoothing tuning.'
]
record=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),completed=['LP-KL seeds17/31/73','three-seed uncertainty audit','continuous predictions with matched normalization','official calibrated EKS and true LOO'],
    ensemble=ensemble,eks_paired=eks['paired'],
    preprocessing_incident=dict(initial_seed73_direct_vs_saved_discrepancy_px=2.8621826171875,
        after_matching_layout_and_batch_px=.6220092773436932,
        fix='Use exact torchvision CPU ToTensor+Normalize path from LP dataset; regenerate ALL seed clips in new *_exactnorm folders. Never relaxed 0.1px parity gate.',
        matched_path_parity_max_px=parity),
    conclusion='EKS improves LOO median 39.92->34.00px but worsens P90 125.78->150.01px; mixed result, not accepted pseudo-GT. Comparison also changes observation weighting, so it is not a pure temporal ablation.',
    visual='In FR2 frame800, EKS and framewise differ in hind-paw locations; body-surface points remain extrapolated in bottom view. Visualization alone cannot certify anatomical correctness.',
    next_actions=next_actions,locked_holdout_opened=False,original_labels_or_calibration_changed=False,pseudo_gt_released=False)
path=OUT/'heartbeat_20261002_0010_decision.json';path.write_text(json.dumps(record,indent=2))
write_state(stage='eks_pilot_evaluated',pid=None,active_run=None,mode='idle_between_heartbeats',
    latest_decision=str(path),next_action=next_actions,eks_result=eks['paired'],job_log=None)
dest=Path('/mnt/c/Users/55395/Downloads/MouseGeoForge-TeacherTemporal-20261001/official_eks_17_31_73')
shutil.copy2(path,dest/'cycle_decision.json');shutil.copy2(OUT/'official_eks_17_31_73/summary.json',dest/'summary.json')
print(json.dumps(dict(stage='eks_pilot_evaluated',parity=parity,next=next_actions),indent=2))
