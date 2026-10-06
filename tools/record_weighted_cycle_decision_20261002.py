"""Consolidate completed diagnostic runs; never release labels for training."""
import datetime as dt
import json
from pathlib import Path
from run_teacher_cycle_dlc_20261001 import write_state

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/teacher_temporal_cycle_20261001'

def main():
    runs = {}
    for name in ['weighted_framewise_pilot_floor4', 'weighted_framewise_stress1_floor4']:
        report = json.loads((OUT/name/'summary.json').read_text())
        rows = json.loads((OUT/name/'observations.json').read_text())
        report['distinct_manual_source_groups'] = len({(r['session'], r['frame']) for r in rows})
        report['manual_source_frames'] = {s: sorted({r['frame'] for r in rows if r['session']==s}) for s in sorted({r['session'] for r in rows})}
        runs[name] = report
    decision = dict(
        utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        stage='weighted_controls_and_frozen_development_stress_complete',
        results=runs,
        interpretation=[
            'Minimum observation noise prevents tiny ensemble variances from dominating reconstruction. It is a heuristic, not calibrated uncertainty.',
            'Robust weighted framewise median improves in both pilot and later same-source development windows, but large errors persist. Stress P90 is approximately138px for both robust framewise and EKSfloor4.',
            'Gaussian floor4 is the closer no-time control for EKSfloor4; robust loss also changes the objective, so robust-vs-EKS is not a pure temporal ablation. EKS also differs in nonlinear inference/initialization.',
            'Strict gates reject most points and disproportionately remove paws. Retained-cohort errors cannot be compared with full-cohort errors to claim general accuracy gains.',
            'Manual observation counts are not independent sample counts. Both windows share the same two development-validation videos; no new animal or source generalization is established.',
            'Manual 2D residuals do not identify true3D accuracy or isolate calibration, synchronization, annotation correspondence and teacher errors.'
        ],
        visual_inspection=[
            'Inspected A/F snapshot panels for AZR2L2 f129 and1104, FR2 f714 and1104. White rings are sparse manual annotations; all rendered3D projections use all views and are visualization, not LOO evaluation.',
            'AZR2L2 f1104: EKS paw18 shifts away from its bottom-view white ring toward the body; weighted framewise is closer in that view. Side-view dorsal predictions still differ from manual rings.',
            'FR2 f1104: paw discrepancies remain for both methods; strict gating retains mainly nose and leaves most other points blank. Missing points are not filled.',
            'No claim of dense animation correctness based on these four still images. MP4 comparisons exported for user review.'
        ],
        next_actions=[
            'Do not increase teacher epochs yet; preserve all three LP-KL seeds and DLC baselines.',
            'Implement a bounded fixed-camera geometric-consensus diagnostic on existing development clips. Predeclare thresholds; propose hypotheses from camera subsets but require at least3 agreeing, nondegenerate, positive-depth views. Keep all selected/inlier/rejected view IDs and avoid held-camera input during LOO.',
            'Compare consensus against current robust framewise and gated controls at matched retained cohorts. Include coverage per keypoint and camera-subset sensitivity; do not choose favorable2-view solutions to inflate coverage.',
            'If failures persist, isolate teacher-vs-manual correspondence/geometry using accepted manual-only controls, without changing original labels or cameras. Do not solve inconsistency by inventing hidden labels.',
            'Temporal follow-up only after reliable observation selection: measure action amplitude and lag, not merely smoother motion. Stop new experiments at07:21:52UTC and pause automation during cleanup.'
        ],
        locked_holdout_opened=False,
        pending_groups_used=False,
        original_labels_or_calibration_changed=False,
        pseudo_gt_released=False,
        train_allowed=False,
        windows_exports='C:/Users/55395/Downloads/MouseGeoForge-TeacherTemporal-20261001/'
    )
    path=OUT/'heartbeat_20261002_0040_decision.json'
    path.write_text(json.dumps(decision,indent=2))
    write_state(stage=decision['stage'],pid=None,active_run=None,mode='idle_between_heartbeats',
                job_deadline_utc=None,job_log=None,latest_decision=str(path),
                latest_result=str(OUT/'weighted_framewise_stress1_floor4/summary.json'),
                next_action=decision['next_actions'][1],pseudo_gt_released=False)
    print(json.dumps({k:dict(points=v['manual_observations'],source_groups=v['distinct_manual_source_groups'],frames=v['manual_source_frames']) for k,v in runs.items()},indent=2))

if __name__=='__main__':main()
