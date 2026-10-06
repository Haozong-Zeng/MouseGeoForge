"""Durable heartbeat decision based on completed geometry diagnostics."""
import datetime as dt
import json
from run_teacher_cycle_dlc_20261001 import OUT,write_state

def main():
    names=['consensus_pilot_v1_final','consensus_stress1_v1']
    runs={n:dict(metrics=json.loads((OUT/n/'summary.json').read_text()),
                 failure_audit=json.loads((OUT/n/'failure_review/summary.json').read_text())) for n in names}
    report=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),results=runs,
        implemented='Custom fixed-camera pair hypotheses, minimum3 agreeing views and strict majority of eligible views; confidence>=.6, spread<=10px, residual<=15px, rayangle>=3deg, positive depths, equal-support conflicting hypotheses>5mm rejected. No temporal interpolation. Thresholds fixed before run, not calibrated.',
        tests=['Analytic geometry controls inherited from previous run','Exact synthetic4-view reconstruction','Synthetic bad fourth camera rejected','Held-view coordinates including NaN cannot influence fit','Two-view-only solution rejected'],
        comparison=[
            'Pilot62/165 retained, median10.74/P9036.08px, but three errors>100px and mean28.03px vs weighted same-cohort18.19px. Do not accept based on median/P90 alone.',
            'Stress89/219 retained(40.64%), median10.60/P9063.06px vs weighted same-cohort13.72/79.85px. Eleven errors>50px, seven>100px. Increases coverage relative to strict all-view gate but not reliable pseudo-GT.',
            'Stress paw18 only6 retained observations, median254.93px. Aggregate metrics hide severe anatomical identity errors. Dorsalpoint7 has zero retained manual observations in both windows.',
            'Inlier-drop subset shifts approximately1-1.6mmP90 for evaluable>=4-support points do NOT certify accuracy: mutually consistent cameras may select the same wrong paw.'
        ],
        inspected_images=[
            'consensus_stress1_v1/failure_review/01_AZR2L2_f1169_k18_heldA.jpg',
            'consensus_pilot_v1_final/failure_review/01_FR2_f779_k18_heldA.jpg'
        ],
        visual_findings=[
            'FR2f779k18: held bottomA manual/cyan target on one hindpaw, red3D projection on a different paw. B/C/D support another location with confidence.95-.99 and seedspread2.3-4.8px. HeldA error341.8px. Geometric consensus can reinforce anatomical misidentification.',
            'AZR2L2f1169k18: D/E/F support opposite-side location; heldA error466.6px. Manual-only other-view control17.41px for this observation is much smaller, arguing against explaining this instance solely as calibration error. Not proof all labels/calibration are correct.'
        ],
        implementation_incidents=[
            'consensus_pilot_v1 stopped when JSON encountered numpyint64 in inlier indices; partial NPZ retained.',
            'consensus_pilot_v1_complete stopped on numpyint64 coverage counts; partial NPZ/observations retained.',
            'Both types explicitly converted to Pythonint; consensus_pilot_v1_final and consensus_stress1_v1 completed with unchanged numerical settings. Do not consume the two incomplete directories.'
        ],
        next_actions=[
            'Keep current teachers frozen. Diagnose anatomical confusion for paws15-18 using existing development manual labels, per-view confusion matrices and nearest OTHER annotated paw distances. Separate left/right and fore/hind errors. Missing label means unknown, not certified invisible.',
            'Analyze independent heatmap hypotheses/peak geometry or joint one-to-one paw assignment as a bounded diagnostic; do not automatically swap original labels or use held-camera observations in LOO inference. Shared confident errors require identity evidence, not just more cameras.',
            'Evaluate whether body-oriented anatomical constraints are observable from reliable nose/body anchors; dorsal coverage currently poor, so never force rigid skeleton/filled hidden paws as GT.',
            'Only after identity handling improves, assess temporal continuity across retained segments with gap masks and lag/amplitude metrics. Do not smooth across excluded groups.',
            'If confidence/identity cannot be validated, export missing/uncertain annotations for human review rather than bulk pseudo-labels. Deadline remains2026-10-02T07:21:52UTC; automation still active.'
        ],locked_holdout_opened=False,pending_groups_used=False,original_labels_or_calibration_changed=False,pseudo_gt_released=False)
    dest=OUT/'heartbeat_20261002_0105_decision.json';dest.write_text(json.dumps(report,indent=2))
    write_state(stage='consensus_diagnosed_shared_paw_identity_failure',pid=None,active_run=None,
                mode='idle_between_heartbeats',job_deadline_utc=None,latest_decision=str(dest),
                latest_result=str(OUT/'consensus_stress1_v1/failure_review/summary.json'),
                next_action=report['next_actions'][0],pseudo_gt_released=False)
    print(dest)

if __name__=='__main__':main()
