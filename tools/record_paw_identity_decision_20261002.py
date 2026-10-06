"""Record identity/candidate evidence and preserve newly suspected label unchanged."""
import datetime as dt,json
from run_teacher_cycle_dlc_20261001 import OUT,write_state

def main():
    audit=json.loads((OUT/'paw_identity_audit_v1/summary.json').read_text())
    peaks=json.loads((OUT/'paw_heatmap_peaks_v1/summary.json').read_text())
    collisions={n:json.loads((OUT/n/'summary.json').read_text()) for n in ['consensus_pilot_collision15','consensus_stress1_collision15']}
    review=[dict(id='SEM01',session='AZR2L2',frame=584,camera='D',keypoint_id=18,
        definition='right_hindpaw_center',status='suspected_semantic_mismatch_needs_human_review',
        evidence='Original manual k18 is drawn near forepaws in camD, whereas bottomcamA k18 is on hindpaw. Checked synchronized montage of all available development views. No correction performed.',
        image='C:/Users/55395/Downloads/MouseGeoForge-TeacherTemporal-20261001/paw_identity_manual_review/AZR2L2_f584_manual_paws.jpg',
        policy='Keep frozen evaluation results unchanged, disclose questionable reference. Do not promote this group to new pseudo-GT. Not proof every other camera label is correct.')]
    (OUT/'paw_identity_audit_v1/new_semantic_review_candidates.json').write_text(json.dumps(review,indent=2))
    report=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),identity_audit=audit,collision_ablation=collisions,heatmap_candidates=peaks,
        conclusions=[
            '785 annotated development paw observations:52 suspected nearest-other-paw confusions,46left/right,6fore/hind.23 retain high confidence and low ensemble spread. Annotation-relative counts, not independently certified model failures.',
            'Prediction-only15px collision veto catches39/52 suspects but also flags270 predictions already within15px of manual labels. Stress LOO retains only34/219body observations, of which2 are paws. Reject blanket-veto route; do not call lower retained error a general improvement.',
            'Actual LP top5 local heatmap peaks,7x7NMS,relative strength>=.1,3seed union:GT-oracle candidate coverage734/785 vs medianensemble669/785 within15px.41/61medianensemble errors>50px have a candidate within15px. This is candidate recall, NOT an achieved decoder improvement; GT never selects training labels.',
            'For each seed, top3 and top5 at relative strength>=.1 give identical recovery counts in this cohort; bounded top3 candidate geometry is reasonable to test next without increasing teacher epochs.',
            'Visual example AZR2L2f1429camDk16:seed17top1 coincides with reference,seed31/73second peaks at relative strength.11/.44 coincide while highest peaks point elsewhere. Preserve hypotheses rather than averaging/collapsing all modes.',
            'AZR2L2f584camDk18 has a suspected manual semantic error. Keep original labels and all existing metrics intact, report reviewflag separately. Some annotation-relative errors must not be asserted as model-only failures.'
        ],
        next_actions=[
            'Implement diagnostic multi-hypothesis fixed-camera matching using cached validation-only heatmap candidates (no new training):top3 per seed,relative strength>=.1,merge candidates within10px,retain at most3spatial candidates per keypoint/view. Set rules before results.',
            'Compare primary-peak-only and candidate-aware reconstruction on identical sparse pilot/stress cohorts with true held-camera removal. Minimum3nondegenerate positive-depth support views; keep ambiguity/missing, never choose using held-view/manual2D.',
            'Report paw identity failures, severe-error counts, pointcoverage, per-keypoint breakdown and matched retained cohorts. Any joint one-to-one assignment is a separate ablation; do not conflate it with candidate recall.',
            'Do not auto-fix manual labels or silently removeSEM01 from historical metrics. Keep new manual-review list alongside existing24excluded groups; no suspect group becomes pseudo-GT.',
            'Only if candidate route has real evidence of improvement run continuous inference/temporal evaluation; no more epochs or threshold sweeps now. Stop new work at2026-10-02T07:21:52UTC.'
        ],new_review_candidates=review,locked_holdout_opened=False,original_labels_or_calibration_changed=False,pseudo_gt_released=False)
    path=OUT/'heartbeat_20261002_0135_decision.json';path.write_text(json.dumps(report,indent=2))
    write_state(stage='paw_identity_and_heatmap_candidates_audited',pid=None,active_run=None,mode='idle_between_heartbeats',job_deadline_utc=None,
        latest_decision=str(path),latest_result=str(OUT/'paw_heatmap_peaks_v1/summary.json'),next_action=report['next_actions'][0],pseudo_gt_released=False)
    print(path)

if __name__=='__main__':main()
