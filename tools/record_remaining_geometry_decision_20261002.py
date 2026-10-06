"""Record broader frozen validation and bounded temporal-candidate next step."""
import datetime as dt,json
from run_teacher_cycle_dlc_20261001 import OUT,write_state

def main():
    broad=json.loads((OUT/'remaining_geometry_frozen_v1/summary.json').read_text())
    rows=json.loads((OUT/'remaining_geometry_frozen_v1/observations.json').read_text())
    cross=json.loads((OUT/'cross_framework_agreement_v1/summary.json').read_text())
    transitions=dict(common=sum(r['joint'] is not None and r['merged_top3'] is not None for r in rows),
        improved_over1px=sum(r['joint'] is not None and r['merged_top3'] is not None and r['joint']<r['merged_top3']-1 for r in rows),
        worsened_over1px=sum(r['joint'] is not None and r['merged_top3'] is not None and r['joint']>r['merged_top3']+1 for r in rows),
        severe_joint=[{k:r[k] for k in ['session','frame','camera','id','joint','manual_only_loo']} for r in rows if r['joint'] is not None and r['joint']>100])
    report=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),frozen_remaining_validation=broad,cross_framework=cross,transitions=transitions,
        findings=[
            '34predeclared additional development groups,559manual pawobservations,2same videos. Fixed joint scheme retains304(54.38%),median12.80/P9035.51px,9errors>100px. Top1retains251(44.90%),median12.82/P9036.65px,11errors>100px. Cohorts differ; not interchangeable accuracy claims.',
            'On232points accepted by both top1/joint, P90identical29.916px. Joint has1error>100 vs top1has2; most typical-point accuracy unchanged.',
            'Broader data now contains real corrections, unlike earlier14groups: visually inspected FR2f649camFk18 top3error252.34->joint6.50px. Also inspected AZR2L2f519camAk18, all methods489.71px and project to contralateral hindpaw. Keep both evidence types.',
            'Naive independenttop3remains rejected:377retained,P90210.10px,66errors>100. Joint reduces but does not solve identity ambiguity.',
            'DLC70cross-framework agreement is not a better2Drisk ranking than LP3seedspread at matched50%/75%coverage. Bothgates retain526/770paws and still14errors>100px. Do not invest in denseDLCagreementgating without better evidence.',
            'No method qualifies for bulk strong3Dsupervision. All errors measured against manual2D; manualreference uncertainty remains. No checkpoint/labels/cameras modified.'
        ],
        next_actions=[
            'Bounded follow-up: cache actual LP top3heatmap candidates for existing pending-free208frame pilot+stressclips,3seeds, original sourcepixel units, exact CPU torchvision normalization and<.1px parity guard. OneGPUjob at a time, bounded deadline. Preserve all oldartifacts.',
            'After candidates exist, compare frame-independent joint hypotheses with temporal identity selection on IDENTICAL observations. Fixedcamera geometry and held-view removal throughout. Temporal route must not invent points when no>=3viewhypothesis exists and must never bridge excluded gaps.',
            'Predeclare a small finite-state/beam temporal hypothesis selection with explicit unknown state and motioncost before measuring results; keep this separate from EKSposition smoothing. Evaluate sparsemanualLOO,coverage,severeerrors,amplitude and lag. Do not call smoothing or lower speed improvedaccuracy.',
            'If temporalidentity selection merely preserves wrongconsistent tracks or loses fastmotion, recordfailure and do not extend dataexport. No more gate sweeps or teacher epochs currently justified.',
            'Keep original24exclusions plus newSEM01out of newvideo windows; keep lockedholdoutclosed. Experiment deadline07:21:52UTC remains, no pseudoGTreleased.'
        ],
        locked_holdout_opened=False,original_labels_or_calibration_changed=False,pseudo_gt_released=False,
        visual_review='C:/Users/55395/Downloads/MouseGeoForge-TeacherTemporal-20261001/remaining_geometry_frozen_v1_review_v2')
    p=OUT/'heartbeat_20261002_0235_decision.json';p.write_text(json.dumps(report,indent=2))
    write_state(stage='broader_geometry_and_crossframework_audited',pid=None,active_run=None,mode='idle_between_heartbeats',job_deadline_utc=None,
        latest_decision=str(p),latest_result=str(OUT/'remaining_geometry_frozen_v1/summary.json'),next_action=report['next_actions'][0],pseudo_gt_released=False)
    print(json.dumps(transitions,indent=2))

if __name__=='__main__':main()
