"""Freeze matched single-seed evidence and bounded continuous diagnostic decision."""
import datetime as dt
import json
import os
from run_teacher_cycle_dlc_20261001 import OUT, write_state

r = json.loads((OUT / 'lp_context_matched_ablation_v1/summary.json').read_text())
state = json.loads((OUT / 'state.json').read_text())
assert state['active_run'] == 'context_ablation_clips_v1'
os.kill(state['pid'], 0)
report = dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(), matched_result=r,
    decision='No additional training. Validate existing matched ON/OFF teachers on four frozen continuous PNG-exact clips and then trueLOO geometry if time allows.',
    interpretation=[
        'Matched seed17 ON selected mean24.44 vs OFF29.81px; P9047.74 vs63.91; severe pawerrors47 vs66 out of770. Positive pilot evidence for context under matched architecture/steps.',
        'Per-point median improvement is negative(-.254px), with215improvements>5px and209worsenings>5px. Benefit mainly reduces large-error tail; not universal pointwise gain.',
        'ON selected still does not dominate three-seed staticmedian: mean24.44 vs24.12, pawmean21.05 vs19.81. No production teacher replacement justified.',
        'Bootstrap groups48multiviewsourceframes, conditional on2reuseddevelopmentvideos; no independent animal/session or multiseed generalization inference.',
        'No dense manual3D/motionGT. Continuous trajectories can expose discontinuities/persistentidentityerrors but cannot certify lag/amplitudeaccuracy from smoothness alone.'
    ],
    visual_checks=[
        dict(file='on_gain_01_FR2_000649_camE.jpg', observation='ON reduces hindpaw17 error308.8->111.4px but remains wrong: improvement is not successful localization. Staticmedian99.8px also wrong.'),
        dict(file='on_harm_01_AZR2L2_001104_camE.jpg', observation='OFF correctlylocates forepaw16; ON assigns it nearhindpaw17. Pawmean9.0->131.1px. Staticmedian7.9px remainscorrect.'),
        dict(file='on_failure_02_AZR2L2_000194_camF.jpg', observation='Allthree alternatives confuse hindpaw17 across sides. Shared confidenterror not solvedbycontext.')
    ],
    new_job=dict(pid=state['pid'], deadline_utc=state['job_deadline_utc'], framework='ActualLP MHCRNN checkpoints, no retraining',
        data='Four previouslyfrozen PNG-exact FFV1developmentclips;ON realneighbors/OFFrepeatcenter;all24cameraclips.',
        retained='204interiorframes per208frameclip;trim2bothends so no artificialcontextpadding.',
        quality_checks='Originalsourcecoordinate sparsemanualpredictionparity<.1px;no excluded sourceframe in anyinputwindow;sameCPUnormalization.',
        budget='25minutes/globaldeadline, sharedGPUlock, progresspercamera/clip, STOPguard.',
        output=str(OUT / 'context_ablation_clips_v1')),
    next_action='When continuous inference completes, generate paired videos and sparse-manual metrics; use unchanged robustgeometry settings for cameraLOO if runtime permits. No new architecture search or threshold sweeps in remainingcycle.',
    visual_directory='C:/Users/55395/Downloads/MouseGeoForge-TeacherTemporal-20261001/lp_context_matched_ablation_v1',
    locked_holdout_opened=False, original_labels_modified=False, pseudo_gt_released=False)
p = OUT / 'heartbeat_20261002_0555_decision.json'
p.write_text(json.dumps(report, indent=2))
write_state(latest_decision=str(p), next_action=report['next_action'],
            project_config=str(OUT / 'context_ablation_clips_v1/settings.json'), checkpoint=None, cpu_matched_review_pid=None)
print(p)
