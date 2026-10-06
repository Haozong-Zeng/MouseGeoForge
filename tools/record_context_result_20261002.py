"""Freeze context pilot interpretation and the matched repeat-center training control."""
import datetime as dt
import json
import os
from run_teacher_cycle_dlc_20261001 import OUT, write_state

run = OUT / 'lp_context_seed17_kl'
result = json.loads((run / 'summary.json').read_text())
review = json.loads((run / 'review/summary.json').read_text())
state = json.loads((OUT / 'state.json').read_text())
assert state['active_run'] == 'lp_contextoff_seed17_kl'
os.kill(state['pid'], 0)
smoke = json.loads((OUT / 'lp_contextoff_seed17_kl/smoke.json').read_text())
assert smoke['repeated_center_verified'] and smoke['passed']
report = dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(), completed='30epoch native LP MHCRNN seed17; best checkpoint epoch24',
    cohort='Identical288developmentimages/1361manualpoints/770pawpoints; SEM01excluded; two reused sessions, not independent test',
    metrics={name: value for name, value in result['models'].items()}, paired=review['paired'],
    conclusions=[
        'Native context-selected improves over old staticseed17 mean32.30->24.44/P9079.60->47.74px; not a clean temporal ablation because architecture, batch size and optimizer step count differ.',
        'Versus static3seedmedian mean24.12->24.44 worsens slightly; P9054.42->47.74 improves. Pawmean19.81->21.05 worsens; no global dominance.',
        'At SAMEcontexttrainedcheckpoint realneighbors vs repeatedcenter selectedmean26.05->24.44/P9053.67->47.74; 53points improve>5px but69worsen>5px. MFmean25.03->24.38 only. Benefit concentrated in a few large-error cases.',
        'Learning curve largely plateaus after learning-rate decrease; bestepoch24, final30 not better. Do not extend this unchanged run.',
        'Need matched training control to distinguish contextual information from capacity/training settings. No new pseudo-GT or strong3Drelease justified.'
    ],
    visual_inspection=[
        dict(file='context_gain_01_AZR2L2_000259_camE.jpg', observation='Real-neighbor MF corrects point16 misplaced into distant background by MF-repeat/SF; existing staticteachers already locate it correctly.'),
        dict(file='context_gain_02_FR2_000129_camC.jpg', observation='Real-neighbor MF corrects frontpaw16 from hindpaw region; staticteachers already correct. Back-surface errors persist.'),
        dict(file='context_harm_02_FR2_000454_camF.jpg', observation='Real-neighbor MF and selected put forepaw15 near dorsal point6; MF-repeat closer; old staticmodels locate visiblepaw correctly.'),
        dict(file='context_harm_01_FR2_001559_camF.jpg', observation='At videoendpoint MF confuses forepaw15 with oppositehindpaw18; nativeconfidence-selection uses SF and avoids mostdamage. Endpoint clamp may matter; do not infer general temporalgain.'),
        dict(file='selected_failure_01_FR2_001039_camE.jpg', observation='Contextmodel heads place hindpaw17 on opposite side; staticteachers correct.'),
        dict(file='selected_failure_05_AZR2L2_001169_camF.jpg', observation='Persistent hindpaw18 identity error across contextheads; realcontext does not resolve it. Staticmodels also wrong, at a differentpaw.')
    ],
    next_experiment=dict(run='lp_contextoff_seed17_kl', pid=state['pid'], smoke=smoke,
        architecture='Identical nativeHeatmapTrackerMHCRNN/ImageNetResNet50', seed=17, epochs_ceiling=30, batch_size=4,
        optimizer_steps_if_complete=7830, input_wh=[704,768],
        replacement='After identical native dataset augmentation, repeat center image five times in TRAIN and checkpoint-selection VALIDATION.',
        intended_comparison='ON selected/mf realneighbors vs OFF selected_repeat/mf_repeat; identical labels, source split, loss, schedule, batch and modelcapacity.',
        caveat='Single-seed matchedpilot, not statistically established generalization. Both checkpoint selections use the same predefined native validation metric on intended inputs.',
        job_deadline_utc=state['job_deadline_utc']),
    next_action='Inspect matched-control curve and compare intended-input outputs when complete; no concurrent GPUjob. If meaningful benefit, validate continuous motion and geometry before any pseudolabel recommendation.',
    visual_directory='C:/Users/55395/Downloads/MouseGeoForge-TeacherTemporal-20261001/lp_context_seed17_kl',
    original_labels_modified=False, locked_holdout_opened=False, pseudo_gt_released=False)
p = OUT / 'heartbeat_20261002_0455_decision.json'
p.write_text(json.dumps(report, indent=2))
write_state(latest_decision=str(p), next_action=report['next_action'], cpu_review_pid=None,
            latest_result=str(run / 'review/summary.json'), pseudo_gt_released=False)
print(p)
