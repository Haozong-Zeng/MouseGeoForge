"""Record continuous diagnostic outcome and freeze new training for cycle closeout."""
import datetime as dt
import json
from run_teacher_cycle_dlc_20261001 import OUT, write_state

r = json.loads((OUT / 'context_clip_geometry_review_v1/summary.json').read_text())
assert r['parity_checks'] == 168 and r['parity_max_px'] == 0
assert r['all']['common_loo_n'] == r['all']['n_manual'] == 384
report = dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(), result=r,
    conclusions=[
        'Four frozen clips,204interiorframes each, all24camera-clips.168manualpredictionparities across ON/OFF are exact0px: no video/PNG encoding confound.',
        'Sparsemanual raw2D mean37.04->28.65/P90123.08->55.36px. Same384observations, no acceptance filtering.',
        'Trueheldviewout uniform4px robustgeometry mean69.23->62.15/P90201.77->145.35px. All384observations retained for both; severeerrors83->67.',
        'PawLOO mean76.74->55.78/P90231.39->140.79px; severeerrors50->29/211. Positive relative benefit but inadequate absolute3Dproxyquality.',
        'Observed2D error is much lower than heldviewreconstruction error. Multi-view semantic conflicts/calibration-label limitations persist; do not replace good manual2D with these reprojections.',
        'Pawframejumps>50px decrease in all4clips, but this is only a discontinuity diagnostic. Dense motion/amplitude/lagGT is absent; cannot certify that motion is moreaccurate or unattenuated.',
        'This single-teacher geometry uses UNIFORM4px scales, not unavailable ensembleuncertainty; same previous solver settings but NOT numerically equivalent to earlier static3seed weightedfusion.',
        'No strong3D or pseudo-GT released. Do not increase unchanged teacher epochs, tune geometry thresholds further, or initiate newtraining in remainingcycle.'
    ],
    inspected_snapshots=[
        dict(file='pilot_AZR2L2_frame000194.jpg', observation='Both ON/OFF still confuse hindpaw17 acrosssidecamF; invisible/unannotated dorsalpoint6 can appear onbackground. MissingGT isnot certifiedinvisible; do notcreate negative labels fromthis.'),
        dict(file='stress1_AZR2L2_frame001169.jpg', observation='camA pawlocalization looks plausible but camF still maps right_hindpaw18 toopposite/front region. Crossviewsemanticconflict persists despite smoother sequence.'),
        dict(file='pilot_FR2_frame000779.jpg', observation='BottomviewcamA still has hindpaw17/18identityconfusion in ON; sidecamF bodypoints and nose improve inplaces but do not resolve all pawidentity.')
    ],
    next_action='Prepare final concise experiment ledger and review handoff with exact metrics, failedroutes, eligibility masks/provenance requirements and next-data priorities. Verify all job PIDs/GPU idle. At07:21:52UTC stop cycle and pause automation; no more experiments after deadline.',
    visual_directory='C:/Users/55395/Downloads/MouseGeoForge-TeacherTemporal-20261001/context_clip_geometry_review_v1',
    videos='Four H264MP4 files, OFFleft/ONright, camAupper/camFlower, half-speed26fps from52fps source. Raw2Dpredictions, not reprojections.',
    original_labels_modified=False, locked_holdout_opened=False, pseudo_gt_released=False)
p = OUT / 'heartbeat_20261002_0625_decision.json'
p.write_text(json.dumps(report, indent=2))
write_state(stage='context_route_diagnostic_complete_pending_cycle_closeout', pid=None, active_run=None, mode='idle_between_heartbeats',
    job_deadline_utc=None, latest_decision=str(p), latest_result=str(OUT / 'context_clip_geometry_review_v1/summary.json'),
    next_action=report['next_action'], pseudo_gt_released=False)
print(p)
