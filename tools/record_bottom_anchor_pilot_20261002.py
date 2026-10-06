"""Freeze pilot result and visual observations for next scheduled experiment."""
import datetime as dt
import json
from bottom_anchor_common_20261002 import OUT,write_state

r=json.loads((OUT/'epipolar_pilot_v1/summary.json').read_text())
report=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),result=r,
    conclusions=[
        '48developmentgroups,95completebottommanualpawpairs;579sideviewmanualpoints. Originalannotations unchanged; bottommanual isinput, neverevaluation.',
        'Manualanchor conservative exchange changes9annotatedpoints;5improve>5px,1worsens>5px. Side2Dmean21.30->19.45/P9022.23->20.56; >100pxerrors36->31. Smallpositive result, not robustglobal correction.',
        'On common575sideobservations, manualanchor NOswap LOOmean25.56/P9056.69 vs conservative24.75/53.56. Most benefit relative topredictedanchorbaseline29.68/70.70 comesfrom manualanchor positions, not permutation alone.',
        'Unconditional epipolarargmin harmful:189changedlabeledpoints,16improve>5px but58worsen>5px; side2Dmean23.77 vs21.30. Near-epiline does not establish identity or depth.',
        'Predictedanchor conservative18changedpoints:6improve>5px,8worsen>5px; side2Dmean21.01 nearlyunchanged; commonLOOmean29.87 worse than29.68baseline. Manualanchor benefit must not be presented as autonomous correction benefit.',
        'Conservative swaps may relocate an unlabeled badpoint onto another unknown identity. Only visible side-point improvement is certified against manual2D; allfourpaw correctness and3DGT are NOT established.'
    ],
    vision=[
        dict(file='bottom_manual_audit.jpg',review='Inspected12actualcamAframes spanningbothsessions/orientations. IDs15/17 and16/18 appear anatomicallyside-consistent on sampled ventralposes; no obvious global mirror swap. This is NOT individual certification of all48anchorframes.'),
        dict(file='02_conservative_gain_FR2_000454_camF.jpg',review='Rawforepaw15 on dorsalbody; exchanging15/16 moves15 to actual visibleforepaw nearhuman15. Otherforepaw16 remains onbody without sideGT; do not certify that hiddenpoint.'),
        dict(file='05_conservative_harm_FR2_000454_camB.jpg',review='Hindpawpredictions overlap nearvisiblepaw; exchange yields ~5px regression. Geometry weak when candidates nearlycoincident.'),
        dict(file='08_argmin_harm_AZR2L2_000259_camE.jpg',review='Ungatedswap moves correctlylocalized16 onto distantbackgroundprediction originally15. Conservative retainscorrect16. Demonstratesepiline/depthambiguity and occludedpointdanger.')
    ],
    next_experiment=[
        'Build bottom-anchored multi-viewcandidate consensus with explicit front/hindpairpermutations; candidatecoordinates from currentteacher only, no sidemanualinputs.',
        'Test positive-depth and realmulti-view support (bottom plus>=2sideviews), and pairwise uniqueness of sourcecandidate assignments. Permitunknown whenambiguous. Do not tighten/loosenepipolarthresholds tooptimize thissamecohort.',
        'For trueLOO discardheldsideview BEFORE candidatebuilding, permutation selection, confidence scoring and fitting; testinvariance byperturbingheldpredictions.',
        'Compare same-inputmanualanchor no-swap vs newassignment on pairedretainedobservations andreportcoverage, pluspredictedanchor variant separately. Avoid reporting selectedsubset improvements as fullcohortgain.',
        'Visualize largestgains/regressions/unknowns withbottom+sideoriginals. Preserve originalmanualcoordinates and markalloutputs diagnostic.',
        'Only ifthis has credible sparsebenefit, testtemporallypropagated bottomidentity on contiguous existingPNG-exact clips; no cross-gapfill, no lockholdout use.'
    ],
    automatic_cycle='ACTIVE every30min; maxdeadline2026-10-02T17:33:55UTC; stop earlier ondecisive conclusion orneededuserinput',
    locked_holdout_opened=False,pseudo_gt_released=False,original_labels_modified=False)
p=OUT/'heartbeat_20261002_0733_decision.json';p.write_text(json.dumps(report,indent=2))
write_state(stage='epipolar_pilot_visually_reviewed',pid=None,active_run=None,mode='idle_between_heartbeats',
    latest_decision=str(p),latest_result=str(OUT/'epipolar_pilot_v1/summary.json'),
    next_action=report['next_experiment'],pseudo_gt_released=False)
print(p)
