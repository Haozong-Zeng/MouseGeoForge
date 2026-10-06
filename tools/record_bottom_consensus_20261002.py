"""Record inspected evidence and the next bounded diagnostic, not new training labels."""
import datetime as dt
import json
import numpy as np
from bottom_anchor_common_20261002 import ROOT,OUT,PRIOR,BASE,budget,write_state
from bottom_anchor_consensus_20261002 import fit_pair,tests


def main():
    _,_,check=budget(180);check()
    src=OUT/'multiview_consensus_v1'
    summary=json.loads((src/'summary.json').read_text())
    # Real-data invariant: the omitted view is removed at the public API boundary.
    m=json.loads((PRIOR/'lp_context_dataset_v1/manifest.json').read_text())
    with np.load(PRIOR/'lp_context_seed17_kl/predictions_source_pixels.npz') as z:
        entries=[(p,m['entries'][int(idx)]) for p,idx in enumerate(z['manifest_indices']) if m['entries'][int(idx)]['session']=='FR2' and m['entries'][int(idx)]['frame']==454]
        entries.sort(key=lambda pe:pe[1]['camera'])
        cams=[e['camera'] for _,e in entries];uv=np.stack([z['uv_selected'][p,3:5] for p,_ in entries]);anchor=z['gt'][entries[cams.index('A')][0],3:5]
    cal=json.loads((BASE/'calibration/FR2/rectified_calibration.json').read_text())
    held=cams.index('F');keep=[i for i in range(len(cams)) if i!=held];subcams=[cams[i] for i in keep];subcal={c:cal[c] for c in subcams}
    first=fit_pair(subcal,subcams,anchor,uv[keep],True);uv[held]=1e9
    second=fit_pair(subcal,subcams,anchor,uv[keep],True)
    assert first==second
    for g in json.loads((src/'groups.json').read_text()):
        for variants in [g['full'],*g['loo'].values()]:
            # full is name->pairlist, whereas loo is method->camera->pairlist.
            if variants is g['full']:
                results=[r for pairs in variants.values() for r in pairs]
            else:
                results=[r for pairs in variants.values() for r in pairs]
            for r in results:
                if not r['accepted']:continue
                mapping=np.asarray(r['mapping']);shared=(mapping>=0).all(1)
                assert np.all(mapping[shared,0]!=mapping[shared,1])
                assert min(r['side_support'])>=2
    tests_out=dict(synthetic=tests(),real_FR2_454_camF_held_prediction_perturbation_invariant=True,
                   every_accepted_pair_candidate_exclusive=True,every_accepted_paw_at_least_two_side_supporters=True)
    (src/'verification.json').write_text(json.dumps(tests_out,indent=2))
    decision=dict(time_utc=dt.datetime.now(dt.timezone.utc).isoformat(),experiment='multiview_consensus_v1',
        outcomes=summary['methods'],paired=summary['paired'],tests=tests_out,
        actual_visual_inspection=[
            dict(file='01_manual_gain_FR2_000454_camF.jpg',finding='A mislabeled dorsal-body forepaw prediction is rejected; the other forepaw candidate is assigned to visible paw15. Manual input direct5.2px/held-view6.5px. This is a valid local gain, not proof for all paws.'),
            dict(file='03_manual_harm_FR2_000389_camC.jpg',finding='Closely spaced forepaw candidates receive a harmful identity change. Accepted geometry still does not resolve nearby/crossing paws; direct16.6px and held-view19.7px for15.'),
            dict(file='05_predicted_harm_FR2_001104_camB.jpg',finding='Bottom teacher forepaw15/16 visibly reversed relative to human bottom labels. Predicted-anchor variant propagates wrong identity to otherwise correct side predictions:15 direct153.5px and held-view146.3px. Manual-anchor samecase6.5px and10.3px.'),
            dict(file='07_manual_LOO_failure_FR2_001039_camB.jpg',finding='Even human bottom reference leaves an accepted held-view forepaw16 projection on the opposite side near another paw,142.1px from side annotation. Do not assume geometry consensus or either individual manual label is definitive; needs multiview annotation/candidate audit, no automatic label edits.'),
            dict(file='09_unknown_raw_failure_AZR2L2_001169_camF.jpg',finding='Raw hindpaw18 is located near forepaw15, not merely swapped with hindpaw17. Consensus abstains. Bilateral permutations cannot invent the missing correct hindpaw candidate.')],
        conclusion='Do not promote this method or start temporal label propagation yet. Manual flexible direct2D gains exist, but paired trueLOO fixed13.3923 vs flexible13.4202px shows no gain; predicted anchors are a semantic failure source. About23percent coverage and stronger rejection cannot be reported as full-cohort improvement.',
        next_action=[
            'Audit frozen bottom predictions against bottom manual labels EVALUATION-ONLY: per-pair identity vs swapped distance, per-session errors, and actual bottom contact sheets of mismatches. This diagnoses anchor reliability, does not feed manual identities into predicted-anchor inference.',
            'Evaluate candidate availability upper bounds on side labeled points: original-ID vs either same fore/hind pair vs allfour current predictions. Label oracle strictly diagnostic, never a deployable matching method. Report cross-forehind confusions and absent candidate cases; no pseudo-label export.',
            'For rare accepted severe manual-anchor LOO failures, inspect all available real views and manual ray consistency as a SEPARATE label/calibration quality audit. SideGT remains absent from the correction pipeline; do not relabel, change calibrations or optimize thresholds against it.',
            'Use these results to choose a targeted route: better bottom identity teacher/candidate heatmaps vs anchored short-term tracking. Do not blindly add epochs or sweep geometry thresholds; keep holdouts locked.',
            'Only test temporal propagation if sparse identity reliability is credible, with restart/unknown at gaps or occlusion and genuine held-side testing.'
        ],original_labels_modified=False,pseudo_gt_released=False,locked_holdout_opened=False,
        limitation='Reused development videos and one frozen teacher seed; no independent test or 3D GT. The bottom ray is treated exactly in this diagnostic and may be overconstrained for imperfect annotations.')
    path=OUT/'heartbeat_20261002_0805_decision.json';path.write_text(json.dumps(decision,indent=2))
    write_state(stage='multiview_consensus_visually_reviewed',pid=None,active_run=None,mode='idle_between_heartbeats',job_deadline_utc=None,job_log=None,
        latest_result=str(src/'summary.json'),latest_decision=str(path),next_action=decision['next_action'],pseudo_gt_released=False)
    print(json.dumps(tests_out,indent=2))


if __name__=='__main__':main()
