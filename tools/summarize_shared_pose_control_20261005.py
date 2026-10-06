"""Shared-token adaptation control and exact repeatability gate."""
import datetime as dt,json,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];CYCLE=ROOT/'output/core23_spatial_cycle_20261004';BASE=CYCLE/'shared_token_pose_control_v1';DEST=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-Core23-Spatial-20261004')

def main():
    deltas={}
    with np.load(BASE/'predictions.npz') as a,np.load(CYCLE/'view_order_control_v1/predictions.npz') as b:
        np.testing.assert_array_equal(a['image_indices'],b['image_indices'])
        for seed in [17,29,43]:
            for context in ['paired','single']:
                delta=float(np.max(np.abs(a[f'original_seed{seed}_{context}']-b[f'fixed_seed{seed}_{context}'])));assert delta<2e-5;deltas[f'{seed}/{context}']=delta
    results=json.loads((BASE/'results.json').read_text())['results'];table=[]
    for row in results:
        m=row['metrics'];initial=json.loads((BASE/f'{row["arm"]}_seed{row["seed"]}_initial.json').read_text())['metrics'];table.append(dict(seed=row['seed'],arm=row['arm'],single_real=m['single/real_validation/reliable']['error_mean'],paired_real=m['paired/real_validation/reliable']['error_mean'],shared_real=m['shared/real_validation/reliable']['error_mean'],shared_bottom=m['shared/real_validation/camA/feet']['error_mean'],shared_dual=m['shared/synthetic_validation/dual']['error_mean'],initial_shared_real=initial['shared/real_validation/reliable']['error_mean'],initial_shared_bottom=initial['shared/real_validation/camA/feet']['error_mean'],train_shared_real=m['shared/real_train/reliable']['error_mean']))
    rgb={}
    for arm in ['original','shared']:
        path=CYCLE/f'live_view_shared_pose_{arm}800_audit_v1/results.json'
        if path.exists():
            live=json.loads(path.read_text());assert live['parity_passed'] and live['geometry_preserved_by_identity'] and live['shared_rgb_parity_cases']==24 and live['single_rgb_parity_cases']==12
            rgb[arm]=dict(single_delta=live['max_single_rgb_delta'],shared_delta=live['max_shared_rgb_delta'])
    next_action=('Shared-token posecontrol COMPLETE800steps3seeds andcachedrawcamA/camD/dual reviewed. Originalarm replayagainstold fixed-ordercontrol checked <=2e-5 inpaired/single sixseedcontexts; see shared_pose_control_decision_v1.json. Cachegain consistent butsmall, no promotion. FullRGBchain LAUNCHED session24339: validate_core23_live_views_20261004.py --shared-pose-control original then shared. Outputs live_view_shared_pose_original800_audit_v1 and live_view_shared_pose_shared800_audit_v1. Both fullRGBarmsuse shared-token2Droute withsame frozenmixed800detector, differentmatchedtrainedposeheads. Original-token multiviewgeometry output retainedbyidentity; replaceonlysix2Dfields. Auxiliarypose/curveoutputs unreconciled andNOTjointtraining-valid; extraencoderpass notcomputematched. Newhelper shared_token_pose_probe4tests passed. CheckactualPID/status/failures beforeanything; no duplicateGPUjobs. FullRGB checks originalpairedcachegoldens/defaultoff, single12 andsharedpaired24actualRGBcachegoldens, tokenrestore everypass. Oncompletion run review_view_order_audit_20261005.py --shared-pose-control original and --shared-pose-control shared ONCE (flagsIMPLEMENTED). Compare samephysicalcamA+syntheticexamples, detectorTP/predictions/gatedPCK, conditionalanchorerror andprediction-onlyorderdisplacement. Compare also earlier independent-viewbaseline with explicitdifferenthead/detectorweights caveat; forcausalcomparison useoriginalvs sharedtrainedheads samefrozendetector/sharedinference. Then decide whether sharedtokenROI detector adaptation is justified asaseparatematchedstage; ifno robustbenefit stophead-only extensions andpreparehonesthandoff/researchnext architecture. Noautopromotion/labelchanges/lockedaccess. Trainingstop10:18:46UTC hardstop10:48:46UTC unchanged.')
    report=dict(updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),repeatability_deltas=deltas,cache=table,rgb_parity=rgb,visual=['00_real_camA_shared.jpg: oneconfidentqueryinthissample, sharedadaptationmovesfrontpawpredictions closer but somefeetstillcluster; not an anatomicalsolution.','03_real_camD_shared.jpg: pawpredictions stillclose tobody, smallshift notclearvisualwin.','08_synthetic_camC_shared.jpg: distantdual queries remain, subtleshifts only; relyonaggregateandfullRGB.'],production_changed=False,next=next_action)
    out=CYCLE/'shared_pose_control_decision_v1.json';out.write_text(json.dumps(report,indent=2)+'\n');shutil.copy2(out,DEST/out.name)
    plan=json.loads((CYCLE/'plan.json').read_text());plan['next_command']=next_action;(CYCLE/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    state=json.loads((CYCLE/'state.json').read_text());state.update(next=next_action,updated_utc=report['updated_utc']);entry='shared_pose_cached_visual_and_repeatability_review'
    if entry not in state['completed']:state['completed'].append(entry)
    (CYCLE/'state.json').write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(dict(repeatability=deltas,cache=table,rgb_parity=rgb),indent=2))

if __name__=='__main__':main()
