"""Verify physical-camera invariance and record next bounded architecture test."""
import datetime as dt,json,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];CYCLE=ROOT/'output/core23_spatial_cycle_20261004';OUT=CYCLE/'review_independent_pose_order_v1';DEST=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-Core23-Spatial-20261004')

def main():
    live=json.loads((CYCLE/'live_view_independent_pose_audit_v1/results.json').read_text());base=json.loads((CYCLE/'live_view_reference_detector_random_ref800_audit_v1/results.json').read_text());review=json.loads((OUT/'results.json').read_text())
    assert live['parity_passed'] and live['single_rgb_parity_cases']==12 and live['geometry_preserved_by_identity']
    previous={(r['task']['record']['scene'],r['task']['variant']):r for r in base['reports']};singles={r['task']['record']['scene']:r for r in live['reports'] if r['task']['variant']=='single'}
    fields=['instance_keypoint_xy','instance_boxes_cxcywh','instance_objectness_logits'];checks=0;maximum=0.
    for row in live['reports']:
        r=row['task']['record'];anchor=row['task']['anchor_view'];v=r['views'].index(anchor)
        with np.load(ROOT/row['path']) as a,np.load(ROOT/singles[r['scene']]['path']) as b:
            for k in fields:
                delta=float(np.max(np.abs(a[k][0,v]-b[k][0,0])));assert delta==0;maximum=max(maximum,delta)
        if row['task']['variant']=='single':
            with np.load(ROOT/row['path']) as a,np.load(ROOT/previous[(r['scene'],'single')]['path']) as b:
                for k in fields:np.testing.assert_array_equal(a[k],b[k])
            checks+=1
    shifts=[r['mean_point_shift_px'] for r in review['changes'] if r['mean_point_shift_px'] is not None];assert max(shifts)==0
    next_action=('Independent-view2D diagnostic COMPLETE and reviewed; no GPUjobs expected, verify actualPID/lock. Files live_view_independent_pose_audit_v1, review_independent_pose_order_v1 and independent_pose_decision_v1.json. Sameweights exactsingle12tasks and60anchorcontexts; original/reverse shifts0. RealallTP42/48pred51 vs contextual34/48pred70; gatedPCK.554vs.417. But realpairedconditionalanchor47.16vs42.85 worse, bottomfeetstillwrong. Extraencoderpasses and no2Dcrossviewfusion; notproduction. Next distinct bounded architecture experiment: shared-special-token adaptation of the2Dbranch while originalmultiviewgeometry stays unchanged. Prior untrainedtokenprobe was symmetric but inaccurate; test whether headadaptation recoversaccuracy, not anotherloss/stepsextension. Prepare actualshared-first-token paired TRAIN1223/VAL112 caches using sameimages/manifests as originalpair, originalorder, sameoriginalencoderweights and runtime-only shared_first_view_tokens context; restoretokensaftereachforward; verify singleviewunchanged/originalcachedgoldens/sourcehashes/lockedexclusions. Do NOT reverseimages in thiscache. Existing prepare_reversed_view_context_20261005.py offers extractionstructure but needs explicitsharedmodeandcorrectnames/contract. Then matched originaltokens vs sharedtokens heads,50percent actualsingle/50percent paired BOTHarms, same acceptedimage schedules,3seeds800steps fromidentical perseed view_context_control_v1/mixed800 pose and detection_view_context_control_v1/mixed800detector. Treat pose and detector stages separately toattributeeffects; full138queries, frozenencoder/semanticquery/coarse andnoGTcrop. Can firsttestposewithfixeddetector if explicitcovariates, then detector; assess2DfullRGB1/2/allbothorders and geometryretention. Before implementing decide if remainingtime/complexity warrants this versus earlyhandoff; no needfill16h. If pursuing, writeclearplan/protocolfirst and avoidrelabeling oldorder-control runs. No externaldata/paidmodels/newhumantracing; harddeadlineunchanged training10:18:46UTC hardstop10:48:46UTC.')
    decision=dict(updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),single_unchanged_cases=checks,anchor_contexts=60,max_anchor_context_delta=maximum,max_order_displacement_px=max(shifts),summary=review['summary'],conclusion='Independent-view2D is a useful stable baseline, not the final multiviewmodel. It improves fullview coverage versus sameweightscontextual branch, but discards2Dcrossviewinformation and doesnotfixanatomy.',visual=['00_real_order.jpg: allcolumnsidentical asrequired; duplicateq1/q4 andfootclusteringremain.','03_synthetic_order.jpg: samebox/feet acrossallcontexts; remainingpawerrorvisible.'],limitations='Geometry values preserved byidentity, not validated asaccurate3D. Six2Dfields only; auxiliarypose/curvefields unreconciled. Extra sequentialencoderpasses notcomputematched. Independentquery IDs areNOTcrossviewidentities. Developmentvalidationonly, singlefullRGBseed17.',production_changed=False,next=next_action)
    path=CYCLE/'independent_pose_decision_v1.json';path.write_text(json.dumps(decision,indent=2)+'\n');shutil.copy2(path,DEST/path.name)
    review.update(visual_review_pending=False,visual_review=decision['visual'],replay_scope='Separate independent-probe audit:12single outputs and60physicalanchorcontexts exact. Oldmultiviewoutput replay intentionallynotapplicable.',max_original_order_delta=None);(OUT/'results.json').write_text(json.dumps(review,indent=2)+'\n');shutil.copy2(OUT/'results.json',DEST/OUT.name/'results.json')
    plan=json.loads((CYCLE/'plan.json').read_text());plan['next_command']=next_action;(CYCLE/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    state=json.loads((CYCLE/'state.json').read_text());assert state['active_job'] is None;state.update(stage='independent_pose_diagnostic_reviewed',next=next_action,updated_utc=decision['updated_utc'])
    entry='independent_pose_exact_context_invariance_and_visual_review'
    if entry not in state['completed']:state['completed'].append(entry)
    (CYCLE/'state.json').write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(dict(single_exact=checks,anchor_max_delta=maximum,order_shift=max(shifts),next_stage='bounded sharedtoken2D adaptation feasibility')))

if __name__=='__main__':main()
