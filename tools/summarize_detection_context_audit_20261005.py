"""Summarize detector and downstream pose separately, retain non-wins."""
import argparse,datetime as dt,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];CYCLE=ROOT/'output/core23_spatial_cycle_20261004';DEST=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-Core23-Spatial-20261004')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--live',action='store_true');args=parser.parse_args()
    data=json.loads((CYCLE/'detection_context_pose_audit_v1/results.json').read_text());cached=[]
    for r in data['results']:
        cached.append(dict(seed=r['seed'],context=r['context'],arm=r['arm'],real=r['metrics']['real/reliable'],bottom=r['metrics']['real/camA/feet'],synthetic_dual=r['metrics']['synthetic/dual'],detector_real=r['detector']['real'],detector_dual=r['detector']['synthetic/dual']))
    report=dict(cached=cached,parity=data['max_baseline_golden_delta'],scope='Frozen mixed800 pose across detectorarms baseline/paired/mixed. GT association conditional metrics distinct from detector and gatedPCK. No production promotion.')
    for r in cached:print(json.dumps(dict(seed=r['seed'],context=r['context'],arm=r['arm'],real_error=r['real']['error_mean'],real_pck32=r['real']['gated_pck32'],real_tp=r['detector_real']['tp'],real_pred=r['detector_real']['predictions'],dual_error=r['synthetic_dual']['error_mean'],dual_pck32=r['synthetic_dual']['gated_pck32'],dual_tp=r['detector_dual']['tp'],real_query_switches=r['detector_real']['query_switches'])))
    if args.live:
        live=[];parity={};reference=None
        for name in ['live_view_context_mixed800_audit_v1','live_view_detector_context_paired800_audit_v1','live_view_detector_context_mixed800_audit_v1']:
            current=json.loads((CYCLE/name/'results.json').read_text());rows=current['reports'];parity[name]={k:current.get(k) for k in ['parity_passed','single_rgb_parity_cases','max_single_rgb_delta']}
            if reference is None:reference=rows
            assert [x['task'] for x in rows]==[x['task'] for x in reference]
            for kind in ['real','synthetic']:
                for variant in ['single','paired','all']:
                    group=[r for r in rows if r['task']['record']['kind']==kind and r['task']['variant']==variant];points=sum(r['gated_pose']['total_points'] for r in group)
                    row=dict(run=name,kind=kind,variant=variant,tp=sum(r['detector']['tp'] for r in group),gt=sum(r['detector']['gt'] for r in group),predictions=sum(r['detector']['predictions'] for r in group),points=points)
                    for metric in ['pck16','pck32']:row['gated_'+metric]=sum(r['gated_pose'][metric+'_count'] for r in group)/max(points,1)
                    for metric in ['spatial/all','spatial/anchor','spatial/detected_only']:
                        count=sum(r['metrics'].get(metric,{}).get('n',0) for r in group);row[metric]=sum(r['metrics'].get(metric,{}).get('mean',0)*r['metrics'].get(metric,{}).get('n',0) for r in group)/max(count,1)
                    live.append(row);print(json.dumps(row))
        report.update(live=live,live_parity=parity)
        decision=dict(
            conclusion='Keep only as experimental single-view candidate, not an unconditional replacement. Mixed context improves single-view detector coverage substantially, but fullRGB multiview regression prevents promotion.',
            evidence='Cached3seeds single syntheticdual detected45/45/46 of64 vs original26/29/26; gatedPCK32 improves. Frozen pose conditional error changes little: detector change does NOT solve foot semantics. FullRGBseed17 real single predictions16to10 with6/8TP retained; real all-viewTP41to38 and gatedPCK32.509to.463. Synthetic fullRGB singleTP4/6to5/6 but paired9/12to8/12 and all22/32to21/32.',
            visual_review=['Cached00realcamA single: extra yellowbox disappears, primarypawpredictions stillnearhead/abdomen.','Cached08syntheticcamC single: bothoverlappingmice remain distinct predictions; no NMS applied.','Live00real single: mixeddetector removes extraquery, footpositions stillwrong.','Live02real all: duplicatedquery persists, no generalmultiview fix.'],
            next='Order diagnostic is now running: validate_core23_live_views_20261004.py --detector-context mixed --permutation-audit, output live_view_order_mixed800_audit_v1. Check actualPID/status, no duplicate job. This adds paired_reversed/all_reversed on exactly the same RGB sets, same physical anchor and frozen models. Local aggregator slice_expand_and_flatten uses separate first-frame/other camera+registertokens; hypothesis of reference-order sensitivity, not yet proven cause. Once complete compare outputs by physical camera IDs, not array positions or persistent query IDs; verify unchanged-order tasks against prior live_view_detector_context_mixed800_audit_v1. Report matched-anchor and aggregate GT-assisted metrics, detector counts, per-view prediction changes after prediction-only set association; draw raw confidence queries on original images. Do not use GT to choose an optimal vieworder or promote a model. Decide on TRAIN-only order/context augmentation only after this audit, no blind moresteps. Preserve deadline, no lockedholdouts or labelchanges.',
            production_changed=False)
        (CYCLE/'detection_context_decision_v1.json').write_text(json.dumps(decision,indent=2)+'\n');(DEST/'detection_context_decision_v1.json').write_text(json.dumps(decision,indent=2)+'\n')
        plan=json.loads((CYCLE/'plan.json').read_text());plan['next_command']=decision['next'];(CYCLE/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
        state=json.loads((CYCLE/'state.json').read_text());state.update(next=decision['next'],updated_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        if 'detection_context_rgb_and_visual_review' not in state['completed']:state['completed'].append('detection_context_rgb_and_visual_review')
        (CYCLE/'state.json').write_text(json.dumps(state,indent=2)+'\n')
    name='detection_context_comparison_v1.json';(CYCLE/name).write_text(json.dumps(report,indent=2)+'\n');(DEST/name).write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
