"""Fixed-development RGB context-control comparison, no model promotion."""
import datetime as dt,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];CYCLE=ROOT/'output/core23_spatial_cycle_20261004'

def main():
    names=['live_view_decoder_balanced2400_audit_v1','live_view_context_paired800_audit_v1','live_view_context_mixed800_audit_v1'];rows=[];reference=None;parity={}
    for name in names:
        result=json.loads((CYCLE/name/'results.json').read_text());data=result['reports'];parity[name]={k:result.get(k) for k in ['parity_passed','single_rgb_parity_cases','max_single_rgb_delta']}
        if reference is None:reference=data
        assert [x['task'] for x in data]==[x['task'] for x in reference]
        assert [x['detector'] for x in data]==[x['detector'] for x in reference]
        for kind in ['real','synthetic']:
            for variant in ['single','paired','all']:
                group=[x for x in data if x['task']['record']['kind']==kind and x['task']['variant']==variant];points=sum(x['gated_pose']['total_points'] for x in group)
                row=dict(run=name,kind=kind,variant=variant,points=points,tp=sum(x['detector']['tp'] for x in group),gt=sum(x['detector']['gt'] for x in group),predictions=sum(x['detector']['predictions'] for x in group))
                for metric in ['pck16','pck32']:row['gated_'+metric]=sum(x['gated_pose'][metric+'_count'] for x in group)/max(points,1)
                for metric in ['spatial/all','spatial/anchor','spatial/detected_only']:
                    n=sum(x['metrics'].get(metric,{}).get('n',0) for x in group)
                    row[metric]=sum(x['metrics'].get(metric,{}).get('mean',0)*x['metrics'].get(metric,{}).get('n',0) for x in group)/max(n,1)
                rows.append(row);print(json.dumps(row))
    report=dict(rows=rows,parity=parity,fixed_detector_identical=True,scope='Seed17 same36development tasks. GTbox association only AFTER inference. GatedPCK includes missed detections, not duplicate-FP penalties; report detector counts alongside. Not independent test, no production promotion.')
    (CYCLE/'context_live_comparison_v1.json').write_text(json.dumps(report,indent=2)+'\n')
    dest=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-Core23-Spatial-20261004');(dest/'context_live_comparison_v1.json').write_text(json.dumps(report,indent=2)+'\n')
    decision=dict(
        conclusion='Retain context-mixed pose as an experimental candidate, not production. Single-view conditional error improves consistently, but full-view accuracy is not uniformly better.',
        numerical='Across3seeds singlebottomfeet59.15/66.51/64.22px vs paired-only79.91/83.16/79.73; single syntheticdual17.57/17.57/17.36 vs20.03/20.12/20.38. SidecamD single worsens1.57-3.13px. FullRGBseed17 realanchor single52.25 vs68.90px, paired45.57 vs52.20, all54.54 vs54.66. Real all-view gatedPCK32.509 vs.520. No all-view win claimed.',
        visual=['Cached00camA single/paired stillshow feet on head/abdomen and duplicatedqueries; some pointsmove towardpaws but othersmoveaway.','Cached03camD single retains nose/footregion but small regression consistentwith numbers.','Cached08syntheticcamC two overlapping mice remain distinguishable queries; any simple boxNMS could suppress a realanimal.','Live00single and02all inspected: wrong front/hind semantic locations remain; duplicates are NOT repaired by pose-only training.'],
        next='Next bounded investigation: detector also trained only with paired context. Reuse accepted same-image single/paired features to prepare ROI detection inputs (derive originalboxes/logits from immutable original box/objectness heads on saved instance tokens; ROI from native memory through exactfp16/48x48 path). Verify against saved percontext detector predictions before training. Matched paired-only vs50percent single-context detector continuation, same image schedules/seed/checkpoint/steps, freeze mixed800pose+encoder+instance decoder. Start800steps3seeds, report real single/paired recall,precision,duplicate counts, syntheticdual retention. Evaluate downstream ALL-query pose with NEW PREDICTED boxes and queryassociation outside forward; do not count detectorgain as posegain or silently force singlemouse count/NMS. If detection change fails, stop similar probes and summarize architectural limits rather than blindly extending. FullRGB1/2/all gate and visualreview required before any promotion. Keep harddeadline and protectedlabels/lockedholdouts.',
        production_changed=False)
    (CYCLE/'context_live_decision_v1.json').write_text(json.dumps(decision,indent=2)+'\n');(dest/'context_live_decision_v1.json').write_text(json.dumps(decision,indent=2)+'\n')
    plan=json.loads((CYCLE/'plan.json').read_text());plan['next_command']=decision['next'];(CYCLE/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='view_context_control_rgb_review_complete',active_job=None,active_pid=None,updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),next=decision['next'])
    if 'view_context_control_rgb_and_visual_review' not in state['completed']:state['completed'].append('view_context_control_rgb_and_visual_review')
    (CYCLE/'state.json').write_text(json.dumps(state,indent=2)+'\n')

if __name__=='__main__':main()
