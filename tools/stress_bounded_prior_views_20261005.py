"""Recompute prediction-only geometry under genuine camera perturbations."""
import datetime as dt,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from mousegeoforge.models.prior_candidate_selector import PriorCandidateSelector
from mousegeoforge.candidate_selector_features import build_selector_inputs,KP
from audit_independent_unary_20261005 import metrics
from run_depth_pose_ablation import fingerprint,write_json
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';BASE=CYCLE/'bounded_prior_candidate_pilot_v1';OUT=CYCLE/'bounded_prior_viewstress_v1'


def subset_data(data,indices):
    return {k:([v[i] for i in indices] if k=='metadata' else {kk:vv[indices] for kk,vv in v.items()} if k=='inputs' else v[indices] if isinstance(v,torch.Tensor) else v) for k,v in data.items()}


@torch.no_grad()
def main():
    assert not OUT.exists();OUT.mkdir();torch.set_num_threads(2)
    d=torch.load(CYCLE/'local_candidate_selector_pilot_v1/validation_inputs.pt',map_location='cpu',weights_only=False)
    models={};hashes={}
    for arm in ['unary','geometry']:
        path=BASE/f'{arm}_latest.pt';models[arm]=PriorCandidateSelector(arm).eval();models[arm].load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['state']);hashes[arm]=fingerprint(path)
    manifest=json.loads((CYCLE/'cached_local_candidate_inputs_v1/manifest.json').read_text());expected={r['path']:r['sha256'] for r in manifest['records']}
    scene_rows={}
    for i,m in enumerate(d['metadata']):scene_rows.setdefault(m['source_path'],[]).append(i)
    modes=['full','reverse','duplicate','single_target','paired_target','three_target','all_invalid']+['without_'+str(v) for v in range(7)]
    storage={}
    for mode in modes:
        for arm in models:
            storage[mode+'/'+arm]=dict(xy=d['base_xy'].clone(),active=torch.zeros_like(d['inputs']['eligible']),choice=torch.zeros_like(d['inputs']['eligible'],dtype=torch.long),eligible=torch.zeros_like(d['inputs']['eligible']),observed=torch.zeros(len(d['base_xy']),dtype=torch.bool))
    duplicate_delta=0.;feature_delta=0.;off_exact=True
    for path,rows in scene_rows.items():
        assert fingerprint(ROOT/path)==expected[path];s=torch.load(ROOT/path,map_location='cpu',weights_only=False)
        assert s['role']=='validation' and s['record'].get('session') not in ['GR2','UR2','UR3']
        p=s['predictions'];views=list(s['physical_views']);v=len(views)
        by_local={d['metadata'][i]['source_view_index']:i for i in rows};assert len(by_local)==v
        def forward(indices,invalid=False):
            pred={k:x[indices] for k,x in p.items()}
            features=build_selector_inputs(pred,s['cameras']['projections'][indices],[views[i] for i in indices],s['pixels'],view_valid=np.zeros(len(indices),bool) if invalid else None)
            outputs={arm:model(**features,enabled=True) for arm,model in models.items()}
            for model in models.values():assert model(**features)['xy'] is features['baseline_xy']
            return features,outputs
        for mode in modes:
            if mode.endswith('_target'):
                count={'single_target':1,'paired_target':2,'three_target':3}[mode]
                calls=[]
                for target in range(v):
                    others=sorted([i for i in range(v) if i!=target],key=lambda i:views[i]);calls.append(([target]+others[:count-1],[0]))
            else:
                indices=list(range(v))
                if mode=='reverse':indices.reverse()
                elif mode=='duplicate':indices=indices+indices
                elif mode.startswith('without_'):indices=[i for i in indices if views[i]!=int(mode.split('_')[1])]
                calls=[(indices,list(range(len(indices))))] if indices else []
            for indices,positions in calls:
                features,outputs=forward(indices,mode=='all_invalid')
                if mode=='full':
                    for pos,local in enumerate(indices):
                        row=by_local[local]
                        for k,value in features.items():
                            old=d['inputs'][k][row].float();now=value[pos].float()
                            if k=='local':now=now.half().float()
                            feature_delta=max(feature_delta,float((old-now).abs().max()))
                for arm,out in outputs.items():
                    if mode in ['single_target','paired_target','all_invalid']:
                        torch.testing.assert_close(out['xy'],features['baseline_xy'],atol=0,rtol=0);assert not out['active'].any()
                    if mode=='duplicate':duplicate_delta=max(duplicate_delta,float((out['xy'][:v]-out['xy'][v:]).abs().max()))
                    for pos in positions:
                        row=by_local[indices[pos]];store=storage[mode+'/'+arm]
                        store['xy'][row,:,KP]=out['xy'][pos];store['active'][row]=out['active'][pos];store['choice'][row]=out['choice'][pos];store['eligible'][row]=out['eligible'][pos];store['observed'][row]=True
    golden=np.load(BASE/'predictions.npz');tests=dict(default_off_exact=off_exact,duplicate_copies_difference=duplicate_delta,feature_rebuild_difference=feature_delta,V1_V2_allinvalid_fallback_exact=True)
    for arm in models:
        full=storage['full/'+arm]
        tests[arm+'_golden_replay']=float(np.max(np.abs(full['xy'].numpy()-golden[arm+'_validation_matched'])))
        assert tests[arm+'_golden_replay']<2e-5
        for mode in ['reverse','duplicate']:
            change=float((storage[mode+'/'+arm]['xy']-full['xy']).abs().max());tests[arm+'_'+mode+'_difference']=change;assert change==0
    # Forward completed for every condition before all label-based metrics.
    results=[];arrays={}
    for tag,stored in storage.items():
        observed=stored['observed'].nonzero().flatten().tolist();ds=subset_data(d,observed)
        values,points=metrics(ds,stored['xy'][observed],stored['active'][observed],stored['choice'][observed])
        baseline,_=metrics(ds,ds['base_xy'],torch.zeros_like(stored['active'][observed]),torch.zeros_like(stored['choice'][observed]))
        results.append(dict(tag=tag,observed_images=len(observed),eligible_points=int(stored['eligible'].sum()),metrics=values,baseline_same_observed=baseline))
        for p in points:p['row']=observed[p['row']]
        write_json(OUT/(tag.replace('/','_')+'_points.json'),points)
        for key,value in stored.items():arrays[tag.replace('/','_')+'_'+key]=value.numpy()
        if tag.endswith('geometry'):print(tag,[(k,round(v['mean_error'],2),round(baseline[k]['mean_error'],2)) for k,v in values.items() if k in ['real/reliable','real/view0/feet']],flush=True)
    np.savez_compressed(OUT/'predictions.npz',**arrays)
    write_json(OUT/'results.json',dict(results=results,tests=tests,checkpoint_sha256=hashes,source_sha256={p:expected[p] for p in scene_rows},script_sha256=fingerprint(Path(__file__)),utc=dt.datetime.now(dt.timezone.utc).isoformat(),protocol='No training. Rebuild geometry and eligibility from actual subset; NEVER slice old full-context geometry. Target-centric V1/V2/V3 retains each target and first 0/1/2 other physical IDs sorted numerically, predeclared no label selection. Leave-camera-out removes that camera as target and support; report baseline on same observed target set. Exact duplicates count once, reversed order cannot change outputs. All invalid preserves baseline. Unary and geometry matched >=2 distinct others eligibility, independent-unary gate not used. GT only metrics after forward. Reused small development validation, not external test.',production_changed=False))


if __name__=='__main__':main()
