"""Cache replay of independent unary fallback plus supported geometry override."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from mousegeoforge.models.prior_candidate_selector import PriorCandidateSelector
from mousegeoforge.models.bounded_candidate_fusion import BoundedCandidateFusion
from audit_independent_unary_20261005 import metrics
from run_depth_pose_ablation import fingerprint,write_json
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';HEADS=CYCLE/'bounded_prior_candidate_pilot_v1';OUT=CYCLE/'composite_candidate_cache_audit_v1'


def load_wrapper():
    heads=[]
    for arm in ['unary','geometry']:
        h=PriorCandidateSelector(arm);h.load_state_dict(torch.load(HEADS/f'{arm}_latest.pt',map_location='cpu',weights_only=False)['state']);heads.append(h)
    return BoundedCandidateFusion(*heads).eval()


@torch.no_grad()
def main():
    assert not OUT.exists();OUT.mkdir();torch.set_num_threads(2);model=load_wrapper()
    data=torch.load(CYCLE/'local_candidate_selector_pilot_v1/validation_inputs.pt',map_location='cpu',weights_only=False)
    source_rows={}
    for i,m in enumerate(data['metadata']):source_rows.setdefault(m['source_path'],[]).append(i)
    manifest=json.loads((CYCLE/'cached_local_candidate_inputs_v1/manifest.json').read_text());hashes={r['path']:r['sha256'] for r in manifest['records']}
    modes=dict(baseline=(False,False),local=(True,False),fusion_only=(False,True),composite=(True,True))
    preds={name:data['base_xy'].clone() for name in modes};changed={name:torch.zeros_like(data['inputs']['eligible']) for name in modes};covered={name:dict(local=0,fusion=0) for name in modes};tests={name:0. for name in ['permutation','duplicate','V1_local_fallback','V2_local_fallback','all_invalid']}
    for path,rows in source_rows.items():
        assert fingerprint(ROOT/path)==hashes[path];s=torch.load(ROOT/path,map_location='cpu',weights_only=False);p=s['predictions'];v=len(s['physical_views']);by_local={data['metadata'][i]['source_view_index']:i for i in rows}
        def call(ids,mode='composite',**kw):
            local,fusion=modes[mode]
            return model({k:x[ids] for k,x in p.items()},s['cameras']['projections'][ids],[s['physical_views'][i] for i in ids],s['pixels'],local_enabled=local,fusion_enabled=fusion,**kw)
        outputs={name:call(list(range(v)),name) for name in modes}
        for name,out in outputs.items():
            for local,row in by_local.items():preds[name][row]=out['xy'][local];changed[name][row]=out['changed'][local]
            covered[name]['local']+=int(out['local_eligible'].sum());covered[name]['fusion']+=int(out['fusion_eligible'].sum())
        reverse=call(list(reversed(range(v))))['xy'].flip(0);dup=call(list(range(v))*2)['xy'][:v]
        tests['permutation']=max(tests['permutation'],float((reverse-outputs['composite']['xy']).abs().max()));tests['duplicate']=max(tests['duplicate'],float((dup-outputs['composite']['xy']).abs().max()))
        for t in range(v):
            ids=[t]+[i for i in range(v) if i!=t][:1]
            for count in [1,2]:
                sub=ids[:count];out=call(sub)['xy'][0];tests[f'V{count}_local_fallback']=max(tests[f'V{count}_local_fallback'],float((out-outputs['local']['xy'][t]).abs().max()))
        bad=call(list(range(v)),view_valid=[False]*v)['xy'];tests['all_invalid']=max(tests['all_invalid'],float((bad-p['xy']).abs().max()))
    assert all(value==0 for value in tests.values()),tests
    golden=np.load(HEADS/'predictions.npz');tests['local_prior_audit_replay']=float(np.max(np.abs(preds['local'].numpy()-golden['unary_validation_independent'])));tests['fusion_prior_audit_replay']=float(np.max(np.abs(preds['fusion_only'].numpy()-golden['geometry_validation_matched'])));assert max(tests.values())<2e-5
    results=[]
    for name,xy in preds.items():
        m,points=metrics(data,xy,changed[name],torch.zeros_like(changed[name],dtype=torch.long))
        for p in points:p.pop('choice',None)
        write_json(OUT/f'{name}_points.json',points);results.append(dict(mode=name,metrics=m,coverage=covered[name]));print(name,{k:round(v['mean_error'],3) for k,v in m.items() if k in ['real/reliable','real/view0/feet','synthetic/reliable']},flush=True)
    np.savez_compressed(OUT/'predictions.npz',**{k:v.numpy() for k,v in preds.items()});write_json(OUT/'results.json',dict(results=results,tests=tests,head_sha256={arm:fingerprint(HEADS/f'{arm}_latest.pt') for arm in ['unary','geometry']},source_sha256={p:hashes[p] for p in source_rows},module_sha256=fingerprint(ROOT/'mousegeoforge/models/bounded_candidate_fusion.py'),script_sha256=fingerprint(Path(__file__)),protocol='Fixed prior seed17 heads, no training. local independent unique-conf-query; geometry overrides supported points INCLUDING explicit candidate0 reset, else keeps local. Both opt-ins defaultFalse. Allqueries and originalboxes unchanged, only reliable5points. This is restricted singlecohort developer diagnostic, uniqueconfquery not provenmulti-instance association. No new labels or threshold selection.',production_changed=False))


if __name__=='__main__':main()
