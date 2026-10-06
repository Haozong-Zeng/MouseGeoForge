"""GT-permutation oracle is diagnosis only, never a prediction or label repair."""
import sys,json,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import torch,numpy as np
C=ROOT/'output/singleview_fusion_cycle_20261005';I=C/'mixed_local_rgb_inputs_v1'
def read(p):return torch.load(p,weights_only=False,map_location='cpu')
def main():
    rotation4000='--rotation4000' in sys.argv;rotation='--rotation' in sys.argv or rotation4000;run=C/('mixed_rotation4000_v1' if rotation4000 else ('mixed_rotation_control_v1' if rotation else 'mixed_real_pool_control_v1'));out=C/('mixed_rotation4000_identity_audit_v1' if rotation4000 else ('mixed_rotation_identity_audit_v1' if rotation else 'mixed_pool_identity_audit_v1'));out.mkdir(exist_ok=True)
    arms=['native_rgb','quarter_turn'] if rotation else ['same_pool','expanded_real'];rows=[r for r in read(I/'evaluation_only.pt') if r['metadata']['stage']=='calibration' and r['metadata']['kind']=='real'];records=[]
    for arm in arms:
        pp=read(run/(arm+'_predictions.pt'))
        for r in rows:
            i=r['index'];p=pp[i];qq,kk=r['eligible'][:,1:].nonzero(as_tuple=True);kk=kk+1
            if not len(kk):continue
            assert qq.unique().numel()==1;gt=r['target'][qq,kk]*r['pixels'];top1=p['xy'][1:,0]*r['pixels'];dist=torch.cdist(gt,top1);best=min(itertools.permutations(range(4),len(kk)),key=lambda perm:float(dist[torch.arange(len(kk)),list(perm)].sum()));identity=dist[torch.arange(len(kk)),kk-1];oracle=dist[torch.arange(len(kk)),list(best)];all24=(p['xy'][1:]*r['pixels']).reshape(-1,2);union=torch.cdist(gt,all24).min(1).values
            for n,k in enumerate(kk.tolist()):records.append(dict(arm=arm,index=i,bottom=bool(r['metadata']['physical_bottom']),point=[1,15,16,17,18][k],typed_top1=float(identity[n]),untyped_one_to_one_top1=float(oracle[n]),untyped_any24=float(union[n])))
    summary={}
    for arm in arms:
        summary[arm]={}
        for view,b in [('all',None),('bottom',True),('other',False)]:
            rr=[r for r in records if r['arm']==arm and (b is None or r['bottom']==b)];summary[arm][view]=dict(n=len(rr),**{k:dict(mean=float(np.mean([r[k] for r in rr])),pck16=float(np.mean([r[k]<=16 for r in rr]))) for k in ['typed_top1','untyped_one_to_one_top1','untyped_any24']})
    result=dict(summary=summary,records=records,caveats='Real CAL known feet only. Oracle uses GT to assign foot-type channels after inference; not deployable, not a relabeling or improvement claim. One-to-one top1 uses identical four candidate positions but relaxes semantics. Any24 further increases candidates and permits reuse, hence strongly optimistic and confounded by candidate count. Duplicated positions are not magically distinct physical feet. Unknown visibility unchanged.')
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
