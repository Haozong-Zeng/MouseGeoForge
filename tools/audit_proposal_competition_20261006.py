"""Read-only saved-prediction audit; no new inference, training or threshold search."""
import sys,json,hashlib,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np
from scipy.optimize import linear_sum_assignment
from mousegeoforge.models.proposal_instance_grouping import decode_proposal_grouped,proposal_grouping_loss
C=ROOT/'output/local_grouping_cycle_20261005';RUN=C/'proposal_grouping_pilot640_v1';OUT=C/'proposal_competition_audit_v2'
def read(p):return torch.load(p,weights_only=False,map_location='cpu')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert json.loads((RUN/'status.json').read_text())['stage']=='complete'
    OUT.mkdir(exist_ok=False);torch.set_num_threads(2)
    inp=ROOT/'output/singleview_fusion_cycle_20261005/mixed_local_fulltrain_rgb_inputs_v1'
    rows={r['index']:r for r in read(inp/'evaluation_only.pt')};sup={r['index']:r for r in read(C/'instance_supervision_audit_v1/evaluation_instance_targets.pt')}
    manifests={r['index']:r for r in json.loads((inp/'inference_manifest.json').read_text())}
    reports={};hashes={}
    for arm in ['independent','within_instance']:
        p=RUN/f'{arm}_step0640_predictions.pt';hashes[arm]=sha(p);pred=read(p);details=[]
        for i,item in pred.items():
            r,s=rows[i],sup[i];assert r['metadata']['stage'] in ['fit','calibration'];assert r['metadata'].get('source') not in ['FR2','AZR2L2','GR2','UR2','UR3']
            out=item['output'];d=item['decoded'];replay=decode_proposal_grouped(out)
            for key,value in d.items():
                if 'probability' in key:torch.testing.assert_close(replay[key],value,atol=1e-7,rtol=1e-6)
                else:assert torch.equal(replay[key],value),(arm,i,key)
            hw=manifests[i]['image_hw'];scale=torch.tensor([hw[1],hw[0]]);diag=float(scale.float().norm());active=d['instance_active'].nonzero().flatten();gt=s['instance_valid'].nonzero().flatten()
            loss=proposal_grouping_loss(out,r['target'],r['eligible'],s['instance_valid'],s['mask_centroid'],s['centroid_valid'],hw)
            pos=torch.zeros_like(out['center_available']);pos[loss['matched_queries']]=True
            prob=out['object_logits'].sigmoid();near=torch.cdist(out['centers']*scale,s['mask_centroid'][gt]*scale).min(1).values<=64
            matches={}
            if len(active) and len(gt):
                dist=torch.cdist(d['centers'][active]*scale,s['mask_centroid'][gt]*scale);a,b=linear_sum_assignment(dist.numpy())
                matches={int(gt[y]):(int(active[x]),float(dist[x,y])) for x,y in zip(a,b)}
            points=[]
            for g,k in r['eligible'].nonzero().tolist():
                final=raw=diag;changed=False;competitor=False
                if g in matches and matches[g][1]<=64:
                    q=matches[g][0];candidate=int(out['assignment_logits'][q,k].argmax());chosen=int(d['chosen'][q,k]);changed=candidate!=chosen
                    if d['accepted'][q,k]:final=min(diag,float(((d['xy'][q,k]-r['target'][g,k])*scale).norm()))
                    if candidate<out['candidate_xy'].shape[1] and out['candidate_available'][k,candidate]:
                        raw=min(diag,float(((out['candidate_xy'][k,candidate]-r['target'][g,k])*scale).norm()))
                        competitor=any(int(q2)!=q and int(d['chosen'][q2,k])==candidate for q2 in active)
                points.append(dict(keypoint=k,final=final,independent_argmax=raw,changed=changed,competitor_took_argmax=competitor))
            details.append(dict(index=i,metadata=r['metadata'],points=points,gt_count=len(gt),active_count=len(active),near_active=int((d['instance_active']&near).sum()),far_active=int((d['instance_active']&~near).sum()),missing_proposals=float(loss['missing_center_proposals']),positive_scores=prob[pos].tolist(),negative_scores=prob[out['center_available']&~pos].tolist(),near_negative_scores=prob[out['center_available']&~pos&near].tolist(),objectness_loss=float(loss['objectness'])))
        summary={}
        for kind in ['real','real_bottom_feet','single','dual']:
            subset=[x for x in details if x['metadata']['stage']=='calibration' and (x['metadata']['kind']==kind or kind=='real_bottom_feet' and x['metadata']['kind']=='real' and x['metadata']['physical_bottom'])]
            pts=[p for x in subset for p in x['points'] if kind!='real_bottom_feet' or p['keypoint']>0]
            values=lambda key:[v for x in subset for v in x[key]]
            avg=lambda v:float(np.mean(v)) if len(v) else None
            summary[kind]=dict(images=len(subset),points=len(pts),positive_score=avg(values('positive_scores')),negative_score=avg(values('negative_scores')),near_negative_score=avg(values('near_negative_scores')),near_active=sum(x['near_active'] for x in subset),far_active=sum(x['far_active'] for x in subset),missing_proposals=sum(x['missing_proposals'] for x in subset),joint64_pck16=avg([p['final']<=16 for p in pts]),no_exclusivity_diagnostic_pck16=avg([p['independent_argmax']<=16 for p in pts]),changed_points=sum(p['changed'] for p in pts),competitor_took_argmax=sum(p['competitor_took_argmax'] for p in pts),exclusivity_harmed_pck=sum(p['independent_argmax']<=16<p['final'] for p in pts),exclusivity_helped_pck=sum(p['final']<=16<p['independent_argmax'] for p in pts))
        reports[arm]=dict(summary=summary,details=details);assert sha(p)==hashes[arm]
    result=dict(source_hashes=hashes,replay_geometry_and_discrete_exact=True,probability_replay='CPU/CUDA sigmoid tolerance atol1e-7 rtol1e-6',model_inference_count=0,optimizer_steps=0,caveat='No-exclusivity argmax retains SAME predicted active instances and posthoccenterGTmatches. Duplicates remain; this is an error decomposition, NOT a proposed deployable decoder or fair improvement. Near64 is only geometric, not proof of duplicate identity.',arms=reports)
    (OUT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>重复候选与足点竞争：只读诊断</h1><p>固定两组640步输出，解码精确重放；不训练、不改阈值。独立argmax仅用于诊断重复实例争抢足点，不能作为部署收益。</p><pre>'+json.dumps({a:x['summary'] for a,x in reports.items()},indent=2)+'</pre>',encoding='utf-8')
    (OUT/'status.json').write_text(json.dumps(dict(stage='complete')))
    dest=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-LocalGrouping-20261005')/OUT.name;shutil.copytree(OUT,dest)
    print(json.dumps({a:x['summary'] for a,x in reports.items()}))
if __name__=='__main__':main()
