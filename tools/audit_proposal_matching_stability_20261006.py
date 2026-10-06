"""Frozen FIT-only quarter-turn/assignment audit. No optimizer or GT forward."""
import os
os.environ['DLClight']='True'
import sys,json,random,fcntl,datetime as dt,time,traceback,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import torch,numpy as np
from preflight_mixed_local_backbone_20261005 import build_backbone,load_rgb,sha,digest
from train_mixed_local_backbone_20261005 import buffers_digest,atomic_save
from train_dense_center_warmup_20261005 import read,write,now
from mousegeoforge.models.mixed_local_evidence import MixedLocalEvidence
from mousegeoforge.models.dense_instance_centers import DenseInstanceCenters
from mousegeoforge.models.proposal_instance_grouping import ProposalInstanceGrouping,proposal_grouping_loss,decode_proposal_grouped
from mousegeoforge.models.local_instance_grouping import rotate_mask_centroids
from mousegeoforge.models.quarter_turn_evidence import quarter_turn
C=ROOT/'output/local_grouping_cycle_20261005';RUN=C/'proposal_grouping_continue3200_v1';OUT=C/'proposal_matching_stability_audit_v1'
INPUT=ROOT/'output/singleview_fusion_cycle_20261005/mixed_local_fulltrain_rgb_inputs_v1'

def assignment(out,r,s,hw,xy,eligible,centroids):
    loss=proposal_grouping_loss(out,xy,eligible,s['instance_valid'].cuda(),centroids,s['centroid_valid'].cuda(),hw)
    return {int(g):int(q) for q,g in zip(loss['matched_queries'],loss['matched_targets'])}

@torch.no_grad()
def main():
    plan=json.loads((C/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'));assert now()<deadline
    assert json.loads((RUN/'status.json').read_text())['stage']=='complete'
    lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    OUT.mkdir(exist_ok=False);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    state=json.loads((C/'state.json').read_text());state.update(stage='proposal_matching_audit_running',active_job=str(OUT.relative_to(ROOT)),active_pid=os.getpid(),updated_utc=now().isoformat());write(C/'state.json',state)
    def status(stage,**kw):write(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=now().isoformat(),**kw))
    status('loading');rows={r['index']:r for r in read(INPUT/'evaluation_only.pt')};sup_path=C/'instance_supervision_audit_v1/evaluation_instance_targets.pt';sup_hash=sha(sup_path);sup={r['index']:r for r in read(sup_path)}
    manifest={r['index']:r for r in json.loads((INPUT/'inference_manifest.json').read_text())};rng=random.Random(61);samples=[];domains=['real_bottom','real_other','single','dual']
    for domain in domains:
        pool=sorted(i for i,r in rows.items() if r['metadata']['stage']=='fit' and ((domain.startswith('real_') and r['metadata']['kind']=='real' and bool(r['metadata']['physical_bottom'])==(domain=='real_bottom')) or r['metadata']['kind']==domain))
        samples.extend(dict(index=i,domain=domain) for i in rng.sample(pool,12))
    assert all(rows[s['index']]['metadata'].get('source') not in ['FR2','AZR2L2','GR2','UR2','UR3'] for s in samples)
    protocol=dict(samples=samples,seed=61,turns=[0,1,2,3],source_step=3200,arms=['independent','within_instance'],optimizer_steps=0,code_sha256=sha(Path(__file__)),supervision_sha256=sup_hash,purpose='48FIT each4rotations; noCAL/DEV/locked images. Native-coordinate positive proposal drift, nearest/second distance margin, learned positive vs negative scores; GT +/-2px axial perturbation ONLY a loss matching sensitivity diagnostic, not label edits.',limitations='Quarter-turn changes model predictions as well as matching. Nearest candidate geometric correspondence is not persistent identity. Small FIT audit cannot prove generalization cause. No adaptive thresholds or inference GT.',deadline=plan['training_stop_utc'])
    write(OUT/'protocol.json',protocol);start=time.monotonic();results={};hashes={}
    for arm in protocol['arms']:
        source=RUN/f'{arm}_step3200.pt';hashes[arm]=sha(source);cp=read(source)
        backbone,_=build_backbone();model=MixedLocalEvidence(backbone,adapt_backbone=True).cuda();model.backbone.load_state_dict(cp['backbone']);model.readout.load_state_dict(cp['readout']);model.eval()
        center=DenseInstanceCenters().cuda();center.load_state_dict(cp['center']);center.eval();group=ProposalInstanceGrouping(interaction=arm=='within_instance').cuda();group.load_state_dict(cp['grouping']);group.eval();del cp
        before=[digest(m) for m in [model,center,group]]+[buffers_digest(backbone)];details=[];saved={}
        for n,sample in enumerate(samples,1):
            i=sample['index'];r=rows[i];s=sup[i];native_hw=manifest[i]['image_hw'];x=load_rgb(manifest[i]).cuda();target=r['target'].cuda();eligible=r['eligible'].cuda();gt=s['mask_centroid'].cuda();valid=s['centroid_valid'].cuda();native_scale=gt.new_tensor([native_hw[1],native_hw[0]])
            assert bool(s['detection_annotation_complete']) and torch.equal(s['centroid_valid'],s['instance_valid'])
            rotations=[];native_selected={};native_centers=None;saved[i]={}
            for k in range(4):
                if now()>=deadline:raise TimeoutError('Audit cutoff')
                rx,xy,hw=quarter_turn(x,target,eligible,native_hw,k);cg=rotate_mask_centroids(gt,valid,k);scale=cg.new_tensor([hw[1],hw[0]])
                evidence=model(rx,hw);out=group(evidence,center(evidence,hw),hw);dec=decode_proposal_grouped(out);match=assignment(out,r,s,hw,xy,eligible,cg)
                pred_native=rotate_mask_centroids(out['centers'],out['center_available'],(4-k)%4)
                if k==0:native_centers=pred_native.clone();native_selected={g:pred_native[q].clone() for g,q in match.items()}
                prob=out['object_logits'].sigmoid();positive=torch.zeros_like(out['center_available']);positive[list(match.values())]=True
                negative=out['center_available']&~positive;instances=[]
                perturb=[]
                for dx,dy in [(2,0),(-2,0),(0,2),(0,-2)]:
                    shifted=cg.clone();shifted[valid]+=cg.new_tensor([dx,dy])/scale
                    perturb.append(assignment(out,r,s,hw,xy,eligible,shifted))
                for g in valid.nonzero().flatten().tolist():
                    ds=((out['centers'][out['center_available']]-cg[g])*scale).norm(dim=-1).sort().values
                    q=match.get(g);drift=None if q is None or g not in native_selected else float(((pred_native[q]-native_selected[g])*native_scale).norm())
                    instances.append(dict(gt_instance=g,positive_proposal=q,nearest_px=float(ds[0]) if len(ds) else None,nearest_second_margin_px=float(ds[1]-ds[0]) if len(ds)>1 else None,positive_score=float(prob[q]) if q is not None else None,positive_minus_max_negative=float(prob[q]-prob[negative].max()) if q is not None and negative.any() else None,perturb_flips=sum(p.get(g)!=q for p in perturb),native_positive_drift_px=drift,nearest_native_candidate=(int(((native_centers-pred_native[q])*native_scale).norm(dim=-1).argmin()) if q is not None else None)))
                rotations.append(dict(turn=k,active_count=int(dec['instance_active'].sum()),gt_count=int(valid.sum()),instances=instances,negative_scores=prob[negative].tolist()))
                saved[i][k]=dict(output={key:value.cpu() for key,value in out.items()},decoded={key:value.cpu() for key,value in dec.items()})
            details.append(dict(**sample,metadata=r['metadata'],rotations=rotations))
            if n%4==0:status('auditing',arm=arm,samples=n,total=48);print(arm,n,flush=True)
        assert before==[digest(m) for m in [model,center,group]]+[buffers_digest(backbone)];assert sha(source)==hashes[arm]
        atomic_save(OUT/f'{arm}_predictions.pt',saved);summary={}
        for domain in domains:
            rr=[rot for d in details if d['domain']==domain for rot in d['rotations']];ins=[g for rot in rr for g in rot['instances']];matched=[g for g in ins if g['positive_proposal'] is not None];drifts=[g['native_positive_drift_px'] for rot in rr if rot['turn'] for g in rot['instances'] if g['native_positive_drift_px'] is not None]
            avg=lambda xs:float(np.mean(xs)) if xs else None
            summary[domain]=dict(images=12,rotated_images=len(rr),instances=len(ins),missing_positive=len(ins)-len(matched),count_accuracy=avg([r['active_count']==r['gt_count'] for r in rr]),overcount_images=sum(r['active_count']>r['gt_count'] for r in rr),positive_score=avg([g['positive_score'] for g in matched]),positive_below_threshold_fraction=avg([g['positive_score']<.5 for g in matched]),negative_max_exceeds_positive_fraction=avg([g['positive_minus_max_negative']<0 for g in matched if g['positive_minus_max_negative'] is not None]),median_nearest_second_margin_px=float(np.median([g['nearest_second_margin_px'] for g in ins if g['nearest_second_margin_px'] is not None])),perturbation_flip_fraction=sum(g['perturb_flips'] for g in ins)/(len(ins)*4),rotation_positive_drift_median_px=float(np.median(drifts)) if drifts else None,rotation_positive_drift_over8_fraction=avg([d>8 for d in drifts]))
        results[arm]=dict(summary=summary,details=details);del model,center,group,backbone,saved;torch.cuda.empty_cache()
    assert sha(sup_path)==sup_hash and all(sha(ROOT/manifest[s['index']]['image'])==manifest[s['index']]['sha256'] for s in samples)
    result=dict(arms=results,source_sha256=hashes,weights_bn_inputs_unchanged=True,optimizer_steps=0,total_seconds=time.monotonic()-start);write(OUT/'results.json',result)
    (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>3200步：冻结FIT匹配稳定性诊断</h1><p>两臂各48FIT×4旋转，无优化器更新。±2px仅改变loss匹配诊断的GT拷贝，不修改标注或推理。旋转漂移同时包含预测变化，不能直接等同标签错误。</p><pre>'+json.dumps({a:r['summary'] for a,r in results.items()},indent=2)+'</pre>',encoding='utf-8')
    status('complete');state.update(stage='proposal_matching_audit_complete',active_job=None,active_pid=None,updated_utc=now().isoformat(),next='Read frozenFIT diagnostics before choosing a matched loss/matching change; no blind6400.');state['completed'].append('proposal_matching_stability_audit');write(C/'state.json',state)
    dest=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-LocalGrouping-20261005')/OUT.name;dest.mkdir(exist_ok=True)
    for p in OUT.iterdir():
        if p.suffix in ['.json','.html']:shutil.copy2(p,dest/p.name)
    print(json.dumps({a:r['summary'] for a,r in results.items()}),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write(OUT/'failure.json',dict(traceback=traceback.format_exc()));write(OUT/'status.json',dict(stage='failed',pid=os.getpid(),utc=now().isoformat()))
        raise
