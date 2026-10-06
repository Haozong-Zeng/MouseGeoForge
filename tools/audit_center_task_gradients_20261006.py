"""Read-only FIT gradient diagnostics; no optimizer or checkpoint changes."""
import os
os.environ['DLClight']='True'
import sys,json,random,fcntl,datetime as dt,traceback,shutil,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import torch,numpy as np
from preflight_mixed_local_backbone_20261005 import build_backbone,load_rgb,sha,digest
from train_mixed_local_backbone_20261005 import buffers_digest
from train_dense_center_warmup_20261005 import read,write,now
from mousegeoforge.models.mixed_local_evidence import MixedLocalEvidence
from mousegeoforge.models.dense_instance_centers import DenseInstanceCenters,dense_center_loss,dense_center_gaussian_loss
from mousegeoforge.models.local_point_evidence_probe import point_evidence_loss
from mousegeoforge.models.local_instance_grouping import rotate_mask_centroids
from mousegeoforge.models.quarter_turn_evidence import quarter_turn
C=ROOT/'output/local_grouping_cycle_20261005';OUT=C/'center_task_gradient_audit_v2'
INPUT=ROOT/'output/singleview_fusion_cycle_20261005/mixed_local_fulltrain_rgb_inputs_v1'

def gradient_stats(a,b):
    aa=float(a.square().sum());bb=float(b.square().sum());ab=float((a*b).sum())
    assert np.isfinite([aa,bb,ab]).all() and aa>0 and bb>0
    return dict(center_norm=aa**.5,point_norm=bb**.5,center_to_point_norm=(aa/bb)**.5,
                cosine=ab/(aa*bb)**.5,negative_dot=ab<0,
                summed_gradient_center_dot=aa+ab,summed_gradient_point_dot=bb+ab,
                summed_gradient_would_raise_center_first_order=(aa+ab)<0,
                summed_gradient_would_raise_point_first_order=(bb+ab)<0)

def main():
    plan=json.loads((C/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'))
    assert now()<deadline
    assert json.loads((C/'dense_center_loss9216_point_audit_v1/status.json').read_text())['stage']=='complete'
    lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    OUT.mkdir(exist_ok=False);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    state=json.loads((C/'state.json').read_text());state.update(stage='center_task_gradient_audit_running',active_job=str(OUT.relative_to(ROOT)),active_pid=os.getpid(),updated_utc=now().isoformat());write(C/'state.json',state)
    def status(stage,**kw):write(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=now().isoformat(),**kw))
    status('loading')
    rows={r['index']:r for r in read(INPUT/'evaluation_only.pt')}
    manifest={r['index']:r for r in json.loads((INPUT/'inference_manifest.json').read_text())}
    sup_path=C/'instance_supervision_audit_v1/evaluation_instance_targets.pt';sup_hash=sha(sup_path);sup={r['index']:r for r in read(sup_path)}
    domains=['real_bottom','real_other','single','dual'];rng=random.Random(61);samples=[]
    for domain in domains:
        pool=sorted(i for i,r in rows.items() if r['metadata']['stage']=='fit' and
                    ((domain.startswith('real_') and r['metadata']['kind']=='real' and bool(r['metadata']['physical_bottom'])==(domain=='real_bottom')) or r['metadata']['kind']==domain))
        for j,i in enumerate(rng.sample(pool,12)):samples.append(dict(index=i,domain=domain,quarter_turn=j%4))
    assert all(rows[s['index']]['metadata'].get('source') not in ['FR2','AZR2L2','GR2','UR2','UR3'] for s in samples)
    protocol=dict(samples=samples,seed=61,source_step=9216,arms=['focal','gaussian'],
                  purpose='Gradient diagnostics only, 12 deterministic FIT samples per realbottom/realother/single/dual stratum, quarterturn0/1/2/3 balanced. No optimizer updates or CAL/DEV/locked image inference.',
                  parameters='All shared backbone parameters, raw loss gradients before clipping/AdamW; task heads excluded. Each loss separately differentiated on same forward.',
                  limitations='Negative cosine alone is not evidence of harmful training; raw gradient sums do not model AdamW state, clipping or generalization. Small FIT sample, not benchmark. No new loss automatically selected.',
                  references=['https://arxiv.org/abs/2001.06782','https://arxiv.org/abs/1711.02257'],
                  deadline=plan['training_stop_utc'],code_sha256=sha(Path(__file__)),supervision_sha256=sup_hash)
    write(OUT/'protocol.json',protocol);results={};start=time.monotonic();hashes={}
    for arm in protocol['arms']:
        source=C/f'dense_center_loss_{arm}_9216_v1/center_step9216.pt';hashes[arm]=sha(source);cp=read(source)
        backbone,_=build_backbone();model=MixedLocalEvidence(backbone,adapt_backbone=True).cuda();model.backbone.load_state_dict(cp['backbone']);model.readout.load_state_dict(cp['readout']);model.eval()
        center=DenseInstanceCenters().cuda();center.load_state_dict(cp['center']);center.eval();del cp
        before=(digest(model),digest(center),buffers_digest(backbone));params=list(backbone.parameters());sums={};details=[]
        objective=dense_center_loss if arm=='focal' else dense_center_gaussian_loss
        for n,sample in enumerate(samples,1):
            if now()>=deadline:raise TimeoutError('Gradient diagnostic cutoff')
            i=sample['index'];r=rows[i];s=sup[i];hw=manifest[i]['image_hw'];x=load_rgb(manifest[i]).cuda();xy=r['target'].cuda();eligible=r['eligible'].cuda()
            centers=s['mask_centroid'].cuda();valid=s['centroid_valid'].cuda();k=sample['quarter_turn']
            assert bool(s['detection_annotation_complete']) and torch.equal(s['centroid_valid'],s['instance_valid'])
            if k:x,xy,hw=quarter_turn(x,xy,eligible,hw,k);centers=rotate_mask_centroids(centers,valid,k)
            evidence=model(x,hw);out=center(evidence,hw)
            lc=objective(out,centers,valid,hw,annotation_complete=True);lp=point_evidence_loss(evidence,xy,eligible,hw)
            def flat(loss,retain):
                gs=torch.autograd.grad(loss,params,retain_graph=retain,allow_unused=True)
                return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for p,g in zip(params,gs)]).detach()
            gc=flat(lc,True);gp=flat(lp,False);metrics=gradient_stats(gc,gp)
            domain=sample['domain']
            if domain not in sums:sums[domain]=[torch.zeros_like(gc),torch.zeros_like(gp)]
            sums[domain][0].add_(gc/12);sums[domain][1].add_(gp/12)
            details.append(dict(**sample,metadata=r['metadata'],center_loss=float(lc.detach()),point_loss=float(lp.detach()),**metrics))
            del gc,gp,lc,lp,out,evidence,x
            if n%4==0:status('auditing',arm=arm,samples=n,total=len(samples));print(arm,n,flush=True)
        summary={}
        for domain in domains:
            ds=[d for d in details if d['domain']==domain]
            summary[domain]=dict(samples=len(ds),mean_cosine=float(np.mean([d['cosine'] for d in ds])),
                negative_dot_fraction=float(np.mean([d['negative_dot'] for d in ds])),
                median_norm_ratio=float(np.median([d['center_to_point_norm'] for d in ds])),
                point_ascent_fraction=float(np.mean([d['summed_gradient_would_raise_point_first_order'] for d in ds])),
                center_ascent_fraction=float(np.mean([d['summed_gradient_would_raise_center_first_order'] for d in ds])),
                mean_gradient=gradient_stats(*sums[domain]))
        cross={}
        for task,name in enumerate(['center','point']):
            cross[name]={}
            for j,a in enumerate(domains):
                for b in domains[j+1:]:cross[name][a+'__'+b]=gradient_stats(sums[a][task],sums[b][task])['cosine']
        weights={'real_bottom':1/6,'real_other':1/6,'single':1/3,'dual':1/3}
        aggregate=[sum(sums[d][t]*weights[d] for d in domains) for t in [0,1]]
        results[arm]=dict(strata=summary,cross_domain_cosines=cross,training_weighted_task_gradients=gradient_stats(*aggregate))
        write(OUT/f'{arm}_details.json',details)
        assert before==(digest(model),digest(center),buffers_digest(backbone)) and sha(source)==hashes[arm]
        assert all(p.grad is None for p in list(model.parameters())+list(center.parameters()))
        del model,center,backbone,params,sums,aggregate;torch.cuda.empty_cache()
    assert sha(sup_path)==sup_hash and all(sha(ROOT/manifest[s['index']]['image'])==manifest[s['index']]['sha256'] for s in samples)
    result=dict(summary=results,checkpoint_sha256=hashes,weight_bn_sources_unchanged=True,optimizer_steps=0,seconds=time.monotonic()-start)
    write(OUT/'results.json',result)
    (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>中心与关键点梯度诊断</title><h1>仅FIT梯度诊断，不代表因果或泛化结论</h1><pre>'+json.dumps(dict(protocol=protocol,result=result),indent=2)+'</pre>',encoding='utf-8')
    dest=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-LocalGrouping-20261005')/OUT.name;dest.mkdir(parents=True,exist_ok=True)
    for p in OUT.iterdir():
        if p.suffix in ['.json','.html']:shutil.copy2(p,dest/p.name)
    status('complete');state.update(stage='center_task_gradient_audit_complete',active_job=None,active_pid=None,next='Review gradient evidence and point audit before choosing an optimization intervention; no automatic training.',updated_utc=now().isoformat());state['completed'].append('center_task_gradient_audit');write(C/'state.json',state);print('COMPLETE',json.dumps(results),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write(OUT/'failure.json',dict(utc=now().isoformat(),traceback=traceback.format_exc()));write(OUT/'status.json',dict(stage='failed',utc=now().isoformat(),pid=os.getpid()))
        raise
