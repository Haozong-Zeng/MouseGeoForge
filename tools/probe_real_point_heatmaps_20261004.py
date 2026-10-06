"""Expanded data: extra coordinate optimization vs explicit point heatmap loss."""
import collections,datetime as dt,fcntl,json,math,os,random,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from mousegeoforge.uncertain_back_points import uncertain_back_point_loss
from mousegeoforge.models.back_point_integral import BackPointIntegralProbe
from mousegeoforge.point_heatmap_supervision import point_heatmap_kl
from run_depth_pose_ablation import fingerprint,write_json
CYCLE=ROOT/'output/surface_curve_cycle_20261003'
SOURCE=CYCLE/'back_point_probe_inputs_v1'
OUT=CYCLE/'real_point_heatmap_v1'
EXPANDED=CYCLE/'expanded_real_point_features_v1'


def main():
    assert not OUT.exists() and (SOURCE/'manifest.json').exists()
    lock=(ROOT/'output/teacher_temporal_cycle_20261001/gpu_job.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert torch.cuda.is_available();OUT.mkdir();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    deadline=dt.datetime(2026,10,4,8,5,16,tzinfo=dt.timezone.utc)
    def status(stage,**kw):write_json(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),**kw))
    status('loading');paths={'synthetic':SOURCE/'synthetic_inputs.pt','real':EXPANDED/'real_inputs.pt'}
    saved={k:torch.load(path,map_location='cpu',weights_only=False) for k,path in paths.items()}
    original=torch.load(SOURCE/'real_inputs.pt',map_location='cpu',weights_only=False)
    assert saved['real']['metadata'][:236]==original['metadata']
    for key,value in original['tensors'].items():assert torch.equal(saved['real']['tensors'][key][:236],value)
    data={k:{name:value.cuda() for name,value in s['tensors'].items()} for k,s in saved.items()};meta={k:s['metadata'] for k,s in saved.items()}
    train={k:[i for i,m in enumerate(mm) if m['role']=='train' and data[k]['valid'][i].any()] for k,mm in meta.items()}
    val={k:[i for i,m in enumerate(mm) if m['role']=='validation'] for k,mm in meta.items()}
    assert all(m['session'] not in ('GR2','UR2','UR3') for m in meta['real'])
    manifests={'synthetic':CYCLE/'curve_spatial_memory_v2/manifest.json','real':EXPANDED/'memory_manifest.json'}
    spatial={}
    for kind,path in manifests.items():
        manifest=json.loads(path.read_text());maps=[None]*len(meta[kind])
        if kind=='real':assert manifest['source_inputs_sha256']==fingerprint(EXPANDED/'real_inputs.pt')
        else:
            original=torch.load(CYCLE/'expanded_curve_features_v1/frozen_inputs.pt',map_location='cpu',weights_only=False)
            assert original['metadata']==meta[kind]
        for record in manifest['records']:
            assert fingerprint(ROOT/record['path'])==record['sha256']
            cached=torch.load(ROOT/record['path'],map_location='cpu',weights_only=False)
            memory=F.interpolate(cached['memory'].float(),(48,48),mode='bilinear',align_corners=False)
            for bind in record['bindings']:
                i=bind['observation'];assert maps[i] is None;maps[i]=memory[bind['view_index']]
        assert all(m is not None for m in maps);spatial[kind]=torch.stack(maps).cuda()
    modes=['coordinate','heatmap'];steps=3200;seeds=[17,29,43];heatmap_weight=.002
    write_json(OUT/'protocol.json',dict(modes=modes,seeds=seeds,steps=steps,synthetic_batch=8,real_batch=8,real_weight=.25,
        learning_rate='cosine1e-4to1e-5',weight_decay=.001,heatmap_weight=heatmap_weight,heatmap_sigma_px=12.,
        model='Same2pointspatialintegralprobe, frozenencoder/mainheads, noGTforward.',
        initialization='Identical per-seed expanded1600step pointcheckpoint. Both get3200extra updates withoptimizerreset; coordinatearm controls longertraining.',
        supervision='StrictfinitepointHuber inbotharms; optional0.002*(syntheticGaussianKL+0.25realGaussianKL). Same12pxisotropicchosenkernel, NOTestimatedhumanuncertainty. No realcurve/fractionassignment. Pointlabelcenters unchanged.',
        schedule='Identicalperseed+5000 syntheticdataset/eligible-realTRAINsession-uniform batches forbotharms. Real1043TRAINimages, same48VALimages/58points, synthetic326unchanged.',
        evaluation='Fixedfinal3200extra steps, no validationcheckpointselection. Reportpointaccuracy andpeak/expectation gap. ExistingGTassociation-assistedquery.',
        constraints='No labelupdate/pseudoGT, no lockedholdout. This testsfinitepointlocalization only, NOT fullsurfacecurve correctness.',
        hashes={k:fingerprint(p) for k,p in paths.items()},memory_hashes={k:fingerprint(p) for k,p in manifests.items()},
        initial_hashes={str(seed):fingerprint(CYCLE/f'real_point_data_scaling_v1/expanded_seed{seed}_latest.pt') for seed in seeds},
        hard_deadline=deadline.isoformat()))

    def predict(model,kind,ids):
        d=data[kind];return model(d['pose'][ids],d['coarse_xy'][ids],spatial[kind][ids])

    def loss_for(model,kind,ids):
        d=data[kind]
        prediction,prob=model(d['pose'][ids],d['coarse_xy'][ids],spatial[kind][ids],True)
        p=prediction*d['pixels'][ids,None];g=d['gt_xy'][ids]*d['pixels'][ids,None];coord=[];heat=[]
        for n,i in enumerate(ids):
            valid=d['valid'][i];assert valid.any()
            coord.append(F.smooth_l1_loss(p[n,valid]/768,g[n,valid]/768,beta=.01))
            heat.append(point_heatmap_kl(prob[n:n+1],d['gt_xy'][i:i+1],valid[None],d['pixels'][i:i+1],sigma_px=12.))
        return torch.stack(coord).mean(),torch.stack(heat).mean()

    def evaluate(model,kind,ids):
        d=data[kind];metrics=collections.defaultdict(list)
        with torch.no_grad():
            p,prob=model(d['pose'][ids],d['coarse_xy'][ids],spatial[kind][ids],True);delta=(p-d['gt_xy'][ids])*d['pixels'][ids,None]
            cell=prob.flatten(2).argmax(-1);peak=torch.stack([(cell%48+.5)/48,(cell//48+.5)/48],-1)
            metrics['peak_error_px'].extend(((peak-d['gt_xy'][ids])*d['pixels'][ids,None]).norm(dim=-1)[d['valid'][ids]].tolist())
            metrics['peak_expectation_px'].extend(((peak-p)*d['pixels'][ids,None]).norm(dim=-1)[d['valid'][ids]].tolist())
            for n,i in enumerate(ids):
                valid=d['valid'][i];error=delta[n].norm(dim=-1)
                metrics['point_error_px'].extend(error[valid].tolist())
                for k in range(2):
                    if valid[k]:metrics[f'point{k+6}_error_px'].append(float(error[k]))
                tv=valid&(d['pair_reliability'][i]>0);t=d['pair_tangent'][i];normal=torch.stack([-t[:,1],t[:,0]],-1)
                metrics['common_proxy_normal_px'].extend((delta[n]*normal).sum(-1).abs()[tv].tolist())
                metrics['common_proxy_along_px'].extend((delta[n]*t).sum(-1).abs()[tv].tolist())
                metrics['predicted_pair_separation_px'].append(float(((p[n,1]-p[n,0])*d['pixels'][i]).norm()))
        return {k:dict(n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),p90=float(np.quantile(v,.9))) for k,v in metrics.items() if v}

    def evaluate_all(model):return {k+'_'+role:evaluate(model,k,ids[k]) for k in data for role,ids in [('train',train),('validation',val)]}
    buckets={k:collections.defaultdict(list) for k in data}
    for k in data:
        for i in train[k]:buckets[k][meta[k][i]['dataset'] if k=='synthetic' else meta[k][i]['session']].append(i)
    keys={k:sorted(buckets[k]) for k in data};results=[];predictions={}
    for seed in seeds:
        rng=random.Random(seed+5000);schedule=[{k:[rng.choice(buckets[k][rng.choice(keys[k])]) for _ in range(8)] for k in data} for _ in range(steps)]
        state=torch.load(CYCLE/f'real_point_data_scaling_v1/expanded_seed{seed}_latest.pt',map_location='cpu',weights_only=False)['state']
        initial_metrics=None
        for mode in modes:
            arm=f'{mode}_seed{seed}';torch.manual_seed(seed);model=BackPointIntegralProbe('spatial').cuda();model.load_state_dict(state)
            optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.001)
            history=[dict(step=0,**evaluate_all(model))];status('training',arm=arm,step=0)
            if initial_metrics is None:initial_metrics=history[0]
            else:assert history[0]==initial_metrics
            for step,ids in enumerate(schedule,1):
                assert dt.datetime.now(dt.timezone.utc)<deadline
                lr=1e-5+(1e-4-1e-5)*.5*(1+math.cos(math.pi*(step-1)/steps))
                for group in optimizer.param_groups:group['lr']=lr
                optimizer.zero_grad(set_to_none=True);sl,sh=loss_for(model,'synthetic',ids['synthetic']);rl,rh=loss_for(model,'real',ids['real']);coord=sl+.25*rl;kl=sh+.25*rh
                if step==1 and mode=='heatmap':
                    params=list(model.parameters());cg=torch.autograd.grad(coord,params,retain_graph=True);hg=torch.autograd.grad(kl,params,retain_graph=True)
                    cn=float(torch.sqrt(sum(g.square().sum() for g in cg)));hn=float(torch.sqrt(sum(g.square().sum() for g in hg)))
                    write_json(OUT/f'gradient_seed{seed}.json',dict(coordinate_norm=cn,heatmap_norm=hn,weighted_ratio=heatmap_weight*hn/max(cn,1e-12),fixed_weight=heatmap_weight))
                loss=coord+(heatmap_weight if mode=='heatmap' else 0.)*kl
                assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm);optimizer.step()
                if step%400==0:
                    row=dict(step=step,lr=lr,synthetic_loss=float(sl.detach()),real_loss=float(rl.detach()),heatmap_loss=float(kl.detach()),**evaluate_all(model));history.append(row)
                    write_json(OUT/f'{arm}_history.json',history);status('training',arm=arm,step=step)
                    torch.save(dict(state=model.state_dict(),optimizer=optimizer.state_dict(),seed=seed,mode=mode,step=step),OUT/f'{arm}_latest.pt')
                    print(arm,step,'synval',row['synthetic_validation']['point_error_px']['mean'],'realval',row['real_validation']['point_error_px']['mean'],flush=True)
            results.append(dict(arm=arm,seed=seed,mode=mode,**history[-1]));write_json(OUT/'partial_results.json',results)
            with torch.no_grad():
                for kind in data:predictions[arm+'__'+kind]=predict(model,kind,list(range(len(meta[kind])))).cpu().numpy()
    np.savez_compressed(OUT/'predictions.npz',**predictions);write_json(OUT/'results.json',dict(results=results,inspection_pending=True,no_production_promotion=True));status('complete')


if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(exist_ok=True);write_json(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()));raise
