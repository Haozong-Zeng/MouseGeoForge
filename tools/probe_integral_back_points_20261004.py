"""Spatial expectation versus pooled memory, crossed with strict/weak point loss."""
import collections,datetime as dt,fcntl,json,math,os,random,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from mousegeoforge.uncertain_back_points import uncertain_back_point_loss
from mousegeoforge.models.back_point_integral import BackPointIntegralProbe
from run_depth_pose_ablation import fingerprint,write_json
CYCLE=ROOT/'output/surface_curve_cycle_20261003'
SOURCE=CYCLE/'back_point_probe_inputs_v1'
OUT=CYCLE/'integral_back_point_probe_v1'


def main():
    assert not OUT.exists() and (SOURCE/'manifest.json').exists()
    lock=(ROOT/'output/teacher_temporal_cycle_20261001/gpu_job.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert torch.cuda.is_available();OUT.mkdir();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    deadline=dt.datetime(2026,10,4,8,5,16,tzinfo=dt.timezone.utc)
    def status(stage,**kw):write_json(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),**kw))
    status('loading');saved={k:torch.load(SOURCE/f'{k}_inputs.pt',map_location='cpu',weights_only=False) for k in ['synthetic','real']}
    data={k:{name:value.cuda() for name,value in s['tensors'].items()} for k,s in saved.items()};meta={k:s['metadata'] for k,s in saved.items()}
    train={k:[i for i,m in enumerate(mm) if m['role']=='train' and data[k]['valid'][i].any()] for k,mm in meta.items()}
    val={k:[i for i,m in enumerate(mm) if m['role']=='validation'] for k,mm in meta.items()}
    assert all(m['session'] not in ('GR2','UR2','UR3') for m in meta['real'])
    manifests={'synthetic':CYCLE/'curve_spatial_memory_v2/manifest.json','real':CYCLE/'real_point_spatial_memory_v1/manifest.json'}
    spatial={}
    for kind,path in manifests.items():
        manifest=json.loads(path.read_text());maps=[None]*len(meta[kind])
        if kind=='real':assert manifest['source_inputs_sha256']==fingerprint(SOURCE/'real_inputs.pt')
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
    modes=['global_strict','spatial_strict','global_weak_pair','spatial_weak_pair'];steps=1600;seeds=[17,29,43]
    write_json(OUT/'protocol.json',dict(modes=modes,seeds=seeds,steps=steps,synthetic_batch=8,real_batch=8,real_weight=.25,
        learning_rate='cosine3e-4to1e-5',weight_decay=.001,
        model='Experimental per-instance 2point conditioned heatmap plus spatial expectation; no production changes.',
        comparison='2x2: actual spatial memory vs pooled memory expanded to all cells; strict vs weak_pair. Same parameters, initialization, batches and update budget within seed.',
        features='Exact previous frozen pose tokens, originaldecoder patch16memory fp16 cached then48x48float32resize. Spatial memory includes original positionalencoding. Global control loses spatialvariation, explicitXY embedding retained in both.',
        initialization='Zero final queryprojection: identical broad Gaussian prior about frozen coarse predictions, sigma0.25normalized; not the same as exact coarse prediction near boundaries.',
        supervision='Same exact synthetic06/07points; real strictHuber or previous directionalweakpoint withcappedchordfallback. Coordinateexpectationloss only, no heatmapGT or fullcurveGT.',
        evaluation='Fixedfinal1600updates, no validationselection. Association-assisted existing query binding, not end-to-end detector metric. Dev sources not pristineexternaltest. No lockedholdout.',
        limitations='Expectation can average multiple peaks; inspect modes/entropy before claimingpointlocalization. Broadpriorandtokenconditioning are prediction-only, noGTcrop.',
        schedule='Identical seed+3000 dataset/session-uniform minibatches to prior pointprobe.',
        references=['https://arxiv.org/abs/1801.07372','https://github.com/JimmySuen/integral-human-pose'],
        implementation='Independent small readout inspired by spatialexpectation, not a complete reproduction of citedmethods.',
        hashes={k:fingerprint(SOURCE/f'{k}_inputs.pt') for k in data},memory_hashes={k:fingerprint(p) for k,p in manifests.items()},hard_deadline=deadline.isoformat()))

    def predict(model,kind,ids):
        d=data[kind];return model(d['pose'][ids],d['coarse_xy'][ids],spatial[kind][ids])

    def loss_for(model,kind,ids,mode):
        mode=mode.split('_',1)[1]
        d=data[kind];p=predict(model,kind,ids)*d['pixels'][ids,None];g=d['gt_xy'][ids]*d['pixels'][ids,None];terms=[]
        for n,(pp,gg,valid) in enumerate(zip(p,g,d['valid'][ids])):
            if not valid.any():terms.append(pp.sum()*0);continue
            strict=F.smooth_l1_loss(pp[valid]/768,gg[valid]/768,beta=.01)
            if kind=='synthetic' or mode=='strict':terms.append(strict)
            elif mode=='isotropic_quarter':terms.append(.25*strict)
            else:
                prefix='pair' if mode=='weak_pair' else 'primary';i=ids[n]
                terms.append(uncertain_back_point_loss(pp,gg,valid,tangent=d[prefix+'_tangent'][i],reliability=d[prefix+'_reliability'][i]))
        return torch.stack(terms).mean()

    def evaluate(model,kind,ids):
        d=data[kind];metrics=collections.defaultdict(list)
        with torch.no_grad():
            p=predict(model,kind,ids);delta=(p-d['gt_xy'][ids])*d['pixels'][ids,None]
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
        rng=random.Random(seed+3000);schedule=[{k:[rng.choice(buckets[k][rng.choice(keys[k])]) for _ in range(8)] for k in data} for _ in range(steps)]
        initial=None
        for mode in modes:
            arm=f'{mode}_seed{seed}';torch.manual_seed(seed);model=BackPointIntegralProbe(mode.split('_',1)[0]).cuda()
            if initial is None:initial={k:v.detach().clone() for k,v in model.state_dict().items()}
            else:assert all(torch.equal(v,initial[k]) for k,v in model.state_dict().items())
            optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.001)
            history=[dict(step=0,**evaluate_all(model))];status('training',arm=arm,step=0)
            for step,ids in enumerate(schedule,1):
                assert dt.datetime.now(dt.timezone.utc)<deadline
                lr=1e-5+(3e-4-1e-5)*.5*(1+math.cos(math.pi*(step-1)/steps))
                for group in optimizer.param_groups:group['lr']=lr
                optimizer.zero_grad(set_to_none=True);sl=loss_for(model,'synthetic',ids['synthetic'],mode);rl=loss_for(model,'real',ids['real'],mode);loss=sl+.25*rl
                assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm);optimizer.step()
                if step%400==0:
                    row=dict(step=step,lr=lr,synthetic_loss=float(sl.detach()),real_loss=float(rl.detach()),**evaluate_all(model));history.append(row)
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
