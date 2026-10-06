"""Four loss controls on an experimental two-point readout; main heads frozen."""
import collections,datetime as dt,fcntl,json,math,os,random,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from mousegeoforge.uncertain_back_points import uncertain_back_point_loss
from run_depth_pose_ablation import fingerprint,write_json
CYCLE=ROOT/'output/surface_curve_cycle_20261003'
SOURCE=CYCLE/'back_point_probe_inputs_v1'
OUT=CYCLE/'uncertain_back_point_probe_v1'


class BackPointProbe(nn.Module):
    def __init__(self):
        super().__init__();self.delta=nn.Sequential(nn.LayerNorm(1024),nn.Linear(1024,256),nn.GELU(),nn.Linear(256,4))
        nn.init.zeros_(self.delta[-1].weight);nn.init.zeros_(self.delta[-1].bias)
    def forward(self,pose,coarse):
        return coarse.detach()+.5*self.delta(pose.detach()).reshape(-1,2,2).tanh()


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
    modes=['strict','isotropic_quarter','weak_neighbors','weak_pair'];steps=1600;seeds=[17,29,43]
    write_json(OUT/'protocol.json',dict(modes=modes,seeds=seeds,steps=steps,synthetic_batch=8,real_batch=8,real_weight=.25,
        learning_rate='cosine3e-4to1e-5',weight_decay=.001,
        model='New1024->256->4zero-initialized residual on frozen coarse06/07predictions. NOT fullexistinglocal-refinedCore23head; originaloutputs/weights unchanged.',
        losses={'strict':'RealcomponentwiseHuber exactpoint','isotropic_quarter':'0.25*sameRealHuber controls overall weakening',
                'weak_neighbors':'Anisotropictolerance4along/1normal wherelegacyneighboraxisavailable; otherwise0.25isotropic',
                'weak_pair':'Same, plus06->07chord fallback if>=20px withreliability0.5blend, notexacttangent'},
        synthetic='Exact06/07surfacepointHuber inallarms; no noisyfraction/curveendpointtargets',
        initialization='Identical per seed across4arms; zero residual meanscoarsebaselineexact',
        schedule='Uniformsyntheticdataset and realTRAINsession; seed+3000,identicalminibatches forall4arms',
        evaluation='Fixedfinal1600steps; no validationcheckpointselection. Realnormal/along useCOMMONproxyaxes acrossarms, notindependentGT. ReportexactpointandsyntheticGT too.',
        constraints='No curve extrapolation output, no real3DGT orvisibilityguesses, no labelupdate/promotion, no lockedholdout.',
        hashes={k:fingerprint(SOURCE/f'{k}_inputs.pt') for k in data},hard_deadline=deadline.isoformat()))

    def predict(model,kind,ids):
        d=data[kind];return model(d['pose'][ids],d['coarse_xy'][ids])

    def loss_for(model,kind,ids,mode):
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
            arm=f'{mode}_seed{seed}';torch.manual_seed(seed);model=BackPointProbe().cuda()
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
