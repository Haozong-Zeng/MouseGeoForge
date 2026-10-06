"""Matched coordinate-only vs coordinate+known-material heatmap supervision."""
import collections, datetime as dt, fcntl, json, math, os, random, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import numpy as np
import torch
from torch.nn import functional as F
from mousegeoforge.models.surface_curve_integral import SurfaceCurveIntegralProbe
from mousegeoforge.point_heatmap_supervision import point_heatmap_kl
from mousegeoforge.surface_curve_supervision import cubic_bspline,curve_loss,material_curve_loss,points_to_polyline
from probe_diverse_curve_features_20261003 import FeatureProbe
from run_depth_pose_ablation import write_json,fingerprint
CYCLE=ROOT/'output/surface_curve_cycle_20261003'
SOURCE=CYCLE/'expanded_curve_features_v1'
MEMORY=CYCLE/'curve_spatial_memory_v2'
BASE=CYCLE/'curve_data_scaling_v1'
OUT=CYCLE/'curve_heatmap_supervision_v1'


def main():
    assert not OUT.exists() and json.loads((MEMORY/'status.json').read_text())['stage']=='complete'
    lock=(ROOT/'output/teacher_temporal_cycle_20261001/gpu_job.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert torch.cuda.is_available();OUT.mkdir();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    deadline=dt.datetime(2026,10,4,8,5,16,tzinfo=dt.timezone.utc)
    def status(stage,**kw):write_json(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),**kw))
    status('loading_memory')
    saved=torch.load(SOURCE/'frozen_inputs.pt',map_location='cpu',weights_only=False)
    manifest=json.loads((MEMORY/'manifest.json').read_text());assert manifest['source_inputs_sha256']==fingerprint(SOURCE/'frozen_inputs.pt')
    data={k:v.cuda() for k,v in saved['tensors'].items()};meta=saved['metadata'];maps=[None]*len(meta)
    for record in manifest['records']:
        assert fingerprint(ROOT/record['path'])==record['sha256']
        cache=torch.load(ROOT/record['path'],map_location='cpu',weights_only=False)
        memory=F.interpolate(cache['memory'].float(),(48,48),mode='bilinear',align_corners=False)
        for bind in record['bindings']:
            i=bind['observation'];assert maps[i] is None;maps[i]=memory[bind['view_index']]
    assert all(m is not None for m in maps);spatial=torch.stack(maps).cuda();del maps
    train=[i for i,m in enumerate(meta) if m['role']=='train'];val=[i for i,m in enumerate(meta) if m['role']=='validation']
    assert len(train)==238 and len(val)==88
    seeds=[17,29,43];steps=1600;heatmap_weight=.002
    write_json(OUT/'protocol.json',dict(seeds=seeds,steps=steps,batch_size=8,learning_rate='cosine1e-4to1e-5',weight_decay=.001,
        arms=['coordinate','heatmap'],heatmap_weight=heatmap_weight,heatmap_sigma_px=12.,
        initialization='Same corresponding synthetic-only spatialintegralcurve1600step checkpoint inbotharms; priorcontrols fromsamefrozenexpandedbase.',
        supervision='Both materialHuber+.1shape(order0); heatmaparm adds0.002KL(targetGaussian||predictedprobability) at6KNOWNsynthetic materialu0,.2,.4,.6,.8,1. No realmanualpointsassignedfractions.',
        kernels='Isotropic12pxsigma at768trainingresolution. Discrete48x48heatmaps, pixelmetricusesactualWH. Chosensupervisionkernel, NOTcalibrateduncertainty. Finiteprojectedcoordinatesinsideimageonly forheatmaploss. Allknownamodalsynthetictargets, notvisibilitylabels.',
        schedule='Uniformdataset, seed+1000 samepaired1600minibatches as priorrefinementcontrols',
        selection='Fixedfinal1600extra updates, no validationcheckpointselection. Sharedinit,optimizer,schedule,andparametercount.',
        constraints='No realtraining, no lockedholdout, sourceannotationsunchanged, frozenencoder/mainheads. No productionpromotion.',
        inputs_sha256=fingerprint(SOURCE/'frozen_inputs.pt'),memory_manifest_sha256=fingerprint(MEMORY/'manifest.json'),
        initial_hashes={str(seed):fingerprint(CYCLE/f'integral_surface_curve_probe_v1/spatial_seed{seed}.pt') for seed in seeds},
        limitations='Sameassetsyntheticdevelopmentvalidation; GTassociation-assistedquery. Gaussianlabelsareapriordistributionnotanatomicaluncertainty; peakaccuracyandwholecurvemetrics reported.',
        hard_deadline=deadline.isoformat()))
    valid=torch.ones(65,device='cuda',dtype=torch.bool);u=torch.linspace(0,1,65,device='cuda')
    def evaluate(model,controls,ids):
        values=collections.defaultdict(list)
        with torch.no_grad():
            norm,ppoints,prob=model(data['pose'][ids],spatial[ids],controls[ids],True)
            pred=norm*data['pixels'][ids,None]
            target6=F.interpolate(data['gt'][ids].transpose(1,2),size=6,mode='linear',align_corners=True).transpose(1,2)
            good=((target6>=0)&(target6<=1)).all(-1)&torch.isfinite(target6).all(-1)
            index=prob.flatten(2).argmax(-1);peak=torch.stack([(index%48+.5)/48,(index//48+.5)/48],-1)
            values['six_peak_error_px'].extend(((peak-target6)*data['pixels'][ids,None]).norm(dim=-1)[good].tolist())
            values['six_peak_expectation_px'].extend(((peak-ppoints)*data['pixels'][ids,None]).norm(dim=-1)[good].tolist())
            gt=data['gt'][ids]*data['pixels'][ids,None]
            for p,g in zip(pred,gt):
                values['pred_to_gt_px'].extend(points_to_polyline(p,g,valid)[0].sqrt().tolist())
                values['gt_to_pred_px'].extend(points_to_polyline(g,p,torch.ones(33,device='cuda',dtype=torch.bool))[0].sqrt().tolist())
                values['endpoints_px'].extend((p[[0,-1]]-g[[0,-1]]).norm(dim=-1).tolist())
        return {k:dict(mean=float(np.mean(v)),median=float(np.median(v)),p90=float(np.quantile(v,.9))) for k,v in values.items()}
    results=[];predictions={};buckets=collections.defaultdict(list)
    for i in train:buckets[meta[i]['dataset']].append(i)
    keys=sorted(buckets)
    with np.load(BASE/'predictions.npz') as z:references={k:z[k] for k in z.files if k.startswith('expanded')}
    for seed in seeds:
        frozen=FeatureProbe(False).cuda().eval();checkpoint=BASE/f'expanded_seed{seed}.pt'
        frozen.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=False)['state']);frozen.requires_grad_(False)
        with torch.no_grad():
            d=frozen.delta(data['pose']).reshape(-1,6,2)
            controls=data['boxes'][:,:2,None].transpose(-1,-2)+(data['base_logits'].sigmoid()-.5)*data['boxes'][:,2:,None].transpose(-1,-2)+.5*d.tanh()
            original=cubic_bspline(controls,33).cpu().numpy()
        assert np.allclose(original,references[f'expanded_seed{seed}'],atol=1e-6,rtol=0)
        predictions[f'base_seed{seed}']=original;del frozen
        rng=random.Random(seed+1000);schedule=[[rng.choice(buckets[rng.choice(keys)]) for _ in range(8)] for _ in range(steps)]
        init_state=None
        for mode in ['coordinate','heatmap']:
            arm=f'{mode}_seed{seed}';torch.manual_seed(seed)
            model=SurfaceCurveIntegralProbe('spatial').cuda()
            model.load_state_dict(torch.load(CYCLE/f'integral_surface_curve_probe_v1/spatial_seed{seed}.pt',map_location='cpu',weights_only=False)['state'])
            if init_state is None:init_state={k:v.detach().clone() for k,v in model.state_dict().items()}
            else:assert all(torch.equal(v,init_state[k]) for k,v in model.state_dict().items())
            optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.001)
            history=[dict(step=0,train=evaluate(model,controls,train),validation=evaluate(model,controls,val))]
            baseline=json.loads((BASE/f'expanded_seed{seed}_history.json').read_text())[-1]
            if mode=='coordinate':initial_metrics=history[0]
            else:assert history[0]==initial_metrics
            status('training',arm=arm,step=0)
            for step,ids in enumerate(schedule,1):
                assert dt.datetime.now(dt.timezone.utc)<deadline
                lr=1e-5+(1e-4-1e-5)*.5*(1+math.cos(math.pi*(step-1)/steps))
                for group in optimizer.param_groups:group['lr']=lr
                optimizer.zero_grad(set_to_none=True)
                norm,ppoints,prob=model(data['pose'][ids],spatial[ids],controls[ids],True)
                p=norm*data['pixels'][ids,None];g=data['gt'][ids]*data['pixels'][ids,None]
                loss=torch.stack([material_curve_loss(pp,gg,valid,u,material_parameter_known=True,scale=768)+.1*curve_loss(pp,gg,valid,full_curve=True,scale=768,order_weight=0.) for pp,gg in zip(p,g)]).mean()
                target6=F.interpolate(data['gt'][ids].transpose(1,2),size=6,mode='linear',align_corners=True).transpose(1,2)
                good=((target6>=0)&(target6<=1)).all(-1)&torch.isfinite(target6).all(-1)
                kl=point_heatmap_kl(prob,target6,good,data['pixels'][ids],sigma_px=12.)
                if step==1 and mode=='heatmap':
                    params=list(model.parameters());cg=torch.autograd.grad(loss,params,retain_graph=True);hg=torch.autograd.grad(kl,params,retain_graph=True)
                    cn=float(torch.sqrt(sum(v.square().sum() for v in cg)));hn=float(torch.sqrt(sum(v.square().sum() for v in hg)))
                    write_json(OUT/f'gradient_seed{seed}.json',dict(coordinate_norm=cn,heatmap_norm=hn,weighted_ratio=heatmap_weight*hn/max(cn,1e-12),fixed_weight=heatmap_weight))
                coord_loss=loss;loss=coord_loss+(heatmap_weight if mode=='heatmap' else 0.)*kl
                assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm);optimizer.step()
                if step%400==0:
                    row=dict(step=step,lr=lr,loss=float(loss.detach()),coordinate_loss=float(coord_loss.detach()),heatmap_kl=float(kl.detach()),train=evaluate(model,controls,train),validation=evaluate(model,controls,val));history.append(row)
                    write_json(OUT/f'{arm}_history.json',history);torch.save(dict(state=model.state_dict(),optimizer=optimizer.state_dict(),seed=seed,mode=mode,step=step,base_sha256=fingerprint(checkpoint)),OUT/f'{arm}_latest.pt');status('training',arm=arm,step=step);print(arm,step,row['train']['pred_to_gt_px']['mean'],row['validation']['pred_to_gt_px']['mean'],flush=True)
            torch.save(dict(state=model.state_dict(),optimizer=optimizer.state_dict(),seed=seed,mode=mode,step=steps,base_sha256=fingerprint(checkpoint)),OUT/f'{arm}.pt')
            results.append(dict(arm=arm,seed=seed,mode=mode,**history[-1]));write_json(OUT/'partial_results.json',results)
            with torch.no_grad():predictions[arm]=model(data['pose'],spatial,controls).cpu().numpy()
    np.savez_compressed(OUT/'predictions.npz',**predictions);write_json(OUT/'results.json',dict(results=results,inspection_pending=True,not_promoted=True));status('complete')


if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(exist_ok=True);write_json(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()));raise
