"""Three-seed spatial-local versus global-pool residual refinement control."""
import collections, datetime as dt, fcntl, json, math, os, random, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import numpy as np
import torch
from torch.nn import functional as F
from mousegeoforge.models.surface_curve_refinement import SurfaceCurveSpatialRefiner
from mousegeoforge.surface_curve_supervision import cubic_bspline,curve_loss,material_curve_loss,points_to_polyline
from probe_diverse_curve_features_20261003 import FeatureProbe
from run_depth_pose_ablation import write_json,fingerprint
CYCLE=ROOT/'output/surface_curve_cycle_20261003'
SOURCE=CYCLE/'expanded_curve_features_v1'
MEMORY=CYCLE/'curve_spatial_memory_v2'
BASE=CYCLE/'curve_data_scaling_v1'
OUT=CYCLE/'curve_spatial_refinement_v1'


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
    seeds=[17,29,43];steps=1600
    write_json(OUT/'protocol.json',dict(seeds=seeds,steps=steps,batch_size=8,learning_rate='cosine3e-4to1e-5',weight_decay=.001,
        arms=['global','local'],frozen_base='Corresponding expanded_seed checkpoint, fixed2400step; same per-seed base in both arms',
        image_features='Frozen decoder spatial memory incl position at patch16, fp16cache then float32resize48x48. No GT feature sampling.',
        local_sampling='Five-position cross at current predicted curve locations, radius0.04normalized; three shared residual iterations',
        global_control='Same parameters/init/coordinate inputs, image memory globally pooled instead of locally sampled',
        supervision='Same synthetic materialHuber+0.1shape(order0); no real fractional correspondence',
        schedule='Uniformdataset then uniformtrainobservation; randomseed+1000; same schedule per local/global pair',
        no_production_change=True,no_validation_checkpoint_selection=True,hard_deadline=deadline.isoformat(),
        inputs_sha256=fingerprint(SOURCE/'frozen_inputs.pt'),memory_manifest_sha256=fingerprint(MEMORY/'manifest.json'),
        limitations='Development synthetic validation only. Same extra optimization budget per arm; compares image localization, not overall fine-tuning vs frozen baseline at equal total updates.'))
    valid=torch.ones(65,device='cuda',dtype=torch.bool);u=torch.linspace(0,1,65,device='cuda')
    def evaluate(model,controls,ids):
        values=collections.defaultdict(list)
        with torch.no_grad():
            pred=model(data['pose'][ids],spatial[ids],controls[ids])*data['pixels'][ids,None]
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
        for mode in ['global','local']:
            arm=f'{mode}_seed{seed}';torch.manual_seed(seed)
            model=SurfaceCurveSpatialRefiner(mode).cuda()
            if init_state is None:init_state={k:v.detach().clone() for k,v in model.state_dict().items()}
            else:assert all(torch.equal(v,init_state[k]) for k,v in model.state_dict().items())
            optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.001)
            history=[dict(step=0,train=evaluate(model,controls,train),validation=evaluate(model,controls,val))]
            baseline=json.loads((BASE/f'expanded_seed{seed}_history.json').read_text())[-1]
            assert abs(history[0]['validation']['pred_to_gt_px']['mean']-baseline['validation']['pred_to_gt_px']['mean'])<1e-5
            status('training',arm=arm,step=0)
            for step,ids in enumerate(schedule,1):
                assert dt.datetime.now(dt.timezone.utc)<deadline
                lr=1e-5+(3e-4-1e-5)*.5*(1+math.cos(math.pi*(step-1)/steps))
                for group in optimizer.param_groups:group['lr']=lr
                optimizer.zero_grad(set_to_none=True)
                p=model(data['pose'][ids],spatial[ids],controls[ids])*data['pixels'][ids,None];g=data['gt'][ids]*data['pixels'][ids,None]
                loss=torch.stack([material_curve_loss(pp,gg,valid,u,material_parameter_known=True,scale=768)+.1*curve_loss(pp,gg,valid,full_curve=True,scale=768,order_weight=0.) for pp,gg in zip(p,g)]).mean()
                assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm);optimizer.step()
                if step%400==0:
                    row=dict(step=step,lr=lr,loss=float(loss.detach()),train=evaluate(model,controls,train),validation=evaluate(model,controls,val));history.append(row)
                    write_json(OUT/f'{arm}_history.json',history);status('training',arm=arm,step=step);print(arm,step,row['train']['pred_to_gt_px']['mean'],row['validation']['pred_to_gt_px']['mean'],flush=True)
            torch.save(dict(state=model.state_dict(),optimizer=optimizer.state_dict(),seed=seed,mode=mode,step=steps,base_sha256=fingerprint(checkpoint)),OUT/f'{arm}.pt')
            results.append(dict(arm=arm,seed=seed,mode=mode,**history[-1]));write_json(OUT/'partial_results.json',results)
            with torch.no_grad():predictions[arm]=model(data['pose'],spatial,controls).cpu().numpy()
    np.savez_compressed(OUT/'predictions.npz',**predictions);write_json(OUT/'results.json',dict(results=results,inspection_pending=True,not_promoted=True));status('complete')


if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(exist_ok=True);write_json(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()));raise
