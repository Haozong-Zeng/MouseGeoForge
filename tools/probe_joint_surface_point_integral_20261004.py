"""Shared spatial keys: synthetic fullcurve replay vs replay+realpoint auxiliary."""
import collections,datetime as dt,fcntl,json,math,os,random,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import numpy as np
import torch
from mousegeoforge.surface_curve_supervision import curve_loss,material_curve_loss,points_to_polyline
from torch.nn import functional as F
from mousegeoforge.models.joint_surface_point_integral import JointSurfacePointIntegralProbe
from probe_diverse_curve_features_20261003 import FeatureProbe
from run_depth_pose_ablation import fingerprint,write_json
CYCLE=ROOT/'output/surface_curve_cycle_20261003'
SYN=CYCLE/'expanded_curve_features_v1'
REAL=CYCLE/'real_curve_adaptation_inputs_v1'
OUT=CYCLE/'joint_surface_point_integral_v1'


def main():
    assert not OUT.exists() and json.loads((REAL/'status.json').read_text())['stage']=='complete'
    lock=(ROOT/'output/teacher_temporal_cycle_20261001/gpu_job.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert torch.cuda.is_available();OUT.mkdir();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    deadline=dt.datetime(2026,10,4,8,5,16,tzinfo=dt.timezone.utc)
    def status(stage,**kw):write_json(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),**kw))
    status('loading')
    saved={k:torch.load(p/'frozen_inputs.pt',map_location='cpu',weights_only=False) for k,p in [('synthetic',SYN),('real',REAL)]}
    data={k:{name:value.cuda() for name,value in v['tensors'].items()} for k,v in saved.items()}
    meta={k:v['metadata'] for k,v in saved.items()}
    train={k:[i for i,m in enumerate(mm) if m['role']=='train'] for k,mm in meta.items()}
    val={k:[i for i,m in enumerate(mm) if m['role']=='validation'] for k,mm in meta.items()}
    real_eligible=[i for i in train['real'] if data['real']['observed_valid'][i].any()]
    assert train['synthetic'] and real_eligible and val['synthetic'] and val['real']
    assert all(m['session'] not in ('GR2','UR2','UR3') for m in meta['real'])
    point_saved=torch.load(CYCLE/'back_point_probe_inputs_v1/real_inputs.pt',map_location='cpu',weights_only=False)
    assert point_saved['metadata']==meta['real'] and torch.equal(point_saved['tensors']['pose'],saved['real']['tensors']['pose'])
    data['real']['coarse_xy']=point_saved['tensors']['coarse_xy'].cuda()
    spatial={};memory_manifests={'synthetic':CYCLE/'curve_spatial_memory_v2/manifest.json','real':CYCLE/'real_point_spatial_memory_v1/manifest.json'}
    for kind,path in memory_manifests.items():
        manifest=json.loads(path.read_text());maps=[None]*len(meta[kind])
        for record in manifest['records']:
            assert fingerprint(ROOT/record['path'])==record['sha256']
            cached=torch.load(ROOT/record['path'],map_location='cpu',weights_only=False)
            memory=F.interpolate(cached['memory'].float(),(48,48),mode='bilinear',align_corners=False)
            for bind in record['bindings']:
                i=bind['observation'];assert maps[i] is None;maps[i]=memory[bind['view_index']]
        assert all(m is not None for m in maps);spatial[kind]=torch.stack(maps).cuda()
    controls={}
    seeds=[17,29,43];steps=1600;real_weight=.25
    write_json(OUT/'protocol.json',dict(seeds=seeds,steps=steps,synthetic_batch=8,real_batch=8,real_weight=real_weight,
        learning_rate='cosine1e-4to1e-5',weight_decay=.001,arms=['replay','joint_points'],
        initialization='Botharms SAME per-seed pointhead from integral_back_point spatial_strict1600(real+synthetictraining), curvequery from integral_surface_curve spatial1600. Pointhead key/position shared withcurvehead. Curvepredictions canchange atinit; pointpredictions mustremainidentical.',
        objective='Synthetic fullknownmaterialcurveHuber+.1shape(order0); jointarm adds.25strictreal06/07Huber toSEPARATEpointoutput. No realcurvepointdistance training.',
        sharing='Only spatialkey andexplicitpositionprojection shared. Point andcurvequeries separate. Realpointloss hasno gradient tocurvequery. Originalencoder/Core23mainheads frozen.',
        sampling='Uniformsyntheticdataset andeligible realTRAINsession; pairedseed+2000, identicalbatchschedule.',
        point_supervision='Realmanual06/07 finitepointtargets, notmaterialfractions, virtualendpoints orfullcurvelabels. NO pseudolabels.',
        provenance={k:fingerprint(p/'frozen_inputs.pt') for k,p in [('synthetic',SYN),('real',REAL)]},
        initialization_hashes={str(seed):{kind:fingerprint(CYCLE/path) for kind,path in [('point',f'integral_back_point_probe_v1/spatial_strict_seed{seed}_latest.pt'),('curve',f'integral_surface_curve_probe_v1/spatial_seed{seed}.pt')]} for seed in seeds},
        memory_hashes={k:fingerprint(p) for k,p in memory_manifests.items()},
        constraints='Botharms have realpointpretraining exposure. Replay is NOT synthetic-only, not compared as a cleanzero-realtransfer. No lockedholdout, labels/sourceimagesunchanged. No productionpromotion.',
        limitations='Weakrealpoint-to-curve metric evaluatedonly, notfullcurveaccuracy. Checksyntheticretention, actualpointheadaccuracy, extensions/crossings. Sameassetdevelopmentvalidationnotexternaltest.',
        hard_deadline=deadline.isoformat()))
    valid=torch.ones(65,device='cuda',dtype=torch.bool);u=torch.linspace(0,1,65,device='cuda')

    def predict(model,kind,ids):
        d=data[kind];return model.forward_curve(d['pose'][ids],spatial[kind][ids],controls[kind][ids])

    def syn_loss(model,ids):
        d=data['synthetic'];p=predict(model,'synthetic',ids)*d['pixels'][ids,None];g=d['gt'][ids]*d['pixels'][ids,None]
        return torch.stack([material_curve_loss(pp,gg,valid,u,material_parameter_known=True,scale=768)+.1*curve_loss(pp,gg,valid,full_curve=True,scale=768,order_weight=0.) for pp,gg in zip(p,g)]).mean()

    def real_loss(model,ids):
        d=data['real'];p=model.forward_points(d['pose'][ids],spatial['real'][ids],d['coarse_xy'][ids])*d['pixels'][ids,None]
        g=d['observed_xy'][ids]*d['pixels'][ids,None]
        return torch.stack([F.smooth_l1_loss(pp[v]/768,gg[v]/768,beta=.01) for pp,gg,v in zip(p,g,d['observed_valid'][ids])]).mean()

    def evaluate(model,kind,ids):
        d=data[kind];values=collections.defaultdict(list)
        with torch.no_grad():
            norm=predict(model,kind,ids);pred=norm*d['pixels'][ids,None]
            point_pred=model.forward_points(d['pose'][ids],spatial[kind][ids],d['coarse_xy'][ids])*d['pixels'][ids,None] if kind=='real' else None
            for j,(i,p) in enumerate(zip(ids,pred)):
                length=(p[1:]-p[:-1]).norm(dim=-1).sum();chord=(p[-1]-p[0]).norm()
                values['curve_length_px'].append(float(length));values['length_to_chord'].append(float(length/chord.clamp_min(1e-6)))
                if kind=='synthetic':
                    g=d['gt'][i]*d['pixels'][i]
                    values['pred_to_gt_px'].extend(points_to_polyline(p,g,valid)[0].sqrt().tolist())
                    values['gt_to_pred_px'].extend(points_to_polyline(g,p,torch.ones(33,device='cuda',dtype=torch.bool))[0].sqrt().tolist())
                    values['endpoints_px'].extend((p[[0,-1]]-g[[0,-1]]).norm(dim=-1).tolist())
                else:
                    good=d['observed_valid'][i];g=d['observed_xy'][i]*d['pixels'][i]
                    if good.any():
                        values['actual_point_head_error_px'].extend((point_pred[j,good]-g[good]).norm(dim=-1).tolist())
                        dist,param=points_to_polyline(g[good],p,torch.ones(33,device='cuda',dtype=torch.bool));values['observed_point_to_curve_px'].extend(dist.sqrt().tolist())
                        if good.all():values['observed_order_reversal'].append(float(param[0]>param[1]))
                    if d['mask_valid'][i]:
                        xy=(norm[j]*norm.new_tensor([176,192])).round().long();inside=(xy[:,0]>=0)&(xy[:,0]<176)&(xy[:,1]>=0)&(xy[:,1]<192)
                        mask=d['mask_small'][i];membership=mask[xy[:,1].clamp(0,191),xy[:,0].clamp(0,175)]&inside
                        values['mask_membership_fraction_approx'].append(float(membership.float().mean()))
        return {k:dict(n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),p90=float(np.quantile(v,.9))) for k,v in values.items()}

    def evaluate_all(model):
        return {kind+'_'+role:evaluate(model,kind,ids[kind]) for kind in ['synthetic','real'] for role,ids in [('train',train),('validation',val)]}

    syn_buckets=collections.defaultdict(list);real_buckets=collections.defaultdict(list)
    for i in train['synthetic']:syn_buckets[meta['synthetic'][i]['dataset']].append(i)
    for i in real_eligible:real_buckets[meta['real'][i]['session']].append(i)
    skeys=sorted(syn_buckets);rkeys=sorted(real_buckets);results=[];predictions={};gradient_reports=[]
    for seed in seeds:
        rng=random.Random(seed+2000)
        schedule=[([rng.choice(syn_buckets[rng.choice(skeys)]) for _ in range(8)],[rng.choice(real_buckets[rng.choice(rkeys)]) for _ in range(8)]) for _ in range(steps)]
        checkpoint=CYCLE/f'curve_data_scaling_v1/expanded_seed{seed}.pt'
        frozen=FeatureProbe(False).cuda().eval();frozen.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['state']);frozen.requires_grad_(False)
        with torch.no_grad():
            for kind,d in data.items():
                controls[kind]=d['boxes'][:,:2,None].transpose(-1,-2)+(d['base_logits'].sigmoid()-.5)*d['boxes'][:,2:,None].transpose(-1,-2)+.5*frozen.delta(d['pose']).reshape(-1,6,2).tanh()
        del frozen
        point_state=torch.load(CYCLE/f'integral_back_point_probe_v1/spatial_strict_seed{seed}_latest.pt',map_location='cpu',weights_only=False)['state']
        curve_state=torch.load(CYCLE/f'integral_surface_curve_probe_v1/spatial_seed{seed}.pt',map_location='cpu',weights_only=False)['state']
        initial_metrics=None
        for mode in ['replay','joint_points']:
            arm=f'{mode}_seed{seed}';model=JointSurfacePointIntegralProbe(point_state,curve_state).cuda()
            optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.001)
            history=[dict(step=0,**evaluate_all(model))];status('training',arm=arm,step=0)
            if initial_metrics is None:initial_metrics=history[0]
            else:assert history[0]==initial_metrics
            expected=json.loads((CYCLE/f'integral_back_point_probe_v1/spatial_strict_seed{seed}_history.json').read_text())[-1]['real_validation']['point_error_px']['mean']
            assert abs(history[0]['real_validation']['actual_point_head_error_px']['mean']-expected)<1e-4
            if mode=='joint_points':
                sl=syn_loss(model,schedule[0][0]);rl=real_loss(model,schedule[0][1]);params=list(model.parameters())
                sg=torch.autograd.grad(sl,params,allow_unused=True);rg=torch.autograd.grad(rl,params,allow_unused=True)
                sn=float(torch.sqrt(sum(g.square().sum() for g in sg if g is not None)));rn=float(torch.sqrt(sum(g.square().sum() for g in rg if g is not None)))
                gradient_reports.append(dict(seed=seed,synthetic_norm=sn,real_unweighted_norm=rn,weighted_ratio=real_weight*rn/max(sn,1e-12),real_weight=real_weight));write_json(OUT/'initial_gradient_audit.json',gradient_reports)
            for step,(sids,rids) in enumerate(schedule,1):
                assert dt.datetime.now(dt.timezone.utc)<deadline
                lr=1e-5+(1e-4-1e-5)*.5*(1+math.cos(math.pi*(step-1)/steps))
                for group in optimizer.param_groups:group['lr']=lr
                optimizer.zero_grad(set_to_none=True);sl=syn_loss(model,sids);rl=real_loss(model,rids)
                loss=sl+(real_weight if mode=='joint_points' else 0.)*rl
                assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm);optimizer.step()
                if step%400==0:
                    row=dict(step=step,lr=lr,loss=float(loss.detach()),synthetic_loss=float(sl.detach()),real_point_loss=float(rl.detach()),**evaluate_all(model));history.append(row)
                    write_json(OUT/f'{arm}_history.json',history);status('training',arm=arm,step=step)
                    torch.save(dict(state=model.state_dict(),optimizer=optimizer.state_dict(),seed=seed,mode=mode,step=step,initial_sha256=fingerprint(checkpoint)),OUT/f'{arm}_latest.pt')
                    print(arm,step,'synval',row['synthetic_validation']['pred_to_gt_px']['mean'],'realval',row['real_validation']['observed_point_to_curve_px']['mean'],flush=True)
            results.append(dict(arm=arm,seed=seed,mode=mode,**history[-1]));write_json(OUT/'partial_results.json',results)
            with torch.no_grad():
                for kind in ['synthetic','real']:predictions[arm+'__'+kind]=predict(model,kind,list(range(len(meta[kind])))).cpu().numpy()
    np.savez_compressed(OUT/'predictions.npz',**predictions);write_json(OUT/'results.json',dict(results=results,inspection_pending=True,no_production_promotion=True));status('complete')


if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(exist_ok=True);write_json(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()));raise
