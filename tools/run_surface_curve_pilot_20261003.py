"""Deadline-bounded three-arm pilot. No production promotion or heldout access."""
import argparse,collections,datetime as dt,fcntl,hashlib,json,os,random,sys,time,traceback
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'vggt-omega'),str(ROOT/'tools')]
import numpy as np
from PIL import Image
import torch
from torch.utils.data import default_collate
from training.finetune_blendedmvs import load_scene,BlendedMVSDataset,depth_loss
from training.instance_pose_losses import instance_targets_from_batch,InstancePoseLossWeights,hungarian_match_instances
from vggt_omega.models import VGGTOmega
from mousegeoforge.models.surface_curve_pose import SurfaceCurveDepthPoseModel
from mousegeoforge.surface_curve_instance_loss import surface_instance_loss
from mousegeoforge.surface_curve_supervision import points_to_polyline
from mousegeoforge.pilot_metrics import box_iou
from run_depth_pose_ablation import write_json,fingerprint,tensor_state,state_hash,move_batch,CORE23
BASE=ROOT/'output/surface_curve_cycle_20261003/training_v1'
DEPTH_ARGS=SimpleNamespace(semantic_label_weights='',body_depth_balance=True,body_depth_fraction=.5,depth_gradient_weight=0.)
WEIGHTS=InstancePoseLossWeights(local_heatmap=0.,keypoint_3d=0.,reprojection=0.)

def now():return dt.datetime.now(dt.timezone.utc)
def atomic_torch(path,payload):
    tmp=path.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(path)
def records(cfg):
    real=json.loads((BASE/'real_derived_report.json').read_text());assert real['passed']
    rows=[]
    for r in real['records']:
        ids=r['views'];pairs=[ids[i:i+2] for i in range(0,len(ids),2)]
        if len(pairs[-1])==1:pairs[-1].append(ids[0])
        for pair in pairs:rows.append(dict(r,views=pair,kind='real',image_hw=[768,704]))
    for group,kind in [('synthetic_train','synthetic'),('synthetic_development','synthetic'),('new_synthetic_train','curve')]:
        for r in cfg[group]:
            scene=load_scene(ROOT/r['scene']);ids=sorted(scene.frames)
            if kind=='curve':pairs=[ids[i:i+2] for i in range(0,len(ids),2)];pairs[-1]=pairs[-1] if len(pairs[-1])==2 else pairs[-1]+[ids[0]]
            else:pairs=[r.get('views',[0,2])]
            with Image.open(scene.frames[ids[0]].image_path) as im:w,h=im.size
            if kind=='curve':hw=[768,704]
            else:hw=[max(16,int(round(h*768/max(w,h)/16))*16),max(16,int(round(w*768/max(w,h)/16))*16)]
            for pair in pairs:
                assert len(pair)==2 and len(set(pair))==2 and set(pair)<=set(ids)
                rows.append(dict(r,views=pair,kind=kind,image_hw=hw))
    for i,r in enumerate(rows):r['key']=f'{i:04d}'
    assert not any(any('/'+s+'/' in r['scene'] or '/'+s+'_frame_' in r['scene'] for s in ('GR2','UR2','UR3')) for r in rows)
    return rows

def load_item(r):
    root=ROOT/r['scene'];scene=load_scene(root);ids=r['views'];h,w=r['image_hw']
    scene=replace(scene,frames={i:scene.frames[i] for i in ids},pair_map={ids[0]:ids[1:]})
    ds=BlendedMVSDataset([scene],len(ids),w,h,1,17,False,True,False,'median_baseline',6,True,normalization_scale_override=r.get('normalization_scale'))
    assert ds.keypoint_names==CORE23;ds.max_instances=2
    batch=default_collate([ds[0]]);batch['detection_annotation_complete']=torch.ones(1,len(ids),dtype=torch.bool)
    order=[int(x) for x in batch['view_ids'][0].split(',')];assert order==ids
    assert batch['camera_available'].all() and torch.isfinite(batch['images']).all()
    aux={}
    if r['kind']=='real':
        assert not batch['depth_available'].any() and not batch['instance_keypoint_visibility_valid'].any()
        with np.load(root/'surface_aux.npz') as a:
            indices=[a['view_ids'].tolist().index(i) for i in order]
            aux['tangent']=torch.zeros(1,len(ids),2,23,2);aux['tangent_valid']=torch.zeros(1,len(ids),2,23,dtype=torch.bool)
            aux['tangent'][0,:,0]=torch.tensor(a['tangent'][indices,0]);aux['tangent_valid'][0,:,0]=torch.tensor(a['tangent_valid'][indices,0])
    else:assert batch['depth_available'].all()
    if r['kind']=='curve':
        with np.load(root/'surface_curve.npz') as a:
            indices=[a['view_ids'].tolist().index(i) for i in order];uv=a['uv'][indices].copy()
            with Image.open(scene.frames[ids[0]].image_path) as im:sw,sh=im.size
            uv*=np.array([w/sw,h/sh]);aux['curve_target']=torch.zeros(1,len(ids),2,uv.shape[-2],2)
            aux['curve_valid']=torch.zeros(1,len(ids),2,uv.shape[-2],dtype=torch.bool)
            aux['curve_target'][0,:,0]=torch.tensor(uv[:,0]);aux['curve_valid'][0,:,0]=True
    files=[root/'keypoints.npz']+[scene.frames[i].image_path for i in ids]
    for i in ids:
        files += [p for p in [scene.frames[i].body_mask_path,scene.frames[i].cam_path,scene.frames[i].depth_path] if p is not None]
    for name in ['surface_aux.npz','surface_curve.npz','keypoint_validity.npz']:
        if (root/name).exists():files.append(root/name)
    return dict(record=r,batch=batch,aux=aux,source_hashes={p.relative_to(ROOT).as_posix():fingerprint(p) for p in files})

def forward(model,item):
    batch=move_batch(item['batch'],'cuda');aux=move_batch(item['aux'],'cuda')
    feats=[None if x is None else x.cuda() for x in item['cached']]
    return batch,aux,model.forward_cached(batch['images'],feats,item['patch_start'],coupling='off')

def loss_for(model,item,arm):
    batch,aux,pred=forward(model,item);target=instance_targets_from_batch(batch)
    pl,logs,_=surface_instance_loss(pred,target,WEIGHTS,image_size=batch['images'].shape[-2:],mode=arm,is_real=item['record']['kind']=='real',**aux)
    dl=pred['depth'].sum()*0
    if batch['depth_available'].any():dl=depth_loss(pred['depth'],batch['depths'],batch['depth_masks'],batch['body_masks'],batch['semantic_masks'],DEPTH_ARGS)['loss_depth']
    return pl+dl,{**logs,'loss_depth':dl}

def summary(values):
    a=np.asarray(values,float);a=a[np.isfinite(a)]
    return dict(n=len(a),mean=float(a.mean()) if len(a) else None,median=float(np.median(a)) if len(a) else None,p90=float(np.quantile(a,.9)) if len(a) else None)

@torch.no_grad()
def evaluate(model,rows,cache,deadline):
    model.eval();values=collections.defaultdict(list);details=[]
    for r in rows:
        if now()>=deadline:raise TimeoutError('Evaluation reached bounded deadline')
        item=cache(r);b,aux,out=forward(model,item);t=instance_targets_from_batch(b);h,w=b['images'].shape[-2:];scale=out['depth'].new_tensor([w,h])
        # Evaluation assignment is box/objectness-only, never GT point locations.
        match_target=dict(t);match_target['keypoint_valid']=torch.zeros_like(t['keypoint_valid'])
        matches=hungarian_match_instances(out,match_target);prefix=r['kind']+'_'+r['role']
        per=[]
        for v,(qs,ms) in enumerate(matches[0]):
            for q,m in zip(qs.tolist(),ms.tolist()):
                good=t['keypoint_valid'][0,v,m];delta=(out['instance_keypoint_xy'][0,v,q]-t['keypoint_xy'][0,v,m])*scale;err=delta.norm(dim=-1)
                reliable=good.clone();reliable[5:7]=False
                if r['kind']=='real':
                    select=torch.zeros_like(good);select[[0,14,15,16,17]]=True;reliable &= select
                values[prefix+'/reliable_px'].extend(err[reliable].tolist());values[prefix+'/back_point_px'].extend(err[good & (torch.arange(23,device='cuda')[:,None]==torch.tensor([5,6],device='cuda')).any(-1)].tolist())
                if 'tangent' in aux:
                    tv=aux['tangent_valid'][0,v,m]&good;tangent=aux['tangent'][0,v,m];normal=torch.stack([-tangent[:,1],tangent[:,0]],-1)
                    values[prefix+'/back_normal_px'].extend((delta*normal).sum(-1).abs()[tv].tolist());values[prefix+'/back_along_px'].extend((delta*tangent).sum(-1).abs()[tv].tolist())
                if t['box_valid'][0,v,m]:
                    iou=box_iou(out['instance_boxes_cxcywh'][0,v,q:q+1],t['boxes_cxcywh'][0,v,m:m+1])[0,0]
                    values[prefix+'/box_iou'].append(float(iou));values[prefix+'/detected'].append(float((iou>=.5)&(out['instance_objectness_logits'][0,v,q].sigmoid()>=.5)))
                if t['mask_valid'][0,v,m]:
                    pm=torch.nn.functional.interpolate(out['instance_mask_logits'][0,v,q][None,None],size=(h,w),mode='bilinear',align_corners=False)[0,0]>0;gm=t['masks'][0,v,m]
                    values[prefix+'/mask_iou'].append(float((pm&gm).sum()/(pm|gm).sum().clamp_min(1)))
                if 'curve_target' in aux and model.surface_curve_enabled:
                    pred=out['instance_surface_curve_xy'][0,v,q]*scale;gt=aux['curve_target'][0,v,m];valid=aux['curve_valid'][0,v,m]
                    dist=points_to_polyline(pred,gt,valid)[0].sqrt()
                    values[prefix+'/curve_to_gt_px'].extend(dist.tolist());values[prefix+'/curve_endpoints_px'].extend((pred[[0,-1]]-gt[[0,-1]]).norm(dim=-1).tolist())
                per.append(dict(view=r['views'][v],query=q,instance=m,reliable=summary(err[reliable].tolist()),back=summary(err[5:7][good[5:7]].tolist())))
        if b['depth_available'].any():
            d=out['depth'];d=d.squeeze(-1) if d.ndim==5 else d
            valid=b['depth_masks']&b['body_masks'];gt=b['depths']
            if valid.any():values[prefix+'/body_depth_absrel'].append(float(((d[valid]-gt[valid]).abs()/gt[valid].clamp_min(1e-6)).mean()))
        details.append(dict(key=r['key'],scene=r['scene'],views=r['views'],instances=per))
    return dict(metrics={k:summary(v) for k,v in values.items()},details=details)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,default=600);args=parser.parse_args()
    cfg=json.loads((BASE/'protocol.json').read_text());assert 0<args.steps<=cfg['initial_steps_per_arm']
    deadline=dt.datetime.fromisoformat(cfg['deadline_utc'].replace('Z','+00:00'))-dt.timedelta(minutes=30)
    assert now()<deadline
    lock=(ROOT/'output/teacher_temporal_cycle_20261001/gpu_job.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run=BASE/'pilot';run.mkdir(exist_ok=True);assert not (run/'complete.json').exists()
    def status(stage,**more):write_json(run/'status.json',dict(stage=stage,pid=os.getpid(),updated_utc=now().isoformat(),deadline_utc=deadline.isoformat(),**more))
    status('initializing');torch.set_num_threads(2);torch.manual_seed(cfg['seed']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    rows=records(cfg);write_json(run/'records.json',rows)
    assert fingerprint(ROOT/cfg['primary_checkpoint'])==cfg['checkpoint_sha256']
    for r in cfg['synthetic_train']:
        for p,h in r['annotation_sha256'].items():assert fingerprint(ROOT/p)==h
    for r in cfg['new_synthetic_train']:
        assert fingerprint(ROOT/r['scene']/'keypoints.npz')==r['keypoints_sha256'] and fingerprint(ROOT/r['scene']/'surface_curve.npz')==r['curve_sha256']
    spatial=VGGTOmega(enable_camera=False,enable_depth=True,enable_multi_instance_pose=True,num_keypoints=23)
    raw=torch.load(ROOT/'vggt-omega/vggt_omega_1b_512.pt',map_location='cpu',weights_only=True,mmap=True)
    selected={k:v for k,v in raw.items() if k.startswith(('aggregator.','dense_head.'))};result=spatial.load_state_dict(selected,strict=False)
    assert not result.unexpected_keys and all(k.startswith('instance_pose_head.') for k in result.missing_keys)
    del raw,selected
    model=SurfaceCurveDepthPoseModel.from_spatial_model(spatial).cuda().eval();del spatial
    checkpoint=torch.load(ROOT/cfg['primary_checkpoint'],map_location='cpu',weights_only=False,mmap=True)
    state=checkpoint['model'] if 'model' in checkpoint else checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    result=model.load_state_dict(state,strict=False)
    assert not result.unexpected_keys and all(k.startswith(('aggregator.','surface_curve_head.')) for k in result.missing_keys)
    del checkpoint,state
    initial=tensor_state(model);initial_hash=state_hash(initial);atomic_torch(run/'initial_heads.pt',initial)
    folder=run/'cache';folder.mkdir(exist_ok=True)
    for i,r in enumerate(rows):
        if now()>=deadline:raise TimeoutError('Cache deadline')
        item=load_item(r);path=folder/(r['key']+'.pt')
        if path.exists():
            old=torch.load(path,map_location='cpu',weights_only=False)
            assert old['record']==item['record'] and old['source_hashes']==item['source_hashes'];del old
        else:
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):features,patch_start=model.aggregator(item['batch']['images'].cuda())
            item.update(cached=[x.cpu() if x is not None and j in (4,11,17,23) else None for j,x in enumerate(features)],patch_start=patch_start)
            atomic_torch(path,item);del features
        status('caching',cached=i+1,total=len(rows));print('CACHE',i+1,len(rows),flush=True)
    memory=collections.OrderedDict();memory_bytes=0
    def cache(r):
        nonlocal memory_bytes
        k=r['key']
        if k not in memory:
            p=folder/(k+'.pt');size=p.stat().st_size
            while memory and memory_bytes+size>16*1024**3:
                _,(_,oldsize)=memory.popitem(last=False);memory_bytes-=oldsize
            memory[k]=(torch.load(p,map_location='cpu',weights_only=False),size);memory_bytes+=size
        memory.move_to_end(k);return memory[k][0]
    # Cache outputs must reproduce a live forward before any training.
    with torch.no_grad():
        b,_,out=forward(model,cache(rows[0]));live=model(b['images'])
        for k in out:torch.testing.assert_close(out[k],live[k],atol=0,rtol=0)
    buckets={kind:[r for r in rows if r['role']=='train' and r['kind']==kind] for kind in ('real','synthetic','curve')}
    rng=random.Random(cfg['seed']);schedule=[rng.choice(buckets[rng.choices(['real','synthetic','curve'],[.5,1/3,1/6])[0]])['key'] for _ in range(args.steps)]
    write_json(run/'schedule.json',dict(keys=schedule,sha256=hashlib.sha256(json.dumps(schedule).encode()).hexdigest(),initial_sha256=initial_hash))
    rowmap={r['key']:r for r in rows};ev=[r for r in rows if r['role']=='validation' or r['kind']=='curve']
    write_json(run/'provenance.json',dict(protocol_sha256=fingerprint(BASE/'protocol.json'),script_sha256=fingerprint(Path(__file__)),initial_sha256=initial_hash,live_cache_exact=True,steps=args.steps,eval_count=len(ev),gpu=torch.cuda.get_device_name(),limitations=[cfg['validation_limitation'],cfg['new_dataset_limitation'],'synthetic curve input resized to704x768 with K and UV scaled consistently; real is letterboxed','Pixel losses normalized by long side for all arms; not identical to old independent x/y normalization','Auxiliary curve supervised only by three synthetic poses, no full real curve GT']))
    for end in range(100,args.steps+100,100):
        end=min(end,args.steps)
        for arm in cfg['arms']:
            if now()>=deadline:status('deadline_stopped');return
            dest=run/arm;dest.mkdir(exist_ok=True);cp=dest/'latest.pt'
            previous=torch.load(cp,map_location='cpu',weights_only=False) if cp.exists() else None
            start=previous['step'] if previous else 0
            if start>=end:continue
            state=previous['model'] if previous else initial
            result=model.load_state_dict(state,strict=False);assert not result.unexpected_keys and all(k.startswith('aggregator.') for k in result.missing_keys)
            model.surface_curve_enabled=arm=='weak_points_curve6'
            model.surface_curve_head.requires_grad_(model.surface_curve_enabled)
            base_params=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('surface_curve_head.')]
            groups=[dict(params=base_params,lr=cfg['learning_rate'])]
            if model.surface_curve_enabled:groups.append(dict(params=list(model.surface_curve_head.parameters()),lr=cfg['curve_head_learning_rate']))
            optimizer=torch.optim.AdamW(groups,weight_decay=.01)
            torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
            if previous:
                optimizer.load_state_dict(previous['optimizer']);torch.set_rng_state(previous['torch_rng']);torch.cuda.set_rng_state_all(previous['cuda_rng'])
            else:write_json(dest/'eval_000000.json',evaluate(model,ev,cache,deadline))
            del previous,state
            model.train();history=[];began=time.time();status('training',arm=arm,step=start,block_end=end)
            for step in range(start+1,end+1):
                if now()>=deadline:break
                optimizer.zero_grad(set_to_none=True);loss,logs=loss_for(model,cache(rowmap[schedule[step-1]]),arm)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],cfg['gradient_clip'])
                if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradients')
                optimizer.step();history.append(dict(step=step,loss=float(loss),grad_norm=float(norm),**{k:float(v.detach()) for k,v in logs.items()}))
                if step%10==0:status('training',arm=arm,step=step,block_end=end,seconds_per_step=(time.time()-began)/len(history));print(arm,step,float(loss),flush=True)
            achieved=history[-1]['step'] if history else start
            atomic_torch(cp,dict(model=tensor_state(model),optimizer=optimizer.state_dict(),step=achieved,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),initial_sha256=initial_hash))
            write_json(dest/f'history_{start+1:06d}_{achieved:06d}.json',history)
            if now()>=deadline:status('deadline_stopped',arm=arm,step=achieved);return
            status('evaluating',arm=arm,step=achieved);write_json(dest/f'eval_{achieved:06d}.json',evaluate(model,ev,cache,deadline))
            del optimizer;torch.cuda.empty_cache()
        if end==args.steps:break
    write_json(run/'complete.json',dict(completed_utc=now().isoformat(),steps_per_arm=args.steps,arms=cfg['arms'],status='pilot_complete_not_promoted'))
    status('complete',steps_per_arm=args.steps)

if __name__=='__main__':
    try:main()
    except Exception:
        (BASE/'pilot').mkdir(exist_ok=True)
        write_json(BASE/'pilot'/'failure.json',dict(utc=now().isoformat(),traceback=traceback.format_exc()));raise
