"""Reproducible, bounded J-off/J-on spatial pilot on audited Blender data.

No data modifications, no test-set access, no validation checkpoint selection.
Frozen RGB features are computed once and kept in RAM, never mixed across views.
This is a small full-label experiment, not the production mixed-label trainer.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'vggt-omega')]
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import default_collate
from training.finetune_blendedmvs import BlendedMVSDataset, load_scene, depth_loss
from training.instance_pose_losses import instance_targets_from_batch, instance_pose_loss, InstancePoseLossWeights
from vggt_omega.models import VGGTOmega
from vggt_omega.models.heads.instance_pose_head import MultiInstanceMouseHead
from mousegeoforge.models import DepthPoseModel
from mousegeoforge.models.adapters import SharedTokenAdapter, LocalDepthFeatureBridge
from mousegeoforge.pilot_metrics import evaluate_scene, summarize_records
from mousegeoforge.pilot_schedule import learning_rates

DUAL = 'Homecage-BlackBrownFur-LightningPose-CRIM13-DiversePose400-DepthHeightV1'
CORE23 = ['nose','head_center','left_ear','right_ear','neck','spine_center','lumbar_spine',
          'tail_base','tail_q1','tail_q2','tail_q3','tail_q4','tail_q5','tail_tip',
          'left_forepaw_center','right_forepaw_center','left_hindpaw_center','right_hindpaw_center',
          'left_elbow','right_elbow','left_knee','right_knee','abdomen_center']


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def fingerprint(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def spaced(items, count):
    if count > len(items):
        raise ValueError('Insufficient distinct source frames')
    return [items[i] for i in np.linspace(0, len(items)-1, count).round().astype(int)]


def pilot_image_hw(long_side, dual):
    if long_side not in (256,512,768):
        raise ValueError('Audited resolution ladder: 256, 512 or 768')
    return [long_side*9//16 if dual else long_side,long_side]


def build_manifest(config):
    plan_path = ROOT/'dataset/Homecage/LightningPose-CRIM13-Diverse400.plan.json'
    plan = json.loads(plan_path.read_text())
    by_session = {}
    for key in plan['keyframes']:
        session = Path(key['source_image'].replace('\\','/')).parent.name
        by_session.setdefault(session, []).append(key)
    sessions = ['021709_A25_Block22_BCfe1_t','021309_A29_Block16_BCma1_t']
    assert set(by_session) == set(sessions)
    result = []
    for split, session, action in zip(('train','val'), sessions, ('ClimbWall','WashFace')):
        count = config[f'{split}_frames_per_source']
        for key in spaced(sorted(by_session[session],key=lambda k:k['source_image_frame']), count):
            result.append(dict(split=split,source_group=f'crim13/{session}',
                               dataset=DUAL,frame=key['scene_frame'],
                               source_frame=key['source_image_frame'],expected_instances=2,
                               image_hw=pilot_image_hw(config['long_side'],True),depth_units='millimeter'))
        dataset = f'PyMouseStudio-BrownFur-{action}-Core23'
        group = 'mouse_climb_wall_91' if action == 'ClimbWall' else 'mouse_wash_face_91'
        groups = json.loads((ROOT/'dataset/motion_source_groups.json').read_text())['groups']
        assert groups[group]['split'] == split
        assert f'BlenderPyMouse/{dataset}' in groups[group]['members']
        for frame in spaced(list(range(1,92)), count):
            result.append(dict(split=split,source_group=group,dataset=dataset,frame=int(frame),
                               source_frame=int(frame),expected_instances=1,
                               image_hw=pilot_image_hw(config['long_side'],False),depth_units='Blender world units'))
    groups = {s:{r['source_group'] for r in result if r['split']==s} for s in ('train','val')}
    assert not groups['train'] & groups['val']
    for r in result:
        r['scene'] = f"dataset/BlenderPyMouse/{r['dataset']}/frame_{r['frame']:06d}"
        r['views'] = config['view_ids']
        r['pfm_flip_y'] = config['pfm_flip_y']
        r['detection_annotation_complete'] = True
    assert len({r['scene'] for r in result}) == len(result)
    return dict(config=config, source_plan_sha256=fingerprint(plan_path),
                grouping_sha256=fingerprint(ROOT/'dataset/motion_source_groups.json'),
                limitations=['same synthetic assets and rigs across splits',
                             'one held-out motion group and one held-out CRIM13 session',
                             'no real-image or unseen-animal generalization claim'],records=result)


def load_batch(record):
    scene = load_scene(ROOT/record['scene'])
    scene = replace(scene, frames={i:scene.frames[i] for i in record['views']}, pair_map={})
    h,w = record['image_hw']
    ds = BlendedMVSDataset([scene],2,w,h,1,17,False,record['pfm_flip_y'],False,
                          'median_baseline',2,True)
    if ds.keypoint_names != CORE23:
        raise ValueError(f"Unexpected keypoint order: {ds.keypoint_names}")
    ds.max_instances = 2
    batch = default_collate([ds[0]])
    batch['detection_annotation_complete'] = torch.ones(1,2,dtype=torch.bool)
    if int(batch['actual_instance_count'][0]) != record['expected_instances']:
        raise ValueError('Instance count mismatch')
    if not batch['depth_available'].all() or not batch['camera_available'].all():
        raise ValueError('Pilot requires complete depth and calibration')
    if not batch['depth_masks'].any(-1).any(-1).all():
        raise ValueError('Empty depth view')
    valid = batch['instance_valid']
    if not (batch['mask_valid'][valid].all() and batch['box_valid'][valid].all()):
        raise ValueError('Pilot requires full instance boxes and masks')
    if not batch['instance_keypoint_valid'][valid].any(-1).all():
        raise ValueError('Missing instance keypoints')
    if not valid.sum(-1).eq(record['expected_instances']).all():
        raise ValueError('Selected views must contain all annotated mice')
    return batch


def move_batch(batch, device):
    return {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}


def depth_alignment_audit(batch):
    # Compare annotated visible surface anchors, not internal joint depth.
    world = batch['surface_world_xyz_instances'][0]
    valid = batch['instance_keypoint_valid'][0] & batch['instance_keypoint_visible'][0]
    uv = batch['instance_keypoint_xy'][0]
    h,w = batch['images'].shape[-2:]
    residuals = [[],[]]
    for v in range(2):
        e = batch['extrinsics'][0,v]
        z = (world @ e[:3,:3].T + e[:3,3])[...,2]
        x = (uv[v,...,0]*w).round().long().clamp(0,w-1)
        y = (uv[v,...,1]*h).round().long().clamp(0,h-1)
        for flip in (0,1):
            yy = h-1-y if flip else y
            d = batch['depths'][0,v,yy,x]
            mask = valid[v] & torch.isfinite(d) & (d>0) & (z>0)
            residuals[flip].extend(((d-z).abs()/z.clamp_min(1e-6))[mask].tolist())
    return {'surface_anchor_absrel_median':float(np.median(residuals[0])) if residuals[0] else None,
            'flipped_surface_anchor_absrel_median':float(np.median(residuals[1])) if residuals[1] else None,
            'surface_anchor_count':len(residuals[0])}


def tensor_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items() if not k.startswith('aggregator.')}


def state_hash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        h.update(k.encode()); h.update(v.contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def restore(model, state):
    result=model.load_state_dict(state,strict=False)
    assert not result.unexpected_keys and all(k.startswith('aggregator.') for k in result.missing_keys)


def cached_forward(model, item, mode):
    batch=move_batch(item['batch'],'cuda')
    features=[None if x is None else x.cuda() for x in item['cached']]
    output=model.forward_cached(batch['images'],features,item['patch_start'],coupling=mode)
    return batch,output


def evaluate(model, data, mode, split, visuals=None):
    model.eval()
    records=[]
    seen=Counter()
    with torch.no_grad():
        for item in data:
            if item['record']['split'] != split:
                continue
            batch,out=cached_forward(model,item,mode)
            target=instance_targets_from_batch(batch)
            metrics,assignment=evaluate_scene(out,batch,target)
            r=item['record']
            records.append(dict(scene=r['scene'],source_group=r['source_group'],metrics=metrics))
            if visuals is not None and seen[r['source_group']] < 2:
                render_comparison(batch,out,target,assignment,visuals/f"{r['dataset']}_{r['frame']:06d}.png")
            seen[r['source_group']]+=1
    return dict(summary=summarize_records(records),records=records)


def render_comparison(batch,out,target,assignment,path):
    # Scientific diagnostic: RGB + GT/predicted landmarks + common-scale depth.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,4,figsize=(14,7),layout='constrained')
    gt_depth=batch['depths'][0].cpu().numpy()
    pd=out['depth'][0,...,0].cpu().numpy()
    valid=batch['depth_masks'][0].cpu().numpy()
    lo,hi=np.quantile(gt_depth[valid],[.01,.99])
    for v in range(2):
        rgb=batch['images'][0,v].permute(1,2,0).cpu().numpy()
        h,w=rgb.shape[:2]
        for c in (0,1): axes[v,c].imshow(rgb)
        for q,m in zip(*assignment[v]):
            color=['#ff4757','#24c8ff'][m%2]
            good=target['keypoint_valid'][0,v,m].cpu().numpy()
            for c,xy in ((0,target['keypoint_xy'][0,v,m]),(1,out['instance_keypoint_xy'][0,v,q])):
                p=xy.cpu().numpy()*[w,h]
                axes[v,c].scatter(p[good,0],p[good,1],s=10,c=color)
                for chain in ([0,1,4,5,6,7],[7,8,9,10,11,12,13],[4,18,14],[4,19,15],[6,20,16],[6,21,17]):
                    if good[chain].all(): axes[v,c].plot(p[chain,0],p[chain,1],c=color,lw=.7)
        axes[v,2].imshow(np.where(valid[v],gt_depth[v],np.nan),vmin=lo,vmax=hi,cmap='viridis')
        axes[v,3].imshow(pd[v],vmin=lo,vmax=hi,cmap='viridis')
        for c,title in enumerate(('GT Core23','Prediction (box-matched)','GT depth / rig scale','Pred depth / same scale')):
            axes[v,c].set_title(title); axes[v,c].axis('off')
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=130); plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/experiments/depth_pose_pilot_v1.json')
    p.add_argument('--output',type=Path,default=ROOT/'output/depth_pose_ablation_pilot_v1')
    p.add_argument('--checkpoint',type=Path,default=ROOT/'vggt-omega/vggt_omega_1b_512.pt')
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--run-seed',type=int,help='Run one configured seed in an independent worker directory')
    args=p.parse_args()
    cfg=json.loads(args.config.read_text())
    if args.run_seed is not None:
        if args.run_seed not in cfg['seeds']: raise ValueError('Seed not in registered config')
        cfg['seeds']=[args.run_seed]
    lr_values=learning_rates(cfg)
    if cfg['long_side'] not in (256,512,768) or cfg['view_ids']!=[0,2]:
        raise ValueError('This pilot supports the registered resolution ladder and views [0,2]')
    if not cfg['modes'] or len(set(cfg['modes'])) != len(cfg['modes']):
        raise ValueError('At least one distinct mode is required')
    if (args.output/'results.json').exists():
        raise FileExistsError('Completed result exists; choose a new output directory')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    manifest=build_manifest(cfg)
    write_json(args.output/'manifest.json',manifest)
    data=[]
    for n,r in enumerate(manifest['records']):
        batch=load_batch(r)
        audit=depth_alignment_audit(batch)
        if audit['surface_anchor_absrel_median'] is None or audit['surface_anchor_absrel_median']>.15:
            raise ValueError(f"Depth/annotation alignment needs inspection: {r['scene']} {audit}")
        data.append(dict(record=r,batch=batch,audit=audit))
        if n%16==0: print(f'Loaded/audited {n+1}/{len(manifest["records"])} scenes',flush=True)
    write_json(args.output/'data_audit.json',[
        dict(scene=i['record']['scene'],view_order=i['batch']['view_ids'],scale=float(i['batch']['scale'][0]),
             body_pixels=int(i['batch']['body_masks'].sum()),**i['audit']) for i in data])
    if args.prepare_only:
        print('PREPARE PASS',flush=True); return
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    start=time.time()
    torch.manual_seed(17)
    spatial=VGGTOmega(enable_camera=False,enable_depth=True,enable_multi_instance_pose=True,num_keypoints=23)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True,mmap=True)
    expected={k for k in spatial.state_dict() if k.startswith(('aggregator.','dense_head.'))}
    selected={k:v for k,v in state.items() if k.startswith(('aggregator.','dense_head.'))}
    assert set(selected)==expected
    loaded=spatial.load_state_dict(selected,strict=False)
    assert not loaded.unexpected_keys and all(k.startswith('instance_pose_head.') for k in loaded.missing_keys)
    del selected,state
    model=DepthPoseModel.from_spatial_model(spatial).cuda().eval()
    del spatial
    original_depth={k:v.cpu().clone() for k,v in model.dense_head.state_dict().items()}
    # Exact per-scene cache: same tensors serve every mode and seed. No labels enter encoder.
    for n,item in enumerate(data):
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            cached,start_index=model.aggregator(item['batch']['images'].cuda())
        item['cached']=[x.detach().cpu() if x is not None and i in (4,11,17,23) else None for i,x in enumerate(cached)]
        item['patch_start']=start_index
        if n%16==0: print(f'Cached RGB {n+1}/{len(data)}',flush=True)
    # Audit live-vs-cache equality before training (all fields, both modes).
    first=data[0]
    with torch.no_grad():
        for mode in cfg['modes']:
            batch,out=cached_forward(model,first,mode)
            live=model(batch['images'],coupling=mode)
            for k in out: torch.testing.assert_close(out[k],live[k],atol=0,rtol=0)
    depth_args=SimpleNamespace(semantic_label_weights='',body_depth_balance=cfg['body_depth_balance'],
                               body_depth_fraction=.5,depth_gradient_weight=0.)
    weights=InstancePoseLossWeights(keypoint_3d=0.,reprojection=0.,
                                   local_heatmap=cfg.get('local_heatmap_weight',1.))
    results=[]
    provenance=dict(checkpoint_sha256=fingerprint(args.checkpoint),loaded_tensors=len(expected),
                    torch_version=torch.__version__,gpu=torch.cuda.get_device_name(),
                    git_commit=__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                    deterministic_cudnn=True,cuda_grid_sample_backward_bitwise_determinism_not_guaranteed=True,
                    live_cache_exact_parity=True)
    write_json(args.output/'provenance.json',provenance)
    for seed in cfg['seeds']:
        torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
        model.dense_head.load_state_dict(original_depth)
        model.instance_pose_head=MultiInstanceMouseHead(num_keypoints=23).cuda()
        model.shared_adapter=SharedTokenAdapter(2048).cuda()
        model.depth_pose_bridge=LocalDepthFeatureBridge(256,model.dense_head.proj.in_channels).cuda()
        initial=tensor_state(model)
        initial_hash=state_hash(initial)
        train=[i for i in data if i['record']['split']=='train']
        schedule=[]; rng=random.Random(seed)
        while len(schedule)<cfg['steps']:
            ids=list(range(len(train))); rng.shuffle(ids); schedule.extend(ids)
        schedule=schedule[:cfg['steps']]
        order_hash=hashlib.sha256(json.dumps([train[i]['record']['scene'] for i in schedule]).encode()).hexdigest()
        for mode in cfg['modes']:
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            restore(model,initial)
            assert state_hash(tensor_state(model))==initial_hash
            run=args.output/f'seed{seed}_{mode}'
            run.mkdir(parents=True,exist_ok=True)
            write_json(run/'learning_rate_schedule.json',lr_values)
            optimizer=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],
                lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
            before=evaluate(model,data,mode,'val')
            write_json(run/'initial_validation.json',before)
            history=[]; training_curve=[]; began=time.time(); model.train()
            for step,index in enumerate(schedule,1):
                for group in optimizer.param_groups:
                    group['lr']=lr_values[step-1]
                optimizer.zero_grad(set_to_none=True)
                batch,out=cached_forward(model,train[index],mode)
                target=instance_targets_from_batch(batch)
                dl=depth_loss(out['depth'],batch['depths'],batch['depth_masks'],batch['body_masks'],batch['semantic_masks'],depth_args)['loss_depth']
                pl,logs=instance_pose_loss(out,target,weights,image_size=batch['images'].shape[-2:])
                loss=cfg['depth_weight']*dl+cfg['pose_weight']*pl
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
                loss.backward()
                gn=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True)
                if step==1:
                    gradient_audit={}
                    for name in ('shared_adapter','dense_head','instance_pose_head','depth_pose_bridge'):
                        gs=[v.grad for v in getattr(model,name).parameters() if v.grad is not None]
                        gradient_audit[name]=sum(float(g.abs().sum()) for g in gs)
                    assert all(v.grad is None for v in model.aggregator.parameters())
                    assert (gradient_audit['depth_pose_bridge']>0)==(mode!='off')
                    assert all(gradient_audit[k]>0 for k in ('shared_adapter','dense_head','instance_pose_head'))
                optimizer.step()
                history.append(dict(step=step,scene=train[index]['record']['scene'],loss=float(loss.detach()),
                                    depth_loss=float(dl.detach()),pose_loss=float(pl.detach()),gradient_norm=float(gn),
                                    learning_rate=lr_values[step-1]))
                if step%25==0 or step==1:
                    print(json.dumps(dict(seed=seed,mode=mode,elapsed_s=round(time.time()-began),**history[-1])),flush=True)
                    write_json(run/'progress.json',dict(seed=seed,mode=mode,step=step,total_steps=cfg['steps'],elapsed_s=time.time()-began))
                interval=cfg.get('train_eval_interval',0)
                if interval and (step%interval==0 or step==cfg['steps']):
                    training_curve.append(dict(step=step,learning_rate=lr_values[step-1],**evaluate(model,data,mode,'train')['summary']))
                    write_json(run/'training_curve.json',training_curve)
                    write_json(run/'history.json',history)
                    print('TRAIN METRICS '+json.dumps(training_curve[-1]),flush=True)
                    model.train()
            after=evaluate(model,data,mode,'val',visuals=run/'visuals')
            train_eval=evaluate(model,data,mode,'train')
            diagnostic=evaluate(model,data,'off','val') if mode!='off' else None
            final_state=tensor_state(model)
            torch.save(dict(model=final_state,config=cfg,coupling=mode,seed=seed,
                            optimizer=optimizer.state_dict(),step=cfg['steps'],
                            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                            checkpoint_sha256=provenance['checkpoint_sha256']),run/'final_heads.pt')
            result=dict(seed=seed,mode=mode,steps=cfg['steps'],initial_state_sha256=initial_hash,
                        lr_schedule_sha256=hashlib.sha256(json.dumps(lr_values).encode()).hexdigest(),
                        train_order_first5000_sha256=hashlib.sha256(json.dumps([train[i]['record']['scene'] for i in schedule[:5000]]).encode()).hexdigest(),
                        train_order_sha256=order_hash,gradient_audit=gradient_audit,
                        initial_validation=before['summary'],validation=after['summary'],
                        training=train_eval['summary'],elapsed_seconds=time.time()-began,
                        bridge_disabled_at_eval=diagnostic['summary'] if diagnostic else None)
            write_json(run/'history.json',history); write_json(run/'validation.json',after)
            write_json(run/'result.json',result)
            results.append(result)
            write_json(args.output/'partial_results.json',results)
            print('RUN COMPLETE '+json.dumps(result),flush=True)
            del optimizer,final_state
    for seed in cfg['seeds']:
        pair=[r for r in results if r['seed']==seed]
        assert len({r['initial_state_sha256'] for r in pair})==1
        assert len({r['train_order_sha256'] for r in pair})==1
        assert all(r['initial_validation']==pair[0]['initial_validation'] for r in pair)
    write_json(args.output/'results.json',dict(config=cfg,provenance=provenance,runs=results,
               passed=True,elapsed_seconds=time.time()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated()))
    print('ALL CONFIGURED RUNS COMPLETE',flush=True)


if __name__=='__main__': main()
