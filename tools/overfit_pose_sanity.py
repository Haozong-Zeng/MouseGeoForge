"""Eight-scene pose-only overfit diagnostic, not held-out evaluation.

The A/B pair differs only in local-heatmap loss weight; a third diagnostic
changes the sampling-grid gradient route instead. Frozen encoder
features and fixed depth predictions are cached for reuse; depth is neither
trained nor used by pose. Full-label detection, boxes and masks remain active.
"""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_depth_pose_ablation import (ROOT,torch,np,VGGTOmega,DepthPoseModel,MultiInstanceMouseHead,
    SharedTokenAdapter,load_batch,move_batch,write_json,fingerprint,state_hash,tensor_state,
    instance_targets_from_batch,instance_pose_loss,InstancePoseLossWeights,render_comparison,build_manifest)
from training.instance_pose_losses import hungarian_match_instances
from mousegeoforge.pilot_metrics import evaluate_scene,summarize_records


def pose_forward(model,item):
    batch=move_batch(item['batch'],'cuda')
    cached=[None if x is None else x.cuda() for x in item['cached']]
    adapted=model.shared_adapter(cached,item['patch_start'])
    out=model.instance_pose_head(adapted,batch['images'],item['patch_start'])
    # Read-only depth is included only for compatible plotting/metric helpers;
    # depth metrics are removed from this diagnostic's reported results.
    out['depth']=item['fixed_depth'].cuda()
    return batch,out


def inspect_pose(out,batch,target):
    metric,assignment=evaluate_scene(out,batch,target)
    metric={k:v for k,v in metric.items() if 'depth' not in k}
    values={k:[] for k in ('oracle_nme','coarse_nme','window_coverage','heatmap_peak','heatmap_entropy')}
    h,w=batch['images'].shape[-2:]; scale=out['depth'].new_tensor([w,h])
    for v,(ps,ts) in enumerate(hungarian_match_instances(out,target)[0]):
        for q,t in zip(ps.tolist(),ts.tolist()):
            gt=target['keypoint_xy'][0,v,t]; valid=target['keypoint_valid'][0,v,t]
            coarse=out['instance_keypoint_coarse_xy'][0,v,q]
            final=out['instance_keypoint_xy'][0,v,q]
            diag=(target['boxes_cxcywh'][0,v,t,2:]*scale).norm().clamp_min(1.)
            for name,p in (('oracle_nme',final),('coarse_nme',coarse)):
                values[name].extend((((p-gt)*scale).norm(dim=-1)/diag)[valid].detach().cpu().tolist())
            values['window_coverage'].extend(((gt-coarse).abs()<=.1).all(-1)[valid].float().cpu().tolist())
            prob=out['instance_keypoint_local_heatmap_logits'][0,v,q][valid].flatten(-2).softmax(-1)
            values['heatmap_peak'].extend(prob.max(-1).values.cpu().tolist())
            values['heatmap_entropy'].extend((-(prob*prob.clamp_min(1e-9).log()).sum(-1)/math.log(prob.shape[-1])).cpu().tolist())
    metric.update({k:float(np.mean(v)) for k,v in values.items()})
    return metric,assignment


def evaluate(model,data,visuals=None):
    model.eval(); rows=[]
    with torch.no_grad():
        for i,item in enumerate(data):
            batch,out=pose_forward(model,item)
            target=instance_targets_from_batch(batch)
            metrics,matches=inspect_pose(out,batch,target)
            rows.append(dict(scene=item['record']['scene'],source_group=item['record']['source_group'],metrics=metrics))
            if visuals is not None:
                render_comparison(batch,out,target,matches,visuals/f'scene_{i:02d}.png')
    return dict(summary=summarize_records(rows),records=rows)


def gradient_audit(model,logs,weights):
    params=list(model.instance_pose_head.coarse_keypoint_xy.parameters())
    params+=list(model.shared_adapter.parameters())
    terms={'coordinate':(logs['loss_instance_keypoint_l1'],weights.keypoint_l1),
           'local_heatmap':(logs['loss_instance_local_heatmap'],weights.local_heatmap),
           'detection_box_mask':(logs['loss_instance_objectness']*weights.objectness+
                 logs['loss_instance_box_l1']*weights.box_l1+logs['loss_instance_box_giou']*weights.box_giou+
                 logs['loss_instance_mask_bce']*weights.mask_bce+logs['loss_instance_mask_dice']*weights.mask_dice,1.)}
    vectors={}; result={}
    for name,(loss,weight) in terms.items():
        grads=torch.autograd.grad(loss*weight,params,retain_graph=True,allow_unused=True)
        vector=torch.cat([(torch.zeros_like(p) if g is None else g).flatten() for p,g in zip(params,grads)])
        vectors[name]=vector
        result[name+'_gradient_norm']=float(vector.norm())
    a,b=vectors['coordinate'],vectors['local_heatmap']
    result['coordinate_heatmap_gradient_cosine']=float((a@b)/(a.norm()*b.norm()).clamp_min(1e-12))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant',choices=['legacy','no_local_heatmap','detached_grid'],required=True)
    p.add_argument('--steps',type=int,default=2000)
    p.add_argument('--seed',type=int,default=17)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if not 1<=a.steps<=3000: raise ValueError('Bounded diagnostic: 1..3000 steps')
    if a.output.exists(): raise FileExistsError('Use a new output directory')
    a.output.mkdir(parents=True)
    torch.set_num_threads(2); torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    # Reconstruct the registered split so reproduction does not depend on old output artifacts.
    manifest=build_manifest(json.loads((ROOT/'configs/experiments/depth_pose_pilot_v1.json').read_text()))
    groups={}
    for r in manifest['records']:
        if r['split']=='train': groups.setdefault(r['source_group'],[]).append(r)
    records=[rs[i] for rs in groups.values() for i in np.linspace(0,len(rs)-1,4).round().astype(int)]
    assert len(records)==8 and {r['expected_instances'] for r in records}=={1,2}
    write_json(a.output/'manifest.json',dict(purpose='training_set_overfit_only',records=records))
    spatial=VGGTOmega(enable_camera=False,enable_depth=True,enable_multi_instance_pose=True,num_keypoints=23)
    checkpoint=ROOT/'vggt-omega/vggt_omega_1b_512.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    selected={k:v for k,v in state.items() if k.startswith(('aggregator.','dense_head.'))}
    expected={k for k in spatial.state_dict() if k.startswith(('aggregator.','dense_head.'))}
    assert set(selected)==expected
    loaded=spatial.load_state_dict(selected,strict=False)
    assert not loaded.unexpected_keys and all(k.startswith('instance_pose_head.') for k in loaded.missing_keys)
    del state,selected
    model=DepthPoseModel.from_spatial_model(spatial).cuda().eval(); del spatial
    torch.manual_seed(a.seed)
    model.instance_pose_head=MultiInstanceMouseHead(num_keypoints=23,
        detach_local_sampling_grid=a.variant=='detached_grid').cuda()
    model.shared_adapter=SharedTokenAdapter(2048).cuda()
    model.dense_head.requires_grad_(False); model.depth_pose_bridge.requires_grad_(False)
    model.eval()
    initial_hash=state_hash(tensor_state(model))
    data=[]
    for r in records:
        batch=load_batch(r)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            cached,start=model.aggregator(batch['images'].cuda())
        with torch.no_grad():
            adapted=model.shared_adapter(cached,start)
            fixed_depth,_=model.dense_head(adapted,batch['images'].cuda(),start)
        data.append(dict(record=r,batch=batch,patch_start=start,fixed_depth=fixed_depth.cpu(),
                         cached=[x.cpu() if x is not None and i in (4,11,17,23) else None for i,x in enumerate(cached)]))
    weights=InstancePoseLossWeights(keypoint_3d=0.,reprojection=0.,local_heatmap=0. if a.variant=='no_local_heatmap' else 1.)
    rng=random.Random(a.seed); schedule=[]
    while len(schedule)<a.steps:
        ids=list(range(len(data))); rng.shuffle(ids); schedule.extend(ids)
    schedule=schedule[:a.steps]
    config=dict(variant=a.variant,seed=a.seed,steps=a.steps,weights=asdict(weights),
                depth_training=False,coupling=False,frames=8,views=2,learning_rate=1e-4,
                detach_local_sampling_grid=a.variant=='detached_grid',
                initial_state_sha256=initial_hash,checkpoint_sha256=fingerprint(checkpoint),
                schedule=schedule,criterion='NME < .05, PCK@.1 >= .9, detection recall >= .9')
    write_json(a.output/'config.json',config)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01)
    history=[]; evaluations=[]; began=time.time()
    initial=evaluate(model,data); evaluations.append(dict(step=0,**initial))
    write_json(a.output/'evaluations.json',evaluations)
    for step,index in enumerate(schedule,1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        batch,out=pose_forward(model,data[index]); targets=instance_targets_from_batch(batch)
        loss,logs=instance_pose_loss(out,targets,weights)  # 2D-only: no unused geometry calculation
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
        audit=gradient_audit(model,logs,weights) if step==1 or step%100==0 else {}
        loss.backward()
        assert all(p.grad is None for p in model.aggregator.parameters())
        assert all(p.grad is None for p in model.dense_head.parameters())
        assert all(p.grad is None for p in model.depth_pose_bridge.parameters())
        norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
        optimizer.step()
        history.append(dict(step=step,loss=float(loss.detach()),gradient_norm=float(norm),
                            components={k:float(v.detach()) for k,v in logs.items() if k.startswith('loss_')},**audit))
        if step%100==0 or step==a.steps:
            result=evaluate(model,data)
            evaluations.append(dict(step=step,**result))
            write_json(a.output/'evaluations.json',evaluations); write_json(a.output/'history.json',history)
            print(json.dumps(dict(variant=a.variant,step=step,elapsed_seconds=time.time()-began,
                                  **result['summary']['source_macro'])),flush=True)
    final=evaluate(model,data,visuals=a.output/'visuals')
    saved=tensor_state(model)
    torch.save(dict(model=saved,optimizer=optimizer.state_dict(),config=config,step=a.steps,
                    cpu_rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state()),a.output/'final.pt')
    loaded=torch.load(a.output/'final.pt',map_location='cpu',weights_only=True)
    assert state_hash(loaded['model'])==state_hash(saved)
    model.load_state_dict(loaded['model'],strict=False)
    repeated=evaluate(model,data)
    assert repeated==final
    m=final['summary']['source_macro']
    passed=m['box_matched_nme']<.05 and m['box_matched_pck_0.1']>=.9 and m['detection_recall']>=.9
    write_json(a.output/'result.json',dict(config=config,overfit_passed=passed,reload_exact=True,
                                         final=final,elapsed_seconds=time.time()-began))
    print(f'COMPLETE variant={a.variant} overfit_passed={passed}',flush=True)


if __name__=='__main__': main()
