"""Bounded equal-budget predicted-mask prior OFF/ON synthetic GT pilot."""
import copy
import fcntl
import json
import os
import random
import time
from datetime import datetime,timezone
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from run_depth_pose_ablation import ROOT,VGGTOmega,DepthPoseModel,fingerprint,write_json,instance_targets_from_batch
from train_cylinder_view_balance import oriented_batch
from mousegeoforge.pilot_metrics import evaluate_scene
from mousegeoforge.adaptive_temperature import install_adaptive_temperature,restore_adaptive
from mousegeoforge.spatial_anchor_experiment import SpatialAnchorResidual,install_spatial_anchor,spatial_anchor_context
from mousegeoforge.mask_box_candidate import tensor_mapping_sha256

OUT=ROOT/'output/spatial_anchor_pilot_v1'
PARENT=ROOT/'output/cylinder_adaptive_temperature_v1/protocol.json'
BASE=ROOT/'output/cylinder_adaptive_temperature_v1/adaptive_temperature/final_heads.pt'
DEADLINE=datetime(2026,9,12,17,19,27,tzinfo=timezone.utc).timestamp()
STEPS=200


def choose(spec):
    keys=list(dict.fromkeys(s[0] for s in spec['arms']['adaptive_temperature']))
    selected=[]
    for kind,quota in [('cylinder',8),('single',4),('dual',4)]:
        seen=set()
        for key in keys:
            r=spec['records'][key]
            good=(r['source_group']=='mouse_climb_wall_91' and 'bottom' in r['physical_roles']) if kind=='cylinder' else r['expected_instances']==2 if kind=='dual' else r['source_group']=='moseq_sample10'
            if not good or len(r['views'])>3 or r['scene'] in seen:continue
            selected.append(key);seen.add(r['scene'])
            if len(seen)==quota:break
        assert len(seen)==quota
    evaluation=[]
    for group in ('bottom_only','old_cylinder_side_pair','old_dual_pair','sides_bottom'):
        pool=[e for e in spec['evaluation'] if e['group']==group]
        evaluation.extend(pool[i] for i in np.linspace(0,len(pool)-1,4).round().astype(int))
    return selected,evaluation


def local_refine(sample,coarse):
    n,q,k=coarse.shape[:3];size=sample['size'];radius=sample['radius']
    axis=torch.linspace(-radius,radius,size,device=coarse.device,dtype=coarse.dtype)
    y,x=torch.meshgrid(axis,axis,indexing='ij');offset=torch.stack((x,y),-1)
    grid=(coarse[:,:,:,None,None]*2-1+offset[None,None,None]*2).reshape(n,q*k*size,size,2)
    pixel=F.grid_sample(sample['local_pixel'],grid,mode='bilinear',padding_mode='zeros',align_corners=False)
    pixel=pixel.view(n,-1,q,k,size,size).permute(0,2,3,1,4,5)
    logits=torch.einsum('nqkd,nqkdhw->nqkhw',sample['local_query'],F.normalize(pixel,dim=3,eps=1e-6))
    logits=logits*sample['scale'][...,None,None]
    probs=logits.flatten(-2).softmax(-1).view_as(logits)
    return (coarse+(probs[...,None]*offset).sum((-3,-2))).clamp(0,1)


def loss_fn(refined,coarse,s):
    losses=[]
    for v,(queries,instances) in enumerate(s['matches']):
        for q,m in zip(queries,instances):
            valid=s['target']['keypoint_valid'][0,v,m].to(refined.device)
            gt=s['target']['keypoint_xy'][0,v,m].to(refined.device)
            if valid.any():
                losses.append(F.smooth_l1_loss(refined[v,q,valid],gt[valid],beta=.05)+.5*F.smooth_l1_loss(coarse[v,q,valid],gt[valid],beta=.05))
    if not losses:raise ValueError('No GT supervision')
    return torch.stack(losses).mean()


def main():
    began=time.monotonic();torch.set_num_threads(2);cv2.setNumThreads(1);torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with (ROOT/'output/cylinder_training.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);OUT.mkdir(exist_ok=False)
        def status(state,**kwargs):write_json(OUT/'status.json',dict(state=state,pid=os.getpid(),elapsed_s=time.monotonic()-began,**kwargs))
        def deadline():
            if time.time()>=DEADLINE or time.monotonic()-began>1800:raise TimeoutError('30-minute pilot / user deadline')
        status('preparing');hooks=[]
        try:
            spec=json.loads(PARENT.read_text());train_keys,evaluation=choose(spec)
            assert all(spec['records'][k]['split']=='train' for k in train_keys)
            assert not {spec['records'][k]['source_group'] for k in train_keys}&{spec['records'][e['key']]['source_group'] for e in evaluation}
            encoder=ROOT/'vggt-omega/vggt_omega_1b_512.pt'
            assert fingerprint(encoder)==spec['encoder_sha256']
            code_paths=['tools/train_spatial_anchor_pilot.py','mousegeoforge/spatial_anchor_experiment.py','mousegeoforge/spatial_heatmap_probe.py']
            source_paths={ROOT/spec['records'][k]['scene'] for k in train_keys+[e['key'] for e in evaluation]}
            # Hash labels/calibration/metadata and RGB file lists without modifying sources.
            sources={str(p.relative_to(ROOT)):{str(f.relative_to(p)):dict(size=f.stat().st_size,sha256=fingerprint(f)) for f in sorted(p.rglob('*')) if f.is_file()} for p in source_paths}
            protocol=dict(seed=17,steps_per_arm=STEPS,lr=.001,weight_decay=.01,clip_grad=1.,
                arms=['mask_prior_off','mask_prior_on'],train_keys=train_keys,evaluation=evaluation,records={k:spec['records'][k] for k in set(train_keys+[e['key'] for e in evaluation])},
                base_sha256=fingerprint(BASE),encoder_sha256=fingerprint(encoder),source_files=sources,
                code_sha256={p:fingerprint(ROOT/p) for p in code_paths},parent_sha256=fingerprint(PARENT),
                loss='mean per matched instance SmoothL1 refined +0.5 coarse, beta0.05, all valid synthetic Core23 incl occluded; no pseudo labels',
                matching='Fixed original box+objectness assignment, no target keypoint matching',
                comparison='Both arms train identical spatial residuals; only finite soft predicted mask prior differs. Frozen baseline is NOT an equal-budget trained control.',
                budget='Fixed final200, checkpoints100/200 descriptive only, no best-dev selection. Max30min and user cutoff.',
                decision='Pilot learnability: report training and independent-source synthetic development, body/tail and fixed detections. Do not promote. Consider paired extension only if trends support and side/multimouse retention is acceptable.',
                real_training=False,holdout_accessed=False,production_promoted=False)
            write_json(OUT/'protocol.json',protocol)
            spatial=VGGTOmega(enable_camera=False,enable_depth=True,enable_multi_instance_pose=True,num_keypoints=23)
            weights=torch.load(encoder,map_location='cpu',weights_only=True,mmap=True)
            loaded=spatial.load_state_dict({k:v for k,v in weights.items() if k.startswith(('aggregator.','dense_head.'))},strict=False)
            assert not loaded.unexpected_keys and all(k.startswith('instance_pose_head.') for k in loaded.missing_keys)
            model=install_adaptive_temperature(DepthPoseModel.from_spatial_model(spatial)).cuda().eval()
            restore_adaptive(model,torch.load(BASE,map_location='cpu',weights_only=True)['model']);del weights,spatial
            base_hash=tensor_mapping_sha256(model.state_dict())
            # This isolated experiment deliberately freezes all original weights;
            # no legacy candidate/flag-sensitive replay is used here.
            model.requires_grad_(False)
            install_spatial_anchor(model);model.eval()
            branch=model.instance_pose_head.spatial_anchor
            prefix='instance_pose_head.spatial_anchor.'
            head=model.instance_pose_head
            checks=[]
            for cams in (['A'],['D','E','A']):
                deadline()
                rgb=[cv2.cvtColor(cv2.imread(str(ROOT/f'output/real_cylinder_bottom_frozen_v1/review/AZL3_000811_{c}_input.png')),cv2.COLOR_BGR2RGB) for c in cams]
                image=torch.from_numpy(np.stack(rgb)).permute(0,3,1,2)[None].cuda().float()/255
                with torch.no_grad():
                    off=model(image,coupling='off')
                    with spatial_anchor_context(model,enabled=True):on=model(image,coupling='off')
                differences={k:float((off[k]-on[k]).abs().max()) for k in off}
                assert max(differences.values())==0
                with spatial_anchor_context(model,enabled=True):
                    probe=model(image,coupling='off')
                    probe['instance_keypoint_xy'].square().mean().backward()
                assert branch.gate.grad is not None and torch.isfinite(branch.gate.grad).all() and branch.gate.grad.abs().sum()>0
                assert all(p.grad is None for n,p in model.named_parameters() if not n.startswith(prefix))
                checks.append(dict(views=len(cams),image_hw=list(image.shape[-2:]),zero_max_difference=max(differences.values()),gate_grad_norm=float(branch.gate.grad.norm())))
                branch.zero_grad(set_to_none=True);del off,on,probe,image
            write_json(OUT/'preflight.json',dict(passed=True,gpu_checks=checks,optimizer_steps=0))
            captured={}
            def save(name,inputs=False):
                def hook(module,args,value):
                    captured[name]=value.detach()
                    if inputs:captured['tokens']=args[0].detach()
                return hook
            hooks=[head.pyramid_refine[0].register_forward_hook(save('pixels')),
                   head.coarse_keypoint_xy.register_forward_hook(save('raw',True)),
                   head.local_pixel_projection.register_forward_hook(save('local_pixel')),
                   head.local_query_projection.register_forward_hook(save('local_query'))]
            def sample(record):
                deadline();batch=oriented_batch(record,num_views=len(record['views']));image=batch['images'].cuda()
                with torch.no_grad():pred=model(image,coupling='off')
                pred_cpu={k:v.cpu() for k,v in pred.items()};target=instance_targets_from_batch(batch)
                metrics,matches=evaluate_scene(pred_cpu,batch,target)
                s={k:v for k,v in captured.items()}
                s.update(mask=pred['instance_mask_logits'][0].detach(),local_query=F.normalize(s['local_query'],dim=-1,eps=1e-6),
                    scale=head.last_temperature_scale.detach(),size=head.local_heatmap_size,radius=head.local_window_radius,
                    matches=matches,target=target,baseline=pred_cpu,batch=batch,metrics=metrics)
                with torch.no_grad():replayed=local_refine(s,s['raw'].sigmoid())
                torch.testing.assert_close(replayed,pred['instance_keypoint_xy'][0],atol=1e-6,rtol=1e-6)
                captured.clear();return s
            cache={}
            for i,key in enumerate(train_keys):
                cache[key]=sample(spec['records'][key]);status('caching_train',records=i+1,total=len(train_keys))
            eval_cache={}
            for e in evaluation:eval_cache[e['key']]=sample(spec['records'][e['key']])
            for h in hooks:h.remove()
            hooks=[]
            torch.manual_seed(17);initial=copy.deepcopy(branch.state_dict());histories={};results={}
            rng=random.Random(17);sequence=[]
            while len(sequence)<STEPS:
                chunk=list(train_keys);rng.shuffle(chunk);sequence.extend(chunk)
            sequence=sequence[:STEPS];write_json(OUT/'schedule.json',sequence)
            def output(s,use_prior):
                raw=branch(s['raw'],s['tokens'],s['pixels'],s['mask'],enabled=True,use_mask_prior=use_prior)
                coarse=raw.sigmoid();return coarse,local_refine(s,coarse)
            def evaluate(use_prior):
                records=[]
                with torch.no_grad():
                    for e in evaluation:
                        s=eval_cache[e['key']];coarse,refined=output(s,use_prior)
                        pred=dict(s['baseline'],instance_keypoint_xy=refined[None].cpu(),instance_keypoint_coarse_xy=coarse[None].cpu())
                        metric,assignment=evaluate_scene(pred,s['batch'],s['target'])
                        assert assignment==s['matches']
                        records.append(dict(group=e['group'],key=e['key'],metrics=metric))
                return records
            def fixed_loss(use_prior):
                with torch.no_grad():return float(np.mean([float(loss_fn(*reversed(output(s,use_prior)),s)) for s in cache.values()]))
            for arm in ('mask_prior_off','mask_prior_on'):
                use_prior=arm.endswith('_on');branch.load_state_dict(initial);branch.train()
                opt=torch.optim.AdamW(branch.parameters(),lr=.001,weight_decay=.01)
                rows=[];initial_loss=fixed_loss(use_prior)
                for step,key in enumerate(sequence,1):
                    deadline();s=cache[key];opt.zero_grad(set_to_none=True)
                    coarse,refined=output(s,use_prior);loss=loss_fn(refined,coarse,s)
                    assert torch.isfinite(loss);loss.backward()
                    grad=torch.nn.utils.clip_grad_norm_(branch.parameters(),1.);assert torch.isfinite(grad)
                    opt.step();rows.append(dict(step=step,loss=float(loss.detach()),grad_norm=float(grad)))
                    if step%25==0:
                        status('training',arm=arm,step=step,total=STEPS);write_json(OUT/f'{arm}_history.json',rows);print('TRAIN',arm,step,rows[-1]['loss'],flush=True)
                    if step in (100,200):
                        branch.eval();dev=evaluate(use_prior)
                        write_json(OUT/f'{arm}_eval_{step}.json',dev)
                        torch.save(dict(branch={k:v.cpu() for k,v in branch.state_dict().items()},optimizer=opt.state_dict(),step=step),OUT/f'{arm}_step{step}.pt')
                        branch.train()
                branch.eval();histories[arm]=rows
                results[arm]=dict(initial_fixed_loss=initial_loss,final_fixed_loss=fixed_loss(use_prior),records=evaluate(use_prior))
            base_after={k:v for k,v in model.state_dict().items() if not k.startswith(prefix)}
            assert tensor_mapping_sha256(base_after)==base_hash
            assert all(p.grad is None for n,p in model.named_parameters() if not n.startswith(prefix))
            assert fingerprint(BASE)==protocol['base_sha256']
            assert all(fingerprint(ROOT/p)==h for p,h in protocol['code_sha256'].items())
            baseline=[dict(group=e['group'],key=e['key'],metrics=eval_cache[e['key']]['metrics']) for e in evaluation]
            write_json(OUT/'results.json',dict(baseline=baseline,arms=results))
            write_json(OUT/'verification.json',dict(passed=True,base_weights_unchanged=True,base_gradients_none=True,
                steps_per_arm=STEPS,new_parameters=sum(p.numel() for p in branch.parameters()),
                training_records=len(cache),evaluation_records=len(evaluation),production_promoted=False))
            status('completed');print('COMPLETED',flush=True)
        except BaseException as e:status('failed',error=repr(e));raise
        finally:
            for h in hooks:h.remove()


if __name__=='__main__':main()
