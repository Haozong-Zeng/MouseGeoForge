"""Bounded equal-budget predicted-mask prior OFF/ON synthetic GT pilot."""
import copy
import fcntl
import json
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
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
from mousegeoforge.affine_batch_experiment import augment_batch
from mousegeoforge.visible_heatmap_supervision import matched_visible_loss
PREFLIGHT=ROOT/'output/spatial_anchor_affine_batch_preflight_v1_retry1'

OUT=ROOT/'output/spatial_anchor_visible_heatmap_v1'
PLAN=ROOT/'output/spatial_anchor_diverse32_plan_v1.json'
PARENT=ROOT/'output/cylinder_adaptive_temperature_v1/protocol.json'
BASE=ROOT/'output/cylinder_adaptive_temperature_v1/adaptive_temperature/final_heads.pt'
DEADLINE=datetime(2026,9,12,17,19,27,tzinfo=timezone.utc).timestamp()
STEPS=800


def choose(spec):
    plan=json.loads(PLAN.read_text())
    original=json.loads((ROOT/'output/spatial_anchor_pilot_v1/protocol.json').read_text())
    assert fingerprint(ROOT/'output/spatial_anchor_pilot_v1/protocol.json')==plan['parent_protocol_sha256']
    assert set(original['train_keys'])<=set(plan['train_keys']) and len(plan['train_keys'])==32
    spec['records'].update(plan['records'])
    assert plan['evaluation']==original['evaluation']
    return plan['train_keys'],plan['evaluation']


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


from mousegeoforge.deterministic_local_sampling import deterministic_local_refine as local_refine


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
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
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
            code_paths=['tools/train_spatial_anchor_visible_heatmap.py','mousegeoforge/visible_heatmap_supervision.py','tools/train_spatial_anchor_augmentation_deterministic.py','mousegeoforge/deterministic_local_sampling.py','tools/train_spatial_anchor_augmentation.py','mousegeoforge/affine_batch_experiment.py','mousegeoforge/affine_augmentation.py','tools/train_spatial_anchor_diverse32.py','tools/prepare_spatial_anchor_diverse32.py','mousegeoforge/spatial_anchor_experiment.py','mousegeoforge/spatial_heatmap_probe.py']
            source_paths={ROOT/spec['records'][k]['scene'] for k in train_keys+[e['key'] for e in evaluation]}
            # Hash labels/calibration/metadata and RGB file lists without modifying sources.
            sources={str(p.relative_to(ROOT)):{str(f.relative_to(p)):dict(size=f.stat().st_size,sha256=fingerprint(f)) for f in sorted(p.rglob('*')) if f.is_file()} for p in source_paths}
            protocol=dict(seed=17,steps_per_arm=STEPS,lr=.001,weight_decay=.01,clip_grad=1.,
                arms=['heatmap_off','heatmap_on'],train_keys=train_keys,evaluation=evaluation,records={k:spec['records'][k] for k in set(train_keys+[e['key'] for e in evaluation])},
                sampling_plan_sha256=fingerprint(PLAN),base_sha256=fingerprint(BASE),encoder_sha256=fingerprint(encoder),source_files=sources,
                code_sha256={p:fingerprint(ROOT/p) for p in code_paths},parent_sha256=fingerprint(PARENT),
                loss='mean per matched instance SmoothL1 refined +0.5 coarse, beta0.05, all valid synthetic Core23 incl occluded; no pseudo labels',
                matching='Fixed original box+objectness assignment, no target keypoint matching',
                comparison='Both arms same mask prior ON and same augmented RGB schedule; only visible-only auxiliary Gaussian KL differs, weight0 vs0.005 sigma8 network pixels. All valid coordinate L1 incl occluded/tail preserved.',
                budget='Fixed final800; checkpoints200/400/800 descriptive only; no architecture/loss change; no best-dev selection. Max30min/user cutoff.',
                decision='Pilot learnability: report training and independent-source synthetic development, body/tail and fixed detections. Do not promote. Consider paired extension only if trends support and side/multimouse retention is acceptable.',
                real_training=False,holdout_accessed=False,production_promoted=False)
            assert json.loads((PREFLIGHT/'verification.json').read_text())['passed']
            aug_plan=json.loads((PREFLIGHT/'plan.json').read_text())
            assert all(fingerprint(ROOT/p)==h for p,h in aug_plan['code_sha256'].items())
            protocol['augmentation_plan_sha256']=fingerprint(PREFLIGHT/'plan.json')
            protocol['augmentation_variants']=aug_plan['variants']
            protocol['rejected_crops']=sum(not r['accepted'] for r in aug_plan['records'])
            protocol['control_replay_guard']='Heatmap OFF800 branch AND AdamW must exactly replay previous deterministic augmentation ON800 before Heatmap ON training'
            protocol['auxiliary_loss']=dict(kind='visible-only normalized Gaussian KL',weight=.005,sigma_network_pixels=8,all_visible_points_mean=True,inference_unchanged=True)
            protocol['extra_control_replay_steps']=0
            protocol['numerics']='deterministic algorithms required; gather/weights frozen pixel interpolation; forward parity atol1e-6; original models unmodified'
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
                checks.append(dict(views=len(cams),image_hw=list(image.shape[-2:]),zero_max_difference=max(differences.values()),
                                   full_head_backward_not_run='legacy CUDA grid_sampler backward nondeterministic; cached deterministic backward checked during training'))
                del off,on,image
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
            def sample(record,variant=None):
                deadline();batch=oriented_batch(record,num_views=len(record['views']))
                if variant is not None:batch,_=augment_batch(batch,**{k:v for k,v in variant.items() if k!='name'})
                image=batch['images'].cuda()
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
            variants_by_key={key:[key] for key in train_keys}
            for row in aug_plan['records']:
                if not row['accepted'] or row['variant']['name']=='identity':continue
                key=row['key'];name=key+':'+row['variant']['name']
                cache[name]=sample(spec['records'][key],row['variant']);variants_by_key[key].append(name)
                status('caching_augmentation',records=len(cache),total=sum(r['accepted'] for r in aug_plan['records']))
            eval_cache={}
            for e in evaluation:eval_cache[e['key']]=sample(spec['records'][e['key']])
            for h in hooks:h.remove()
            hooks=[]
            torch.manual_seed(17);initial=copy.deepcopy(branch.state_dict());histories={};results={}
            rng=random.Random(17);sequence=[]
            while len(sequence)<STEPS:
                chunk=list(train_keys);rng.shuffle(chunk);sequence.extend(chunk)
            sequence=sequence[:STEPS];write_json(OUT/'schedule.json',sequence)
            assert sequence==json.loads((ROOT/'output/spatial_anchor_diverse32_v1/schedule.json').read_text())
            aug_rng=random.Random(1701)
            augmented_sequence=[aug_rng.choice(variants_by_key[key]) for key in sequence]
            write_json(OUT/'augmented_schedule.json',augmented_sequence)
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
                with torch.no_grad():return float(np.mean([float(loss_fn(*reversed(output(s,use_prior)),s)) for s in [cache[key] for key in train_keys]]))
            for arm in ('heatmap_off','heatmap_on'):
                use_prior=True;branch.load_state_dict(initial);branch.train()
                arm_sequence=augmented_sequence
                opt=torch.optim.AdamW(branch.parameters(),lr=.001,weight_decay=.01)
                rows=[];initial_loss=fixed_loss(use_prior)
                for step,key in enumerate(arm_sequence,1):
                    deadline();s=cache[key];opt.zero_grad(set_to_none=True)
                    coarse,refined=output(s,use_prior);coordinate_loss=loss_fn(refined,coarse,s)
                    auxiliary,visible_count=matched_visible_loss(branch,s) if arm=='heatmap_on' else (coordinate_loss.detach()*0,0)
                    loss=coordinate_loss+.005*auxiliary
                    assert torch.isfinite(loss);loss.backward()
                    grad=torch.nn.utils.clip_grad_norm_(branch.parameters(),1.);assert torch.isfinite(grad)
                    opt.step();rows.append(dict(step=step,loss=float(loss.detach()),coordinate_loss=float(coordinate_loss.detach()),auxiliary_kl=float(auxiliary.detach()),visible_points=visible_count,grad_norm=float(grad)))
                    if step%25==0:
                        status('training',arm=arm,step=step,total=STEPS);write_json(OUT/f'{arm}_history.json',rows);print('TRAIN',arm,step,rows[-1]['loss'],flush=True)
                    if step in (200,400,800):
                        branch.eval();dev=evaluate(use_prior)
                        write_json(OUT/f'{arm}_eval_{step}.json',dev)
                        torch.save(dict(branch={k:v.cpu() for k,v in branch.state_dict().items()},optimizer=opt.state_dict(),step=step),OUT/f'{arm}_step{step}.pt')
                        branch.train()
                branch.eval();histories[arm]=rows
                results[arm]=dict(initial_fixed_loss=initial_loss,final_fixed_loss=fixed_loss(use_prior),records=evaluate(use_prior))
                if arm=='heatmap_off':
                    reference=torch.load(ROOT/'output/spatial_anchor_augmentation_deterministic_v1/augmentation_on_step800.pt',map_location='cpu',weights_only=True)
                    differences={k:float((v.cpu()-reference['branch'][k]).abs().max()) for k,v in branch.state_dict().items()}
                    write_json(OUT/'control_replay.json',dict(max_abs_difference=differences,exact=max(differences.values())==0))
                    assert max(differences.values())==0,'Deterministic augmentation ON control replay failed'
                    for idx,state in opt.state_dict()['state'].items():
                        for k,value in state.items():
                            torch.testing.assert_close(value.cpu(),reference['optimizer']['state'][idx][k].cpu(),atol=0,rtol=0)
            base_after={k:v for k,v in model.state_dict().items() if not k.startswith(prefix)}
            assert tensor_mapping_sha256(base_after)==base_hash
            assert all(p.grad is None for n,p in model.named_parameters() if not n.startswith(prefix))
            assert fingerprint(PLAN)==protocol['sampling_plan_sha256']
            assert fingerprint(BASE)==protocol['base_sha256']
            assert all(fingerprint(ROOT/p)==h for p,h in protocol['code_sha256'].items())
            baseline=[dict(group=e['group'],key=e['key'],metrics=eval_cache[e['key']]['metrics']) for e in evaluation]
            write_json(OUT/'results.json',dict(baseline=baseline,arms=results))
            write_json(OUT/'verification.json',dict(passed=True,base_weights_unchanged=True,base_gradients_none=True,
                steps_per_arm=STEPS,new_parameters=sum(p.numel() for p in branch.parameters()),
                training_records=len(train_keys),cached_variants=len(cache),evaluation_records=len(evaluation),production_promoted=False))
            status('completed');print('COMPLETED',flush=True)
        except BaseException as e:status('failed',error=repr(e));raise
        finally:
            for h in hooks:h.remove()


if __name__=='__main__':main()






