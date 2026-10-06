"""Initial output/old-gradient parity and new-branch learning feasibility."""
import fcntl
import json
import time
from types import SimpleNamespace
import numpy as np
from run_depth_pose_ablation import (ROOT,torch,VGGTOmega,DepthPoseModel,tensor_state,state_hash,fingerprint,
    write_json,instance_targets_from_batch,depth_loss,move_batch)
from train_cylinder_view_balance import oriented_batch,scene_digest
from mousegeoforge.task_gradient_audit import select_task_records,upstream
from mousegeoforge.adaptive_temperature import install_adaptive_temperature,restore_adaptive,original_state_projection,PREFIX

OUT=ROOT/'output/cylinder_adaptive_temperature_preflight_v1'
PARENT=ROOT/'output/cylinder_source_mix_v1'


def run():
    began=time.monotonic();spec=json.loads((PARENT/'protocol.json').read_text());entries=select_task_records(spec)
    assert json.loads((PARENT/'delivery_verification.json').read_text())['passed']
    assert json.loads((PARENT/'status.json').read_text())['state']=='completed'
    sources={e['record']['scene']:e['record']['source_revision'] for e in entries}
    code=dict(spec['code_sha256'])
    for p in ('mousegeoforge/adaptive_temperature.py','mousegeoforge/task_gradient_audit.py','tools/preflight_cylinder_adaptive_temperature.py'):
        code[p]=fingerprint(ROOT/p)
    for p,h in code.items():assert fingerprint(ROOT/p)==h
    assert all(scene_digest(ROOT/p)==h for p,h in sources.items())
    protocol=dict(entries=entries,source_sha256=sources,code_sha256=code,parent_protocol_sha256=fingerprint(PARENT/'protocol.json'),
        checkpoint_sha256=spec['checkpoint_sha256'],scope='Original state, 16 existing train inputs. Train-mode OFF/ON comparisons; no optimizer steps.',
        new_layer='257 zero-initialized parameters: linear normalized token to log(scale), clamped [.5,8], exp. No RNG consumption.',
        gate='All forward fields bitwise equal at scale1; original parameter max gradient difference <=1e-6; finite nonzero new gradient, median new/head norm <=.25; source/state unchanged.',
        training_proposal='One 800-step source_mix schedule; original parameters lr3e-6, new layer lr3e-4, AdamW otherwise unchanged. No new losses or data.')
    write_json(OUT/'protocol.json',protocol)
    torch.set_num_threads(2);torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    encoder=ROOT/'vggt-omega/vggt_omega_1b_512.pt';assert fingerprint(encoder)==spec['encoder_sha256']
    spatial=VGGTOmega(enable_camera=False,enable_depth=True,enable_multi_instance_pose=True,num_keypoints=23)
    weights=torch.load(encoder,weights_only=True,mmap=True)
    loaded=spatial.load_state_dict({k:v for k,v in weights.items() if k.startswith(('aggregator.','dense_head.'))},strict=False)
    assert not loaded.unexpected_keys and all(k.startswith('instance_pose_head.') for k in loaded.missing_keys)
    del weights
    model=install_adaptive_temperature(DepthPoseModel.from_spatial_model(spatial)).cuda().train();del spatial
    checkpoint=ROOT/spec['config']['checkpoint'];assert fingerprint(checkpoint)==spec['checkpoint_sha256']
    state=torch.load(checkpoint,weights_only=True,map_location='cpu')['model'];restore_adaptive(model,state)
    before=state_hash(tensor_state(model));assert state_hash(original_state_projection(tensor_state(model)))==state_hash(state)
    old=[(k,p) for k,p in model.named_parameters() if p.requires_grad and not k.startswith(PREFIX)]
    new=list(model.instance_pose_head.adaptive_temperature.parameters());assert sum(p.numel() for p in new)==257
    lw=upstream.InstancePoseLossWeights(keypoint_3d=0.,reprojection=0.,local_heatmap=0.,keypoint_uncertainty=0.)
    args=SimpleNamespace(semantic_label_weights='',body_depth_balance=.5,body_depth_fraction=.5,depth_gradient_weight=0.)
    rows=[];forward_max=0.;gradient_max=0.
    for idx,e in enumerate(entries):
        r=e['record'];batch=move_batch(oriented_batch(r,num_views=2),'cuda');targets=instance_targets_from_batch(batch)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):cached,start=model.aggregator(batch['images'])
        modes={}
        for enabled in (False,True):
            model.instance_pose_head.adaptive_temperature_enabled=enabled
            pred=model.forward_cached(batch['images'],cached,start,coupling='off')
            pose,_=upstream.instance_pose_loss(pred,targets,lw,image_size=r['image_hw'])
            depth=depth_loss(pred['depth'],batch['depths'],batch['depth_masks'],batch['body_masks'],batch['semantic_masks'],args)['loss_depth']
            loss=pose+depth
            gradients=torch.autograd.grad(loss,[p for _,p in old]+new,allow_unused=True)
            modes[enabled]=dict(pred={k:v.detach().cpu() for k,v in pred.items()},
                gradients=[torch.zeros_like(p).cpu() if g is None else g.detach().cpu() for p,g in zip([p for _,p in old]+new,gradients)],loss=float(loss.detach()))
            del pred,pose,depth,loss,gradients
        a,b=modes[False],modes[True]
        for k in a['pred']:
            diff=float((a['pred'][k]-b['pred'][k]).abs().max());forward_max=max(forward_max,diff);assert diff==0
        assert a['loss']==b['loss']
        assert (model.instance_pose_head.last_temperature_scale==1).all()
        for x,y in zip(a['gradients'][:len(old)],b['gradients'][:len(old)]):
            difference=float((x-y).abs().max());gradient_max=max(gradient_max,difference);assert difference<=1e-6
        ng=torch.cat([g.flatten() for g in b['gradients'][len(old):]])
        hg=torch.cat([g.flatten() for (k,p),g in zip(old,b['gradients']) if k.startswith('instance_pose_head.')])
        assert torch.isfinite(ng).all() and ng.norm()>0 and hg.norm()>0
        rows.append(dict(record=idx,kind=e['kind'],new_gradient_norm=float(ng.norm()),head_gradient_norm=float(hg.norm()),ratio=float(ng.norm()/hg.norm())))
        del modes,a,b,cached,batch,targets,ng,hg
        print(f'Parity {idx+1}/16',flush=True)
    assert state_hash(tensor_state(model))==before
    assert all(scene_digest(ROOT/p)==h for p,h in sources.items())
    for p,h in code.items():assert fingerprint(ROOT/p)==h
    assert fingerprint(checkpoint)==spec['checkpoint_sha256']
    median=float(np.median([r['ratio'] for r in rows]));supported=median<=.25
    report=dict(passed=True,training_supported=supported,forward_max_difference=forward_max,
        old_parameter_gradient_max_difference=gradient_max,new_gradient_ratio_median=median,new_parameters=257,
        source_code_weights_unchanged=True,optimizer_steps=0,encoder_forwards=16,head_forwards=32,elapsed_s=time.monotonic()-began,rows=rows)
    write_json(OUT/'report.json',report);write_json(OUT/'status.json',dict(state='completed'))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'output/cylinder_training.lock').open('a+') as glock,(OUT/'running.lock').open('a+') as lock:
        fcntl.flock(glock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (OUT/'protocol.json').exists():raise RuntimeError('Do not overwrite preflight')
        try:run()
        except BaseException as exc:
            write_json(OUT/'status.json',dict(state='failed',error=repr(exc),optimizer_steps=0));raise
