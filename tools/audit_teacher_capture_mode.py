"""Diagnose strict teacher/self equality without updating model or old preflight."""
import json
from run_depth_pose_ablation import ROOT,torch,VGGTOmega,DepthPoseModel,restore,tensor_state,state_hash,fingerprint,write_json,instance_targets_from_batch,move_batch
from train_cylinder_view_balance import oriented_batch
from mousegeoforge.replay_retention import build_teacher_targets,matched_replay_retention,upstream

OUT=ROOT/'output/cylinder_teacher_capture_mode_v1'


def main():
    if OUT.exists():raise RuntimeError('Do not overwrite mode audit')
    spec=json.loads((ROOT/'output/cylinder_retention_preflight_v1/protocol.json').read_text());r=spec['records'][0]
    torch.set_num_threads(2);torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    model=VGGTOmega(enable_camera=False,enable_depth=True,enable_multi_instance_pose=True,num_keypoints=23)
    encoder=ROOT/'vggt-omega/vggt_omega_1b_512.pt';weights=torch.load(encoder,weights_only=True,mmap=True)
    model.load_state_dict({k:v for k,v in weights.items() if k.startswith(('aggregator.','dense_head.'))},strict=False);del weights
    model=DepthPoseModel.from_spatial_model(model).cuda().eval()
    checkpoint=ROOT/spec['checkpoints']['teacher']['path'];assert fingerprint(checkpoint)==spec['checkpoints']['teacher']['sha256']
    state=torch.load(checkpoint,weights_only=True,map_location='cpu')['model'];restore(model,state)
    batch=move_batch(oriented_batch(r,num_views=2),'cuda');targets=instance_targets_from_batch(batch)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):cached,start=model.aggregator(batch['images'])
    with torch.no_grad():no_grad=model.forward_cached(batch['images'],cached,start,coupling='off')
    with torch.enable_grad():
        grad=model.forward_cached(batch['images'],cached,start,coupling='off')
        grad_detached={k:v.detach().clone() for k,v in model.forward_cached(batch['images'],cached,start,coupling='off').items()}
    matches=upstream.hungarian_match_instances(grad,targets)
    rows=[]
    for mode,p in (('no_grad_capture',no_grad),('grad_then_detach_capture',grad_detached)):
        teacher=build_teacher_targets(p,targets,upstream.hungarian_match_instances(p,targets))
        loss,counts=matched_replay_retention(grad,targets,matches,teacher)
        gs=torch.autograd.grad(loss,list(model.shared_adapter.parameters())+list(model.instance_pose_head.parameters()),retain_graph=True,allow_unused=True)
        norm=float(sum(g.square().sum() for g in gs if g is not None).sqrt())
        rows.append(dict(mode=mode,loss=float(loss.detach()),gradient_norm=norm,**counts,
            box_max_difference=float((grad['instance_boxes_cxcywh']-p['instance_boxes_cxcywh']).abs().max().detach()),
            xy_max_difference=float((grad['instance_keypoint_xy']-p['instance_keypoint_xy']).abs().max().detach())))
    assert state_hash(tensor_state(model))==state_hash(state)
    report=dict(rows=rows,record=r,weights_unchanged=True,optimizer_steps=0,
        checkpoint_sha256=fingerprint(checkpoint),code_sha256=fingerprint(ROOT/'tools/audit_teacher_capture_mode.py'))
    write_json(OUT/'report.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
