"""Separate numerical backward repeatability from added conditioning path."""
import fcntl
import json
import torch.nn.functional as F
from types import MethodType,SimpleNamespace
from run_depth_pose_ablation import (ROOT,torch,VGGTOmega,DepthPoseModel,fingerprint,write_json,instance_targets_from_batch,move_batch,depth_loss)
from train_cylinder_view_balance import oriented_batch
from mousegeoforge.adaptive_temperature import install_adaptive_temperature,restore_adaptive,PREFIX
from training.instance_pose_losses import instance_pose_loss,InstancePoseLossWeights

OUT=ROOT/'output/cylinder_temperature_gradient_repeatability_v1'


def main():
    if OUT.exists():raise RuntimeError('Do not overwrite repeatability audit')
    pre=json.loads((ROOT/'output/cylinder_adaptive_temperature_preflight_v1/protocol.json').read_text())
    parent=json.loads((ROOT/'output/cylinder_source_mix_v1/protocol.json').read_text())
    torch.set_num_threads(2);torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    spatial=VGGTOmega(enable_camera=False,enable_depth=True,enable_multi_instance_pose=True,num_keypoints=23)
    weights=torch.load(ROOT/'vggt-omega/vggt_omega_1b_512.pt',weights_only=True,mmap=True)
    spatial.load_state_dict({k:v for k,v in weights.items() if k.startswith(('aggregator.','dense_head.'))},strict=False);del weights
    model=install_adaptive_temperature(DepthPoseModel.from_spatial_model(spatial)).cuda().train();del spatial
    state=torch.load(ROOT/parent['config']['checkpoint'],weights_only=True,map_location='cpu')['model'];restore_adaptive(model,state)
    old=[(k,p) for k,p in model.named_parameters() if p.requires_grad and not k.startswith(PREFIX)]
    lw=InstancePoseLossWeights(keypoint_3d=0.,reprojection=0.,local_heatmap=0.,keypoint_uncertainty=0.)
    args=SimpleNamespace(semantic_label_weights='',body_depth_balance=.5,body_depth_fraction=.5,depth_gradient_weight=0.)
    rows=[]
    for idx in (0,1):
        r=pre['entries'][idx]['record'];batch=move_batch(oriented_batch(r,num_views=2),'cuda');target=instance_targets_from_batch(batch)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):cached,start=model.aggregator(batch['images'])
        results={}
        original_forward=model.instance_pose_head.adaptive_temperature.forward
        for mode in ('off1','off2','on1','on2','on_detached'):
            model.instance_pose_head.adaptive_temperature_enabled=mode.startswith('on')
            if mode=='on_detached':
                def detached(self,tokens):return original_forward(tokens.detach())
                model.instance_pose_head.adaptive_temperature.forward=MethodType(detached,model.instance_pose_head.adaptive_temperature)
            pred=model.forward_cached(batch['images'],cached,start,coupling='off')
            pl,_=instance_pose_loss(pred,target,lw,image_size=r['image_hw'])
            dl=depth_loss(pred['depth'],batch['depths'],batch['depth_masks'],batch['body_masks'],batch['semantic_masks'],args)['loss_depth']
            gradients=torch.autograd.grad(pl+dl,[p for k,p in old],allow_unused=True)
            results[mode]=[torch.zeros_like(p).cpu() if g is None else g.detach().cpu() for (k,p),g in zip(old,gradients)]
            del pred,pl,dl,gradients
        model.instance_pose_head.adaptive_temperature.forward=original_forward
        for a,b in (('off1','off2'),('off1','on1'),('on1','on2'),('off1','on_detached')):
            differences=[]
            for (k,p),x,y in zip(old,results[a],results[b]):
                differences.append(dict(parameter=k,max_difference=float((x-y).abs().max()),reference_abs_max=float(x.abs().max())))
            rows.append(dict(record=idx,comparison=a+' vs '+b,max_difference=max(r['max_difference'] for r in differences),
                top_differences=sorted(differences,key=lambda r:r['max_difference'],reverse=True)[:6]))
        del results,cached,batch,target
    report=dict(rows=rows,optimizer_steps=0,code_sha256=fingerprint(ROOT/'tools/audit_temperature_gradient_repeatability.py'))
    write_json(OUT/'report.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':
    with (ROOT/'output/cylinder_training.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);main()
