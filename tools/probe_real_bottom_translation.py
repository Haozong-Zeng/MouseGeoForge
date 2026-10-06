"""Label-free translation equivariance probe, matched by predicted masks only."""
import json
import cv2,numpy as np,torch
from run_depth_pose_ablation import ROOT,VGGTOmega,DepthPoseModel,fingerprint,write_json
from mousegeoforge.adaptive_temperature import install_adaptive_temperature,restore_adaptive

OUT=ROOT/'output/real_cylinder_bottom_frozen_v1'


def main():
    torch.set_num_threads(2);cv2.setNumThreads(1)
    protocol=json.loads((OUT/'protocol.json').read_text())
    model0=VGGTOmega(enable_multi_instance_pose=True,num_keypoints=23)
    loaded=model0.load_state_dict(torch.load(ROOT/'vggt-omega/vggt_omega_1b_512.pt',map_location='cpu',weights_only=True,mmap=True),strict=False)
    assert not loaded.unexpected_keys and all(k.startswith('instance_pose_head.') for k in loaded.missing_keys)
    model=install_adaptive_temperature(DepthPoseModel.from_spatial_model(model0)).cuda().eval().requires_grad_(False)
    states={n:torch.load(ROOT/p,map_location='cpu',weights_only=True)['model'] for n,p in protocol['head_paths'].items()}
    rows=[]
    for frame in [780,796,811]:
        image=cv2.cvtColor(cv2.imread(str(OUT/f'review/AZL3_{frame:06d}_A_input.png')),cv2.COLOR_BGR2RGB)
        h,w=image.shape[:2];shifted=cv2.warpAffine(image,np.float32([[1,0,64],[0,1,0]]),(w,h),borderValue=(0,0,0))
        original=torch.load(OUT/f'predictions/AZL3_{frame:06d}_bottom.pt',map_location='cpu',weights_only=True)['outputs']
        x=torch.from_numpy(shifted).permute(2,0,1).unsqueeze(0).unsqueeze(0).cuda().float()/255
        with torch.inference_mode():
            with torch.autocast('cuda',dtype=torch.bfloat16):cached,start=model.aggregator(x)
            for name,state in states.items():
                restore_adaptive(model,state);new=model.forward_cached(x,cached,start)
                old=original[name];score=old['instance_objectness_logits'][0,0].sigmoid().numpy().reshape(-1);q=int(score.argmax())
                oldmask=cv2.resize(old['instance_mask_logits'][0,0,q].numpy(),(w,h))>0
                expected=cv2.warpAffine(np.uint8(oldmask),np.float32([[1,0,64],[0,1,0]]),(w,h))>0
                masks=[cv2.resize(m,(w,h))>0 for m in new['instance_mask_logits'][0,0].cpu().numpy()]
                ious=[np.count_nonzero(m&expected)/max(1,np.count_nonzero(m|expected)) for m in masks];qn=int(np.argmax(ious))
                box0=old['instance_boxes_cxcywh'][0,0,q,:2].numpy()*[w,h]
                box1=new['instance_boxes_cxcywh'][0,0,qn,:2].cpu().numpy()*[w,h]
                p0=old['instance_keypoint_xy'][0,0,q].numpy()*[w,h]
                p1=new['instance_keypoint_xy'][0,0,qn].cpu().numpy()*[w,h]
                body=list(range(7))+list(range(14,23))
                row=dict(frame=frame,model=name,known_shift_px=[64,0],predicted_mask_equivariance_iou=float(ious[qn]),
                    original_query=q,matched_query=qn,new_query_score=float(new['instance_objectness_logits'][0,0,qn].sigmoid().item()),
                    box_center_displacement_px=(box1-box0).tolist(),body_mean_displacement_px=np.mean(p1[body]-p0[body],axis=0).tolist(),
                    body_translation_residual_px=float(np.linalg.norm(p1[body]-p0[body]-[64,0],axis=-1).mean()),
                    matching='maximum IoU with translated ORIGINAL prediction mask, not GT; may ignore ranking')
                rows.append(row)
                print(row,flush=True)
        del cached,new,x
    assert all(fingerprint(ROOT/p)==protocol['hashes'][n] for n,p in protocol['head_paths'].items())
    write_json(OUT/'translation_probe.json',dict(records=rows,weights_updated=False,manual_labels_used=False,
               caveat='Image-boundary padding/context changes too; not a pure architectural causal proof.'))


if __name__=='__main__':main()
