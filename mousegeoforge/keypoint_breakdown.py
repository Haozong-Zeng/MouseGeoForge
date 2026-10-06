"""Per-joint diagnostics with unchanged box/objectness-only assignments."""
import numpy as np
import torch
from .pilot_metrics import box_iou


def keypoint_rows(pred,batch,target,matches):
    h,w=batch['images'].shape[-2:];scale=torch.tensor([w,h],dtype=torch.float32)
    rows=[]
    for v,(qs,ms) in enumerate(matches):
        for q,m in zip(qs,ms):
            p=pred['instance_boxes_cxcywh'][0,v,q:q+1]
            t=target['boxes_cxcywh'][0,v,m:m+1]
            found=bool(pred['instance_objectness_logits'][0,v,q].sigmoid()>=.5 and box_iou(p,t)[0,0]>=.5)
            diag=(t[0,2:]*scale).norm().clamp_min(1)
            error=((pred['instance_keypoint_xy'][0,v,q]-target['keypoint_xy'][0,v,m])*scale).norm(dim=-1)/diag
            visible_valid=target.get('keypoint_visibility_valid',target['keypoint_valid'])[0,v,m]
            for k in torch.where(target['keypoint_valid'][0,v,m])[0].tolist():
                rows.append(dict(view=v,query=q,instance=m,keypoint=k,nme=float(error[k]),found=found,
                    visible=bool(target['keypoint_visible'][0,v,m,k]) if visible_valid[k] else None,
                    pck=bool(error[k]<=.1),detection_pck=bool(found and error[k]<=.1)))
    return rows


def summarize_joint_rows(rows):
    if not rows:return dict(n=0,nme=None,p90=None,pck=None,detection_pck=None)
    return dict(n=len(rows),nme=float(np.mean([r['nme'] for r in rows])),p90=float(np.percentile([r['nme'] for r in rows],90)),
        pck=float(np.mean([r['pck'] for r in rows])),detection_pck=float(np.mean([r['detection_pck'] for r in rows])))
