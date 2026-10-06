"""Training-only correction of strong box/mask assignment conflict; no pose GT."""
import copy
import torch
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment


@torch.no_grad()
def repair_mask_conflicts(pred_logits,gt_masks,instance_valid,mask_valid,original_matches,*,minimum_iou=.5,minimum_gain=.2):
    if pred_logits.ndim!=5 or pred_logits.shape[0]!=1 or gt_masks.ndim!=5 or gt_masks.shape[:2]!=pred_logits.shape[:2]:raise ValueError('Expected B1,V,Q/H or M/H masks')
    if instance_valid.shape!=gt_masks.shape[:3] or mask_valid.shape!=instance_valid.shape:raise ValueError('Validity shape mismatch')
    h,w=gt_masks.shape[-2:];out=copy.deepcopy(original_matches);records=[]
    for v,(queries,instances) in enumerate(original_matches):
        ids=torch.where(instance_valid[0,v])[0].tolist()
        if set(ids)!=set(instances) or len(queries)!=len(set(queries)):raise ValueError('Original mapping must cover every valid target one-to-one')
        if not mask_valid[0,v,ids].all():raise ValueError('Complete GT masks required for this synthetic diagnostic')
        if not ids:records.append(dict(overridden=False,strong_instances=0));continue
        masks=(F.interpolate(pred_logits[0,v,:,None].float(),size=(h,w),mode='bilinear',align_corners=False)[:,0]>0).cpu()
        gt=gt_masks[0,v,ids].bool().cpu();left=gt.flatten(1).float();right=masks.flatten(1).float()
        inter=left@right.T;iou=inter/(left.sum(-1)[:,None]+right.sum(-1)[None]-inter).clamp_min(1)
        rr,cc=linear_sum_assignment(-iou.numpy());best={ids[int(r)]:int(c) for r,c in zip(rr,cc)};old=dict(zip(instances,queries))
        strong=[m for i,m in enumerate(ids) if float(iou[i,best[m]])>=minimum_iou and float(iou[i,best[m]]-iou[i,old[m]])>=minimum_gain]
        gain=sum(float(iou[i,best[m]]-iou[i,old[m]]) for i,m in enumerate(ids))
        override=bool(strong) and gain>0
        if override:out[v]=([best[m] for m in instances],list(instances))
        records.append(dict(overridden=override,strong_instances=len(strong),sum_mask_iou_gain=gain,
                            original_queries=list(queries),repaired_queries=list(out[v][0]),instances=list(instances)))
    return out,records
