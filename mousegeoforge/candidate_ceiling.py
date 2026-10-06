"""GT-assisted geometric capacity bound only; never use for inference."""
import torch
from mousegeoforge.pilot_metrics import box_iou
from training.instance_pose_losses import _linear_sum_assignment


def geometric_ceiling(boxes,targets):
    if len(targets)==0:return dict(gt=0,covered=0,matches=[])
    if len(boxes)<len(targets):raise ValueError('This audit requires at least as many candidates as targets')
    ious=box_iou(targets,boxes)
    # Maximize the integer match count first; total tie-break is always <1.
    cost=-(ious>=.5).float()-ious/(len(targets)+1)*1e-3
    pred,rows=_linear_sum_assignment(cost)
    return dict(gt=len(targets),covered=int((ious[rows,pred]>=.5).sum()),
                matches=[dict(target=int(t),query=int(q),iou=float(ious[t,q]),covered=bool(ious[t,q]>=.5)) for t,q in zip(rows,pred)])
