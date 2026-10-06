"""Inference-only 2D diagnostic; no model factory or checkpoint changes.

Only six explicitly evaluated 2D fields are replaced. Auxiliary pose tensors
and curve predictions are NOT made consistent and must not train a joint loss.
Geometry remains exactly the original multi-view output. Query IDs in separate
views are not cross-view instance identities.
"""
import torch

POSE_KEYS=('instance_keypoint_xy','instance_keypoint_original_xy',
           'instance_boxes_cxcywh','instance_objectness_logits',
           'instance_boxes_original_cxcywh','instance_objectness_original_logits')

@torch.no_grad()
def independent_view_pose_probe(images, multiview, forward_single):
    if images.ndim!=5 or images.shape[1]<1:
        raise ValueError('Expected B,V,C,H,W with V>=1')
    batch,views=images.shape[:2]
    for key in POSE_KEYS:
        if key not in multiview or multiview[key].shape[:2]!=(batch,views):
            raise ValueError('Missing or misaligned 2D field: '+key)
    if views==1:return dict(multiview)
    parts={key:[] for key in POSE_KEYS}
    for view in range(views):
        single=forward_single(images[:,view:view+1])
        for key in POSE_KEYS:
            if key not in single or single[key].shape!=(batch,1,*multiview[key].shape[2:]):
                raise ValueError('Single-view field shape mismatch: '+key)
            parts[key].append(single[key])
    result=dict(multiview)
    for key in POSE_KEYS:result[key]=torch.cat(parts[key],dim=1)
    return result
