"""Small, explicit evaluation metrics for fully annotated spatial pilots.

Assignment uses boxes and objectness ONLY, never target keypoints. NME is a
box-matched diagnostic (all targets assigned); detection-aware PCK also counts
missed detections as failures. This is not COCO AP or a 3D benchmark.
"""
import math
import numpy as np
import torch
from torch.nn import functional as F
from training.instance_pose_losses import _linear_sum_assignment, box_cxcywh_to_xyxy


def box_iou(a, b):
    a, b = box_cxcywh_to_xyxy(a), box_cxcywh_to_xyxy(b)
    intersection = (torch.minimum(a[:, None, 2:], b[None, :, 2:]) -
                    torch.maximum(a[:, None, :2], b[None, :, :2])).clamp_min(0).prod(-1)
    area_a = (a[:, 2:]-a[:, :2]).clamp_min(0).prod(-1)
    area_b = (b[:, 2:]-b[:, :2]).clamp_min(0).prod(-1)
    return intersection/(area_a[:, None]+area_b[None, :]-intersection).clamp_min(1e-8)


@torch.no_grad()
def evaluate_scene(output, batch, targets):
    assert batch['images'].shape[0] == 1
    h, w = batch['images'].shape[-2:]
    pixel_scale = output['depth'].new_tensor([w, h])
    values = {k: [] for k in ('box_matched_nme', 'visible_nme', 'occluded_nme',
              'body_keypoint_nme', 'tail_nme', 'box_matched_pck_0.1',
              'detection_aware_pck_0.1', 'matched_mask_iou')}
    tp = fp = fn = 0
    assignments = []
    for view in range(batch['images'].shape[1]):
        valid = targets['instance_valid'][0, view]
        indices = torch.where(valid)[0]
        boxes = output['instance_boxes_cxcywh'][0, view]
        scores = output['instance_objectness_logits'][0, view].sigmoid()
        target_boxes = targets['boxes_cxcywh'][0, view, indices]
        cost = 5*torch.cdist(target_boxes, boxes, p=1)-scores[None]
        pred, rows = _linear_sum_assignment(cost)
        match = indices[rows]
        assignments.append((pred.tolist(), match.tolist()))
        iou = box_iou(target_boxes[rows], boxes[pred]).diagonal()
        found = (scores[pred] >= .5) & (iou >= .5)
        tp += int(found.sum())
        fp += int((scores >= .5).sum())-int(found.sum())
        fn += len(match)-int(found.sum())
        for j, (q, m) in enumerate(zip(pred.tolist(), match.tolist())):
            xy = output['instance_keypoint_xy'][0, view, q]
            gt = targets['keypoint_xy'][0, view, m]
            point_valid = targets['keypoint_valid'][0, view, m]
            visible = targets['keypoint_visible'][0, view, m]
            visible_valid = targets.get('keypoint_visibility_valid', targets['keypoint_valid'])[0, view, m]
            diag = (targets['boxes_cxcywh'][0, view, m, 2:]*pixel_scale).norm().clamp_min(1.)
            errors = ((xy-gt)*pixel_scale).norm(dim=-1)/diag
            tail = torch.zeros_like(point_valid)
            tail[7:14] = True  # audited Core23 channel order
            for name, mask in (('box_matched_nme',point_valid),
                               ('visible_nme',point_valid & visible & visible_valid),
                               ('occluded_nme',point_valid & ~visible & visible_valid),
                               ('body_keypoint_nme',point_valid & ~tail),
                               ('tail_nme',point_valid & tail)):
                values[name].extend(errors[mask].cpu().tolist())
            values['box_matched_pck_0.1'].extend((errors[point_valid] <= .1).float().cpu().tolist())
            values['detection_aware_pck_0.1'].extend(((errors[point_valid] <= .1) & found[j]).float().cpu().tolist())
            if targets['mask_valid'][0, view, m]:
                p = F.interpolate(output['instance_mask_logits'][0, view, q][None,None],
                                  size=(h,w), mode='bilinear', align_corners=False)[0,0] > 0
                t = targets['masks'][0, view, m].bool()
                values['matched_mask_iou'].append(float((p&t).sum()/(p|t).sum().clamp_min(1)))
    result = {k: float(np.mean(v)) if v else None for k,v in values.items()}
    result.update(detection_tp=tp, detection_fp=fp, detection_fn=fn,
                  detection_precision=tp/max(tp+fp,1), detection_recall=tp/max(tp+fn,1))
    depth = output['depth'][...,0]
    gt = batch['depths']
    valid = batch['depth_masks'] & batch['depth_available'][...,None,None] & torch.isfinite(gt) & (gt>1e-6)
    for region, mask in (('all',valid),('body',valid & batch['body_masks']),
                         ('background',valid & ~batch['body_masks'])):
        result[f'{region}_depth_absrel'] = float(((depth-gt).abs()/gt.clamp_min(1e-6))[mask].mean()) if mask.any() else None
    if not all(v is None or math.isfinite(v) for v in result.values()):
        raise ValueError('Nonfinite evaluation metric')
    return result, assignments


def summarize_records(records):
    groups = {}
    for record in records:
        groups.setdefault(record['source_group'], []).append(record['metrics'])
    per_group = {}
    for group, metrics in groups.items():
        per_group[group] = {k: float(np.mean([m[k] for m in metrics if m[k] is not None]))
                            if any(m[k] is not None for m in metrics) else None for k in metrics[0]}
    macro = {k: float(np.mean([m[k] for m in per_group.values() if m[k] is not None]))
             if any(m[k] is not None for m in per_group.values()) else None
             for k in next(iter(per_group.values()))}
    return {'source_macro': macro, 'per_source': per_group, 'scenes': len(records)}
