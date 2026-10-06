"""Prediction-only two-view association baseline; no GT boxes/IDs/coordinates.

This is an epipolar baseline, not a complete social-mouse tracker. Association
failure is explicitly retained and measured separately from pose error.
"""
import torch
from training.instance_pose_losses import _linear_sum_assignment
from mousegeoforge.pilot_metrics import box_iou


def fundamental_from_projections(p0, p1):
    _, _, vh = torch.linalg.svd(p0, full_matrices=True)
    center = vh[-1]
    e = p1@center
    zero = e.new_zeros(())
    cross = torch.stack((zero, -e[2], e[1], e[2], zero, -e[0], -e[1], e[0], zero)).reshape(3, 3)
    f = cross@p1@torch.linalg.pinv(p0)
    return f/f.norm().clamp_min(1e-12)


@torch.no_grad()
def associate_two_views(output, projections, image_hw, *, score_threshold=.5, max_epipolar_px=32.):
    if output['instance_keypoint_xy'].shape[:2] != (1, 2):
        raise ValueError('This baseline supports batch=1 and exactly two views')
    h, w = image_hw
    xy = output['instance_keypoint_xy'][0]*output['instance_keypoint_xy'].new_tensor([w, h])
    scores = output['instance_objectness_logits'][0].sigmoid()
    boxes = output['instance_boxes_cxcywh'][0]
    selected = []
    for v in range(2):
        keep = []
        for q in scores[v].argsort(descending=True).tolist():
            if scores[v, q] < score_threshold:
                continue
            if keep and (box_iou(boxes[v, q:q+1], boxes[v, keep]) > .8).any():
                continue
            keep.append(q)
        selected.append(keep)
    if not all(selected):
        return []
    f = fundamental_from_projections(*projections)
    cost = xy.new_full((len(selected[0]), len(selected[1])), 1e6)
    for a, q0 in enumerate(selected[0]):
        for b, q1 in enumerate(selected[1]):
            points0 = torch.cat((xy[0, q0], torch.ones_like(xy[0, q0, :, :1])), -1)
            points1 = torch.cat((xy[1, q1], torch.ones_like(xy[1, q1, :, :1])), -1)
            lines1, lines0 = points0@f.T, points1@f
            residual = (points1*lines1).sum(-1).abs()
            distance = .5*residual*(lines1[:, :2].norm(dim=-1).clamp_min(1e-6).reciprocal()
                                    + lines0[:, :2].norm(dim=-1).clamp_min(1e-6).reciprocal())
            finite = torch.isfinite(distance)
            if len(finite) == 23:
                finite[7:14] = False  # Flexible tail is not a stable identity anchor.
            if finite.any():
                cost[a, b] = distance[finite].median()
    # Add private dummy columns so rejected matches do not force other matches.
    padded = torch.cat((cost, cost.new_full((len(cost), len(cost)), max_epipolar_px)), -1)
    columns, rows = _linear_sum_assignment(padded)
    return [(selected[0][a], selected[1][b], float(cost[a, b]))
            for b, a in zip(columns.tolist(), rows.tolist())
            if b < len(selected[1]) and cost[a, b] < max_epipolar_px]
