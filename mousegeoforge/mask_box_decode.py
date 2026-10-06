"""Opt-in inference-only boxes from predicted masks; no GT or model mutation.

Threshold and resize match the existing mask evaluator. This deliberately keeps
all foreground components and the tail: it is an audited baseline, not a tuned
denoiser. Empty masks retain the regressed box. No claim of amodal body extent.
"""
import torch
from torch.nn import functional as F


@torch.no_grad()
def mask_boxes_from_logits(logits, regressed_boxes, image_hw, *, enabled=False):
    """Return normalized cxcywh and a mask-used flag for each [..., Q] query.

    Disabled is a true identity path, even without masks. Enabled accepts any
    leading batch/view axes, but requires matching mask and box query axes.
    Edges use exclusive pixel maxima, as the dataset's GT mask-box loader does.
    """
    if not enabled:
        return regressed_boxes, torch.zeros_like(regressed_boxes[..., 0], dtype=torch.bool)
    if logits is None or logits.ndim < 3 or regressed_boxes.shape != (*logits.shape[:-2], 4):
        raise ValueError('Expected matching [..., Q, H, W] masks and [..., Q, 4] boxes')
    if logits.device != regressed_boxes.device:
        raise ValueError('Masks and boxes must share a device')
    if not logits.is_floating_point() or not regressed_boxes.is_floating_point():
        raise ValueError('Floating logits and boxes required')
    if not torch.isfinite(logits).all() or not torch.isfinite(regressed_boxes).all():
        raise ValueError('Nonfinite predicted masks or boxes')
    if len(image_hw) != 2 or any(not isinstance(n, int) or n <= 0 for n in image_hw):
        raise ValueError('image_hw must contain two positive integers')
    h, w = image_hw
    flat = logits.reshape(-1, 1, *logits.shape[-2:])
    if flat.shape[0] == 0:
        return regressed_boxes.clone(), torch.zeros_like(regressed_boxes[..., 0], dtype=torch.bool)
    foreground = F.interpolate(flat, size=(h, w), mode='bilinear', align_corners=False)[:, 0] > 0
    columns, rows = foreground.any(dim=-2), foreground.any(dim=-1)
    used = columns.any(dim=-1)
    xs = torch.arange(w, device=logits.device).expand_as(columns)
    ys = torch.arange(h, device=logits.device).expand_as(rows)
    x0 = torch.where(columns, xs, w).amin(-1).float() / w
    y0 = torch.where(rows, ys, h).amin(-1).float() / h
    x1 = (torch.where(columns, xs, -1).amax(-1).float() + 1) / w
    y1 = (torch.where(rows, ys, -1).amax(-1).float() + 1) / h
    boxes = torch.stack(((x0+x1)/2, (y0+y1)/2, x1-x0, y1-y0), dim=-1)
    boxes = boxes.reshape_as(regressed_boxes).to(regressed_boxes.dtype)
    used = used.reshape(regressed_boxes.shape[:-1])
    return torch.where(used[..., None], boxes, regressed_boxes), used
