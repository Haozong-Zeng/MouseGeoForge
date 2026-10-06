"""CPU diagnostic regional-maximum candidates; not the production decoder.

Same output convention as fixed_top_peaks. No GT or instance count input.
8-connected equal-height plateaus use their first row-major pixel; missing
slots stay unavailable rather than being filled with arbitrary background.
"""
import numpy as np
import torch
from scipy.ndimage import label, maximum_filter, binary_propagation


@torch.no_grad()
def regional_top_peaks(output, image_hw, count=6, radius_px=8.):
    if count <= 0 or radius_px < 0:
        raise ValueError('Positive count and nonnegative radius required')
    ih, iw = image_hw
    if min(ih, iw) <= 0:
        raise ValueError('Positive image dimensions required')
    logits = output['logits'].detach().cpu()
    valid = output['spatial_valid'].detach().cpu().bool()
    grid = output['grid'].detach().cpu()
    if logits.ndim != 3 or logits.shape[1:] != valid.shape or grid.shape != (*valid.shape, 2):
        raise ValueError('Incompatible spatial shapes')
    classes, h, w = logits.shape
    if not torch.isfinite(logits[:, valid]).all():
        raise ValueError('Valid logits must be finite')
    probability = torch.zeros_like(logits)
    if valid.any():
        probability = logits.masked_fill(~valid, -torch.inf).flatten(1).softmax(-1).reshape_as(logits)
    xy = torch.zeros(classes, count, 2, dtype=grid.dtype)
    scores = torch.zeros(classes, count, dtype=logits.dtype)
    available = torch.zeros(classes, count, dtype=torch.bool)
    regional_mask = torch.zeros_like(logits, dtype=torch.bool)
    scale = np.asarray([iw, ih])
    grid_np = grid.numpy().reshape(-1, 2)
    for k in range(classes):
        if not valid.any() or logits[k, valid].max() == logits[k, valid].min():
            continue  # no localized evidence in a spatially flat map
        a = logits[k].masked_fill(~valid, -torch.inf).numpy()
        maxima = valid.numpy() & (a == maximum_filter(a, size=3, mode='constant', cval=-np.inf))
        components, n = label(maxima, structure=np.ones((3, 3)))
        representatives = []
        neighborhood_max = maximum_filter(a, size=3, mode='constant', cval=-np.inf)
        for j in range(1, n + 1):
            component = components == j
            first = int(np.flatnonzero(component)[0])
            level = a.ravel()[first]
            plateau = binary_propagation(component, structure=np.ones((3, 3)),
                                        mask=valid.numpy() & (a == level))
            # A locally flat patch is not a regional maximum if its same-value
            # connected plateau touches any higher pixel elsewhere.
            if np.any(neighborhood_max[plateau] > level):
                continue
            representatives.append(int(np.flatnonzero(plateau)[0]))
        representatives = list(set(representatives))
        representatives.sort(key=lambda j: (-float(a.ravel()[j]), j))
        regional_mask[k].view(-1)[representatives] = True
        kept = []
        for j in representatives:
            if all(np.linalg.norm((grid_np[j] - grid_np[t]) * scale) > radius_px for t in kept):
                kept.append(j)
                if len(kept) == count:
                    break
        for slot, j in enumerate(kept):
            xy[k, slot] = grid.view(-1, 2)[j]
            scores[k, slot] = probability[k].view(-1)[j]
            available[k, slot] = True
    return dict(xy=xy, probability=scores, available=available,
                probability_map=probability, regional_mask=regional_mask)
