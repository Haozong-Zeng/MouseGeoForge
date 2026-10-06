"""Read-only matched-loss decomposition and parameter-gradient comparisons."""
from types import FunctionType
import torch
from torch.nn import functional as F
import training.instance_pose_losses as upstream


def matched_task_terms(predictions, targets, weights, **kwargs):
    matches = upstream.hungarian_match_instances(predictions, targets)
    def fixed_matching(p, t):
        if p is not predictions or t is not targets:
            raise AssertionError('Different matching inputs')
        return matches
    original = upstream.instance_pose_loss
    fixed = FunctionType(original.__code__, dict(original.__globals__, hungarian_match_instances=fixed_matching),
                         original.__name__, original.__defaults__, original.__closure__)
    fixed.__kwdefaults__ = original.__kwdefaults__
    loss, logs = fixed(predictions, targets, weights, **kwargs)
    xy = predictions['instance_keypoint_xy']; body = xy.sum() * 0.; tail = body; count = 0
    for b, views in enumerate(matches):
        for v, (qs, ms) in enumerate(views):
            for q, m in zip(qs.tolist(), ms.tolist()):
                valid = targets['keypoint_valid'][b, v, m].bool()
                body_valid = valid.clone(); body_valid[7:14] = False
                tail_valid = valid & ~body_valid
                gt = targets['keypoint_xy'][b, v, m]
                count += int(valid.sum())
                if body_valid.any(): body = body + F.smooth_l1_loss(xy[b,v,q,body_valid], gt[body_valid], beta=.01, reduction='sum')
                if tail_valid.any(): tail = tail + F.smooth_l1_loss(xy[b,v,q,tail_valid], gt[tail_valid], beta=.01, reduction='sum')
    factor = weights.keypoint_l1 / max(2 * count, 1)
    return loss, logs, dict(body=body * factor, tail=tail * factor, valid_points=count)


def compare_gradient_sets(cylinder, dual):
    """Float64 Gram products; None for undefined cosine, not artificial zero."""
    a = torch.stack(cylinder).double(); b = torch.stack(dual).double()
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError('Mismatched gradient vectors')
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError('Nonfinite gradients')
    dots = a @ b.T; norms = a.norm(dim=1)[:,None] * b.norm(dim=1)[None,:]
    valid = norms > 1e-20
    cosines = dots[valid] / norms[valid]
    ma, mb = a.mean(0), b.mean(0); denominator = float(ma.norm() * mb.norm())
    return dict(mean_gradient_cosine=float(ma.dot(mb)) / denominator if denominator > 1e-20 else None,
        mean_gradient_dot=float(ma.dot(mb)), cylinder_mean_norm=float(ma.norm()), dual_mean_norm=float(mb.norm()),
        valid_pairs=int(valid.sum()), total_pairs=dots.numel(),
        negative_pair_fraction=float((cosines < 0).double().mean()) if cosines.numel() else None,
        pair_cosines=[[float(dots[i,j]/norms[i,j]) if valid[i,j] else None for j in range(len(b))] for i in range(len(a))])


def select_task_records(spec):
    seq = [spec['records'][item[0]] for item in spec['arms']['source_mix']]
    dual = sorted({r['scene']: r for r in seq if r['expected_instances'] == 2}.values(), key=lambda r: (r['source_frame'], r['frame']))
    if len(dual) != 8: raise ValueError('Expected existing eight dual replay poses')
    first = {}
    for r in seq:
        if len(r['views']) == 2 and sorted(r['physical_roles']) == ['bottom', 'side']:
            first.setdefault(r['scene'], r)
    by_source = {}
    for r in first.values(): by_source.setdefault((r['dataset'], r['source_group']), []).append(r)
    if len(by_source) != 2: raise ValueError('Expected two Cylinder source datasets')
    records = [dict(kind='dual', record=r) for r in dual]
    for _, pool in sorted(by_source.items()):
        pool = sorted(pool, key=lambda r: (r.get('source_frame', r['frame']), r['frame']))
        if len(pool) < 4: raise ValueError('Not enough Cylinder training poses')
        records.extend(dict(kind='cylinder', record=pool[round(i*(len(pool)-1)/3)]) for i in range(4))
    if not all(e['record']['split'] == 'train' for e in records): raise ValueError('Non-training record')
    return records
