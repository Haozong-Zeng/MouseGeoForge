"""Numerical primitives for the v4 geometry experiment (no model/data I/O)."""
from __future__ import annotations

import math
import numpy as np
import torch
from torch.nn import functional as F


def chordal_rotation(student, teacher):
    """1-cos(theta); zero loss AND zero derivative at identical rotations."""
    return (student - teacher.detach()).square().sum((-2, -1)) / 4


def body_scale(predicted, target, valid, body, per_view=False):
    """GT oracle, never a deployable estimate. Default: ONE scale per pack."""
    if predicted.shape != target.shape or target.shape != valid.shape:
        raise ValueError('Depth/mask shapes must match')
    mask = valid & body & torch.isfinite(predicted) & torch.isfinite(target)
    mask &= (predicted > 0) & (target > 0)
    residual = target.clamp_min(1e-12).log() - predicted.clamp_min(1e-12).log()
    if per_view:
        return [residual[v][mask[v]].median().exp() if mask[v].any() else None
                for v in range(len(target))]
    return residual[mask].median().exp() if mask.any() else None


def relative_pose(extrinsics, source, target):
    rotation = extrinsics[target, :3, :3] @ extrinsics[source, :3, :3].T
    translation = extrinsics[target, :3, 3] - rotation @ extrinsics[source, :3, 3]
    return rotation, translation


def component_oracles(e, k, z, known_e, known_k, known_z, scale):
    """Non-additive GT diagnostics in first-camera gauge, ONE shared body scale.

    Replacing rotation preserves predicted camera centers. Replacing centers
    preserves predicted rotations. Neither operation is a deployable prediction.
    """
    rotation = e[:, :3, :3] @ e[0, :3, :3].T
    translation = e[:, :3, 3] - (rotation @ e[0, :3, 3, None])[..., 0]
    canonical = torch.cat((rotation, (translation*scale)[..., None]), -1)
    centers = -(rotation.transpose(-1, -2) @ canonical[:, :3, 3, None])[..., 0]
    gt_centers = -(known_e[:, :3, :3].transpose(-1, -2) @ known_e[:, :3, 3, None])[..., 0]
    swapped_r = torch.cat((known_e[:, :3, :3], -(known_e[:, :3, :3] @ centers[..., None])), -1)
    swapped_c = torch.cat((rotation, -(rotation @ gt_centers[..., None])), -1)
    return dict(oracle_body_gauge_joint=(canonical, k, z*scale),
                oracle_replace_R_keep_centers=(swapped_r, k, z*scale),
                oracle_replace_centers_keep_R=(swapped_c, k, z*scale),
                oracle_replace_K=(canonical, known_k, z*scale),
                oracle_replace_Z=(canonical, k, known_z))


def angle_degrees(rotation):
    vee = torch.stack((rotation[..., 2, 1] - rotation[..., 1, 2],
                       rotation[..., 0, 2] - rotation[..., 2, 0],
                       rotation[..., 1, 0] - rotation[..., 0, 1]), -1)
    sine = torch.linalg.vector_norm(vee, dim=-1) / 2
    cosine = (rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    return torch.atan2(sine, cosine.clamp(-1, 1)) * (180 / math.pi)


def baseline_scale(predicted_e, known_e, roles, minimum=1e-8):
    """Candidate deployment scale: robust known/predicted non-bottom baselines.

    No annotations/depth GT used. None is an explicit estimation failure, not
    an unaligned fallback. This estimator does NOT ensure depth shares T's gauge.
    """
    ratios = []
    for a in range(len(roles)):
        for c in range(a + 1, len(roles)):
            if 'bottom' in (roles[a], roles[c]):
                continue
            _, pt = relative_pose(predicted_e, a, c)
            _, kt = relative_pose(known_e, a, c)
            pn, kn = pt.norm(), kt.norm()
            if torch.isfinite(pn) and torch.isfinite(kn) and pn > minimum and kn > minimum:
                ratios.append(kn / pn)
    return torch.stack(ratios).median() if ratios else None


def rotation_summary(values, peak_half_width=5.):
    x = np.asarray(values, dtype=np.float64)
    if not len(x):
        return dict(n=0, median=None, mean=None, p95=None, below5=None, below15=None)
    result = dict(n=len(x), median=float(np.median(x)), mean=float(x.mean()),
                  p95=float(np.sort(x)[math.ceil(.95 * len(x)) - 1]),
                  below5=float((x < 5).mean()), below15=float((x < 15).mean()))
    result['error_peak_fraction'] = {
        str(center): float((abs(x - center) <= peak_half_width).mean())
        for center in (90, 120, 180)}
    return result


def project_depth(e, k, z, source, target, xy):
    """World-to-camera reprojection without a unit-dependent positive-Z clamp."""
    uv = torch.cat((xy.T, torch.ones_like(xy.T[:1])), 0)
    camera = torch.linalg.solve(k[source], uv) * z[None]
    world = torch.linalg.solve(e[source, :3, :3], camera - e[source, :3, 3, None])
    target_xyz = e[target, :3, :3] @ world + e[target, :3, 3, None]
    hom = k[target] @ target_xyz
    good = torch.isfinite(hom).all(0) & (target_xyz[2] > 0)
    denominator = torch.where(good, hom[2], torch.ones_like(hom[2]))
    return (hom[:2] / denominator).T, target_xyz[2], good


def gradient_comparison(reference, actual):
    """Chunkwise float64 accumulation avoids invalid cosines from FP32 sums."""
    if set(reference) != set(actual):
        raise ValueError('Gradient parameter sets differ')
    aa = bb = ab = dd = 0.
    for name in reference:
        a, b = reference[name].double(), actual[name].double()
        aa += float(a.square().sum()); bb += float(b.square().sum())
        ab += float((a * b).sum()); dd += float((a - b).square().sum())
    if aa == 0 or bb == 0:
        return dict(relative_l2=0. if dd == 0 else float('inf'), cosine=1. if dd == 0 else 0.)
    return dict(relative_l2=math.sqrt(dd / aa), cosine=ab / math.sqrt(aa * bb))


def teacher_retention(pred, target, batch, decode):
    """Smooth teacher on reliable non-bottom pairs; target always detached."""
    pe, _ = decode(pred['pose_enc'], (512, 512))
    te, _ = decode(target['pose_enc'].detach(), (512, 512))
    losses = []
    for a in range(len(batch['roles'])):
        for c in range(a + 1, len(batch['roles'])):
            if 'bottom' in (batch['roles'][a], batch['roles'][c]):
                continue
            tr, tt = relative_pose(te[0], a, c)
            gr, _ = relative_pose(batch['E'][0], a, c)
            if float(angle_degrees(tr @ gr.T).detach()) > 10:
                continue
            pr, pt = relative_pose(pe[0], a, c)
            d = pred['depth'][0, [a, c], ::16, ::16, 0].clamp_min(1e-6)
            td = target['depth'][0, [a, c], ::16, ::16, 0].detach().clamp_min(1e-6)
            scale = (td.log() - d.log()).mean().exp()
            term = chordal_rotation(pr, tr)
            term = term + .1 * F.smooth_l1_loss(pred['pose_enc'][0, [a, c], 7:],
                                               target['pose_enc'][0, [a, c], 7:].detach())
            term = term + .1 * F.smooth_l1_loss((scale * d).log(), td.log())
            term = term + .1 * F.smooth_l1_loss(scale * pt, tt.detach())
            losses.append(term)
    return torch.stack(losses).mean() if losses else None


def complete_terms(pred, batch, *, teacher=None, real_anchor=True):
    # Import lazily so numerical CPU fixtures do not require the upstream model.
    import bottom_geometry_v3_losses_20261018 as v3
    import pure_geometry_bottom_20261018 as geo
    terms = v3.terms(pred, batch, 'A1-SI', real_anchor=real_anchor)
    if teacher is not None:
        value = teacher_retention(pred, teacher, batch, geo.encoding_to_camera)
        if value is not None:
            terms['teacher'] = value
    return terms
