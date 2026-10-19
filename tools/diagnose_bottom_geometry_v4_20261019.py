"""Fixed TRAIN probes: task/module norms and streamed cross-source gradients."""
import collections
import math

import torch
from torch.nn import functional as F

import bottom_geometry_v3_data_20261018 as data
import bottom_geometry_v4_core_20261019 as core
import bottom_geometry_v4_runtime_20261019 as rt
import calibrate_bottom_geometry_v4_20261019 as calibration
import resume_geometry_workers_20261018 as state


def diagnose(model, teacher, manifest, weights, destination, halt=lambda: None):
    items = calibration.probe_items(manifest); named = {n: p for n, p in model.named_parameters() if p.requires_grad}
    feed = data.Feed(manifest, items, 0); rng = rt.geo.old.rng_state()
    source_sums = {}; counts = collections.Counter(); rows = []
    try:
        for item in items:
            halt(); batch = feed.next(item); pred = model(batch['images']); target = None
            if teacher is not None:
                with torch.no_grad():
                    target = teacher(batch['images'])
            terms = core.complete_terms(pred, batch, teacher=target, real_anchor=True)
            norms = {name: calibration.module_norms(value, named) for name, value in terms.items()}
            pe, _ = rt.geo.encoding_to_camera(pred['pose_enc'], (512, 512))
            pR = pe[:, 1:, :3, :3] @ pe[:, :1, :3, :3].transpose(-1, -2)
            gR = batch['E'][:, 1:, :3, :3] @ batch['E'][:, :1, :3, :3].transpose(-1, -2)
            r_loss = rt.geo.rotation_angle(pR @ gR.transpose(-1, -2)).mean()
            pt = pe[:, 1:, :3, 3] - (pR @ pe[:, :1, :3, 3, None])[..., 0]
            gt = batch['E'][:, 1:, :3, 3] - (gR @ batch['E'][:, :1, :3, 3, None])[..., 0]
            t_loss = (F.normalize(pt, dim=-1)-F.normalize(gt, dim=-1)).square().sum(-1).mean()
            encoded = rt.geo.extri_intri_to_pose_encoding(batch['E'], batch['K'], (512, 512))
            fov_loss = F.smooth_l1_loss(pred['pose_enc'][..., 7:], encoded[..., 7:])
            component_items = [('fov', fov_loss)]
            if len(batch['roles']) > 1:
                component_items += [('relative_rotation_rad', r_loss), ('translation_direction', t_loss)]
            components = {k: dict(value=float(v.detach()), module_norms=calibration.module_norms(v, named))
                          for k, v in component_items}
            total = sum(weights[k]*value for k, value in terms.items())
            gradients = torch.autograd.grad(total, list(named.values()), allow_unused=True)
            role = item['sampling_role']; counts[role] += 1
            if role not in source_sums:
                source_sums[role] = {n: torch.zeros_like(p, device='cpu', dtype=torch.float32) for n, p in named.items()}
            for (name, _), gradient in zip(named.items(), gradients):
                if gradient is not None:
                    source_sums[role][name].add_(gradient.detach().cpu().float())
            rows.append(dict(item=item, group=batch['record']['group'], module_norms=norms,
                             weighted_module_norms={name: {m: weights[name]*v for m, v in value.items()} for name, value in norms.items()},
                             camera_components=components))
            del pred, target, terms, batch, gradients, total, r_loss, t_loss, fov_loss, pe, pR, gR, pt, gt, component_items
        cosines = {}
        for a in sorted(source_sums):
            for b in sorted(source_sums):
                if a >= b:
                    continue
                for module, prefix in [('all', ''), ('encoder', 'aggregator.'), ('camera', 'camera_head.')]:
                    aa = bb = ab = 0.
                    for name in named:
                        if not name.startswith(prefix):
                            continue
                        x, y = source_sums[a][name].double(), source_sums[b][name].double()
                        aa += float(x.square().sum()); bb += float(y.square().sum()); ab += float((x*y).sum())
                    cosines[f'{a}/{b}/{module}'] = ab/math.sqrt(aa*bb) if aa > 0 and bb > 0 else None
        rt.write_new(destination, dict(rows=rows, cross_source_cosines=cosines, source_probe_counts=dict(counts),
                                      physical_probe_batch=1, weights_frozen=True, optimizer_steps=0,
                                      note='Exact full trainable-parameter gradients, equal probes within source; directions of sums equal source means'))
    finally:
        feed.close(); state.restore_rng(rng)
