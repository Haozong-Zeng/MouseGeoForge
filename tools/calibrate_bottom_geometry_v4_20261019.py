"""TRAIN-only module budgets, with fixed perturbations; ZERO optimizer updates."""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

import numpy as np
import torch

import bottom_geometry_v3_data_20261018 as data
import bottom_geometry_v4_core_20261019 as core
import bottom_geometry_v4_runtime_20261019 as rt


def module_norms(value, named):
    grads = torch.autograd.grad(value, list(named.values()), retain_graph=True, allow_unused=True)
    sums = collections.Counter()
    for (name, _), gradient in zip(named.items(), grads):
        if gradient is not None:
            module = 'encoder' if name.startswith('aggregator.') else 'camera' if name.startswith('camera_head.') else 'depth'
            sums[module] += float(gradient.detach().double().square().sum())
    return {k: math.sqrt(v) for k, v in sums.items()}


def perturbed_student(pred):
    pose = pred['pose_enc']; q = pose[..., 3:7]
    # Left-compose alternating 5deg Z rotations in XYZW convention; preserve q norm.
    sign = torch.where(torch.arange(q.shape[1], device=q.device) % 2 == 0, 1., -1.)
    sine = (sign*math.radians(5)/2).sin(); cosine = math.cos(math.radians(5)/2)
    x, y, z, w = q.unbind(-1)
    rotated = torch.stack((cosine*x-sine*y, cosine*y+sine*x, cosine*z+sine*w, cosine*w-sine*z), -1)
    modified = torch.cat((pose[..., :3], rotated, pose[..., 7:]), -1)
    return dict(pose_enc=modified, depth=pred['depth']*1.02)


def budget_weight(norms, references, fraction, maximum):
    candidates = [maximum]
    for module in ('encoder', 'camera'):
        values = [r[module] for r in norms if r.get(module, 0) > 1e-12]
        if not values or references.get(module, 0) <= 0:
            raise ValueError('Insufficient active module probes for ' + module)
        candidates.append(fraction*references[module]/float(np.percentile(values, 75)))
    value = min(candidates)
    if not math.isfinite(value) or value <= 0:
        raise ValueError('Invalid gradient budget')
    return value


def probe_items(manifest):
    all_items = data.calibration_items(manifest); chosen = []
    for role, count in [('new_rig', 12), ('real', 8), ('bottom', 4), ('replay', 4), ('dannce', 4)]:
        pool = [x for x in all_items if x['sampling_role'] == role]
        chosen += [pool[int(i)] for i in np.linspace(0, len(pool)-1, count).round()]
    if any(manifest['records'][x['index']]['split'] != 'train' for x in chosen):
        raise ValueError('Calibration must use TRAIN only')
    return chosen


def calibrate(manifest, release, weights):
    chosen = probe_items(manifest)
    model, _ = rt.model(release, trainable=True)
    named = {n: p for n, p in model.named_parameters() if p.requires_grad}
    references = collections.defaultdict(list); teacher_norms = []; anchor_norms = []; rows = []
    feed = data.Feed(manifest, chosen, 0)
    try:
        for item in chosen:
            batch = feed.next(item); pred = model(batch['images'])
            target = {k: pred[k].detach() for k in ('pose_enc', 'depth')}
            terms = core.complete_terms(pred, batch, real_anchor=True)
            measured = {name: module_norms(value, named) for name, value in terms.items()}
            for name, norms in measured.items():
                if name in ('quaternion_radius', 'real_camera_anchor'):
                    continue
                for module, norm in norms.items():
                    if norm > 1e-12:
                        references[module].append(weights[name]*norm)
            teacher = core.teacher_retention(perturbed_student(pred), target, batch, rt.geo.encoding_to_camera)
            tn = module_norms(teacher, named) if teacher is not None else None
            if tn:
                teacher_norms.append(tn)
            if 'real_camera_anchor' in measured:
                anchor_norms.append(measured['real_camera_anchor'])
            rows.append(dict(item=item, group=batch['record']['group'], term_module_norms=measured,
                             perturbed_teacher_module_norms=tn))
            del pred, target, terms, batch, teacher
    finally:
        feed.close()
    reference = {module: float(np.median(values)) for module, values in references.items()}
    result = dict(weights)
    result['real_camera_anchor'] = budget_weight(anchor_norms, reference, 1., 1.)
    result['teacher'] = budget_weight(teacher_norms, reference, .2, .05)
    return dict(passed=True, weights=result, reference_module_medians=reference, rows=rows,
                optimizer_steps=0, selection_data='TRAIN ONLY', teacher_budget_fraction=.2,
                anchor_budget_fraction=1., teacher_weight_maximum=.05,
                perturbation='student 5deg alternating camera Z rotations +2% depth, target frozen release',
                rule='Fixed coefficient <= module budget / P75 active probe norm, both encoder and camera head; freeze before formal updates',
                warning='Initialization calibration only; log later gradients without adapting weights mid-arm')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Preserve prior calibration; choose a new output')
    cfg = json.loads(args.config.read_text()); manifest_path = rt.geo.ROOT/cfg['manifest']; release = rt.geo.ROOT/cfg['release']
    import fcntl
    lock_path = rt.gpu_lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report = calibrate(rt.read_manifest(manifest_path), release, cfg['weights'])
        report.update(manifest_sha256=rt.sha(manifest_path), release_sha256=rt.sha(release),
                      source_sha256=rt.sha(__file__))
        rt.write_new(args.output, report)


if __name__ == '__main__':
    main()
