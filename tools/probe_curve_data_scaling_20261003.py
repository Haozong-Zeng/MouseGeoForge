"""Paired three-seed data-scale control; frozen model and fixed validation."""
import collections
import datetime as dt
import fcntl
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
import numpy as np
import torch
from mousegeoforge.surface_curve_supervision import curve_loss, material_curve_loss, points_to_polyline
from probe_diverse_curve_features_20261003 import FeatureProbe
from run_depth_pose_ablation import write_json, fingerprint

CYCLE = ROOT / 'output/surface_curve_cycle_20261003'
SOURCE = CYCLE / 'expanded_curve_features_v1'
SMALL = CYCLE / 'diverse_curve_features_v1'
OUT = CYCLE / 'curve_data_scaling_v1'


def main():
    assert not OUT.exists(), 'Never overwrite an experiment'
    lock = (ROOT / 'output/teacher_temporal_cycle_20261001/gpu_job.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert torch.cuda.is_available()
    OUT.mkdir()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    deadline = dt.datetime(2026, 10, 4, 8, 5, 16, tzinfo=dt.timezone.utc)

    def status(stage, **kw):
        write_json(OUT / 'status.json', dict(stage=stage, pid=os.getpid(),
                   utc=dt.datetime.now(dt.timezone.utc).isoformat(), **kw))

    status('checking_inputs')
    saved = torch.load(SOURCE / 'frozen_inputs.pt', map_location='cpu', weights_only=False)
    original = torch.load(SMALL / 'frozen_inputs.pt', map_location='cpu', weights_only=False)
    n_old = len(original['metadata'])
    assert saved['metadata'][:n_old] == original['metadata']
    for key, value in original['tensors'].items():
        assert torch.equal(saved['tensors'][key][:n_old], value), key
    meta = saved['metadata']
    data = {k: v.cuda() for k, v in saved['tensors'].items()}
    small = [i for i, m in enumerate(meta[:n_old]) if m['role'] == 'train']
    expanded = [i for i, m in enumerate(meta) if m['role'] == 'train']
    val = [i for i, m in enumerate(meta) if m['role'] == 'validation']
    assert len(val) == 88 and max(val) < n_old
    assert not {meta[i]['scene'] for i in expanded} & {meta[i]['scene'] for i in val}
    seeds = [17, 29, 43]
    steps = 2400
    write_json(OUT / 'protocol.json', dict(
        source_sha256=fingerprint(SOURCE / 'frozen_inputs.pt'), original_sha256=fingerprint(SMALL / 'frozen_inputs.pt'),
        seeds=seeds, steps=steps, batch_size=8, learning_rate=.001, weight_decay=.001,
        feature='pose', parameterization='free_residual_6control_cubic_bspline',
        objective='Synthetic material Huber(beta0.01) + 0.1 curve shape loss(order0)',
        train_counts=dict(small=len(small), expanded=len(expanded)), validation_count=len(val),
        sampling='Uniform dataset then uniform instance-view observation; identical RNG seed per paired arm. New indices necessarily change example schedule.',
        initialization='Identical per seed and data-size pair. Seed17 small arm must reproduce objective-v1 final metric within1e-5px.',
        selection='Fixed final2400updates, no validation checkpoint selection or early stopping',
        existing_model_weights_frozen=True, all_original_inputs_bitwise_preserved=True,
        limitations=['Development validation: source/action separated but same synthetic assets, not external real data',
                     'Matched-instance metric, not end-to-end detection quality',
                     'Fixed-update comparison is compute matched, not equal epochs',
                     'No real manual point is assigned a material fraction; no locked real sessions loaded'],
        hard_deadline_utc=deadline.isoformat()))
    valid = torch.ones(65, device='cuda', dtype=torch.bool)
    material_u = torch.linspace(0, 1, 65, device='cuda')

    def evaluate(probe, ids):
        values = collections.defaultdict(list)
        with torch.no_grad():
            p = probe(data['pose'][ids], data['boxes'][ids], data['base_logits'][ids]) * data['pixels'][ids, None]
            g = data['gt'][ids] * data['pixels'][ids, None]
            for pp, gg in zip(p, g):
                values['pred_to_gt_px'].extend(points_to_polyline(pp, gg, valid)[0].sqrt().tolist())
                values['gt_to_pred_px'].extend(points_to_polyline(gg, pp, torch.ones(33, device='cuda', dtype=torch.bool))[0].sqrt().tolist())
                values['endpoints_px'].extend((pp[[0, -1]] - gg[[0, -1]]).norm(dim=-1).tolist())
        return {k: dict(mean=float(np.mean(v)), median=float(np.median(v)), p90=float(np.quantile(v, .9))) for k, v in values.items()}

    results, predictions = [], {}
    for seed in seeds:
        initial_state = None
        for size, train in [('small', small), ('expanded', expanded)]:
            arm = f'{size}_seed{seed}'
            buckets = collections.defaultdict(list)
            for i in train:
                buckets[meta[i]['dataset']].append(i)
            rng = random.Random(seed)
            keys = sorted(buckets)
            schedule = [[rng.choice(buckets[rng.choice(keys)]) for _ in range(8)] for _ in range(steps)]
            torch.manual_seed(seed)
            probe = FeatureProbe(False).cuda()
            if initial_state is None:
                initial_state = {k: v.detach().clone() for k, v in probe.state_dict().items()}
            else:
                assert all(torch.equal(v, initial_state[k]) for k, v in probe.state_dict().items())
            optimizer = torch.optim.AdamW(probe.parameters(), lr=.001, weight_decay=.001)
            history = [dict(step=0, train=evaluate(probe, train), validation=evaluate(probe, val))]
            status('training', arm=arm, step=0)
            for step, ids in enumerate(schedule, 1):
                assert dt.datetime.now(dt.timezone.utc) < deadline, 'Hard training deadline'
                optimizer.zero_grad(set_to_none=True)
                pixels = data['pixels'][ids]
                p = probe(data['pose'][ids], data['boxes'][ids], data['base_logits'][ids]) * pixels[:, None]
                gt = data['gt'][ids] * pixels[:, None]
                loss = torch.stack([
                    material_curve_loss(pp, gg, valid, material_u, material_parameter_known=True, scale=768)
                    + .1 * curve_loss(pp, gg, valid, full_curve=True, scale=768, order_weight=0.)
                    for pp, gg in zip(p, gt)]).mean()
                assert torch.isfinite(loss)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.)
                assert torch.isfinite(norm)
                optimizer.step()
                if step % 400 == 0:
                    row = dict(step=step, loss=float(loss.detach()), train=evaluate(probe, train), validation=evaluate(probe, val))
                    history.append(row)
                    write_json(OUT / f'{arm}_history.json', history)
                    status('training', arm=arm, step=step)
                    print(arm, step, row['train']['pred_to_gt_px']['mean'], row['validation']['pred_to_gt_px']['mean'], flush=True)
            if seed == 17 and size == 'small':
                reference = json.loads((CYCLE / 'curve_objectives_v1/pose_material_huber_history.json').read_text())[-1]
                for role in ['train', 'validation']:
                    assert abs(history[-1][role]['pred_to_gt_px']['mean'] - reference[role]['pred_to_gt_px']['mean']) < 1e-5
            torch.save(dict(state=probe.state_dict(), seed=seed, size=size, step=steps), OUT / f'{arm}.pt')
            results.append(dict(arm=arm, seed=seed, size=size, **history[-1]))
            write_json(OUT / 'partial_results.json', results)
            with torch.no_grad():
                predictions[arm] = probe(data['pose'], data['boxes'], data['base_logits']).cpu().numpy()
    np.savez_compressed(OUT / 'predictions.npz', **predictions)
    write_json(OUT / 'results.json', dict(results=results, inspection_pending=True, no_production_promotion=True))
    status('complete')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        OUT.mkdir(exist_ok=True)
        write_json(OUT / 'failure.json', dict(traceback=__import__('traceback').format_exc()))
        raise
