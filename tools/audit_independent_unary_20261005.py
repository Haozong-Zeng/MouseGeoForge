"""Frozen unary eligibility intervention; no training or GT in forward inputs."""
import collections
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools'), str(ROOT / 'vggt-omega')]
import numpy as np
import torch
from mousegeoforge.models.local_candidate_selector import LocalCandidateSelector
from mousegeoforge.candidate_selector_features import KP
from mousegeoforge.detection_readout_loss import match_detection
from mousegeoforge.pilot_metrics import box_iou
from run_depth_pose_ablation import fingerprint, write_json

CYCLE = ROOT / 'output/singleview_fusion_cycle_20261005'
SOURCE = CYCLE / 'local_candidate_selector_pilot_v1'
OUT = CYCLE / 'independent_unary_eligibility_audit_v1'


def independent_inputs(data):
    inputs = dict(data['inputs'])
    confident = data['logits'].sigmoid() >= .5
    unique = confident & (confident.sum(-1, keepdim=True) == 1)
    inputs['eligible'] = unique[..., None].expand(-1, -1, 5)
    inputs['geometry'] = torch.zeros_like(inputs['geometry'])
    return inputs


@torch.no_grad()
def predict(model, inputs):
    outputs = []
    for start in range(0, len(inputs['eligible']), 32):
        batch = {k: v[start:start + 32] for k, v in inputs.items()}
        outputs.append(model(**batch, enabled=True))
    return {k: torch.cat([x[k] for x in outputs]) for k in ['xy', 'active', 'choice', 'eligible']}


def metrics(data, xy, active, choices):
    matches = match_detection(dict(instance_boxes_cxcywh=data['boxes'], instance_objectness_logits=data['logits']), data['gt_boxes'], data['box_valid'] & data['instance_valid'])
    groups = collections.defaultdict(list)
    points = []
    for i, meta in enumerate(data['metadata']):
        for qq, tt in zip(*matches[i]):
            q, t = int(qq), int(tt)
            hit = bool(data['logits'][i, q].sigmoid() >= .5 and box_iou(data['boxes'][i, q:q+1], data['gt_boxes'][i, t:t+1])[0, 0] >= .5)
            for ki, kp in enumerate(KP):
                if not data['point_valid'][i, t, kp]:
                    continue
                error = float(((xy[i, q, kp] - data['gt_xy'][i, t, kp]) * data['pixels'][i]).norm())
                base = float(((data['base_xy'][i, q, kp] - data['gt_xy'][i, t, kp]) * data['pixels'][i]).norm())
                p = dict(row=i, point=kp+1, query=q, target=t, error=error, baseline_error=base, changed=bool(active[i, q, ki]), choice=int(choices[i, q, ki]), hit=hit)
                points.append(p)
                names = [meta['kind'] + '/reliable']
                if kp:
                    names.append(meta['kind'] + '/view' + str(meta['view']) + '/feet')
                for name in names:
                    groups[name].append(p)
    result = {}
    for name, ps in groups.items():
        errors = np.array([p['error'] for p in ps])
        delta = errors - np.array([p['baseline_error'] for p in ps])
        result[name] = dict(points=len(ps), mean_error=float(errors.mean()), median_error=float(np.median(errors)), p95_error=float(np.percentile(errors, 95)), gated_pck32=float(np.mean([p['hit'] and p['error'] <= 32 for p in ps])), changed_points=sum(p['changed'] for p in ps), improved_over16=int((delta < -16).sum()), harmed_over16=int((delta > 16).sum()), worst_harm=float(delta.max()))
    return result, points


@torch.no_grad()
def main():
    assert not OUT.exists()
    OUT.mkdir()
    torch.set_num_threads(2)
    model = LocalCandidateSelector('unary').eval()
    model.load_state_dict(torch.load(SOURCE / 'unary_latest.pt', map_location='cpu', weights_only=False)['state'])
    prior = json.loads((SOURCE / 'results.json').read_text())
    golden = np.load(SOURCE / 'predictions.npz')
    report = dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(), checkpoint_sha256=fingerprint(SOURCE / 'unary_latest.pt'), script_sha256=fingerprint(Path(__file__)), training=False, production_changed=False, protocol='Frozen step800 unary. Independent gate: exactly one predicted query >=0.5 in this image; all five reliable points. No camera/GT/visibility inputs. Explicit zero geometry. Evaluation matching occurs after all forward calls. No hyperparameter search. Reused development validation; not external test. All images/valid matched labels included, including fallback. No multi-instance generalization claim.', results=[], tests={}, input_sha256={})
    arrays = {}
    for role in ['train', 'validation']:
        file = SOURCE / f'{role}_inputs.pt'
        report['input_sha256'][role] = fingerprint(file)
        assert report['input_sha256'][role] == prior['input_sha256'][role]
        data = torch.load(file, map_location='cpu', weights_only=False)
        independent = independent_inputs(data)
        outputs = dict(matched_support=predict(model, data['inputs']), independent_unique=predict(model, independent))
        # Batch shape and ordering must not change this image-only network.
        if role == 'validation':
            ref = outputs['independent_unique']
            singleton = torch.cat([model(**{k: v[i:i+1] for k, v in independent.items()}, enabled=True)['xy'] for i in range(len(data['base_xy']))])
            perm = torch.randperm(len(singleton), generator=torch.Generator().manual_seed(17))
            permuted = predict(model, {k: v[perm] for k, v in independent.items()})['xy'][perm.argsort()]
            duplicated = predict(model, {k: torch.cat([v, v]) for k, v in independent.items()})['xy']
            changed_geometry = dict(independent, geometry=torch.full_like(independent['geometry'], float('nan')))
            no_geometry = predict(model, changed_geometry)['xy']
            for name, output in [('singleton', singleton), ('permutation', permuted), ('duplicate', duplicated[:len(singleton)]), ('ignored_nan_geometry', no_geometry)]:
                torch.testing.assert_close(output, ref['xy'], atol=2e-5, rtol=0)
                report['tests'][name + '_max_normalized_difference'] = float((output - ref['xy']).abs().max())
            assert model(**independent)['xy'] is independent['baseline_xy']
            report['tests']['default_off_exact'] = True
        # Only now consult labels and matching for metrics.
        for arm in ['baseline', 'matched_support', 'independent_unique']:
            xy = data['base_xy'].clone()
            active = torch.zeros_like(data['inputs']['eligible'])
            choices = torch.zeros_like(active, dtype=torch.long)
            eligible = active
            if arm != 'baseline':
                output = outputs[arm]
                xy[:, :, KP] = output['xy']
                active, choices, eligible = output['active'], output['choice'], output['eligible']
            if arm == 'matched_support':
                difference = float(np.max(np.abs(xy.numpy() - golden[role + '_unary'])))
                assert difference < 2e-5, difference
                report['tests'][role + '_prior_gpu_replay_max_difference'] = difference
            values, points = metrics(data, xy, active, choices)
            report['results'].append(dict(role=role, arm=arm, metrics=values, eligible_points=int(eligible.sum()), changed_all_queries=int(active.sum())))
            arrays[role + '_' + arm] = xy.numpy()
            arrays[role + '_' + arm + '_active'] = active.numpy()
            write_json(OUT / f'{role}_{arm}_points.json', points)
            print(role, arm, {k: round(v['mean_error'], 3) for k, v in values.items() if k in ['real/reliable', 'real/view0/feet', 'synthetic/reliable']}, flush=True)
    np.savez_compressed(OUT / 'predictions.npz', **arrays)
    write_json(OUT / 'results.json', report)


if __name__ == '__main__':
    main()
