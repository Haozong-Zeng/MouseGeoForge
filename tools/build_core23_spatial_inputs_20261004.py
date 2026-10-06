"""Extract all semantic point tokens without changing source data or production."""
import collections
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'vggt-omega'), str(ROOT / 'tools')]
import torch
from torch import nn
from training.instance_pose_losses import instance_targets_from_batch, hungarian_match_instances
from vggt_omega.models.heads.dense_head import DenseHead
from vggt_omega.models.heads.instance_pose_head import MultiInstanceMouseHead
from mousegeoforge.models.surface_curve_pose import SurfaceCurveDepthPoseModel
from run_depth_pose_ablation import fingerprint, write_json, move_batch
from mousegeoforge.pilot_metrics import box_iou

OLD = ROOT / 'output/surface_curve_cycle_20261003'
CYCLE = ROOT / 'output/core23_spatial_cycle_20261004'
OUT = CYCLE / 'features_v1'


def now():
    return dt.datetime.now(dt.timezone.utc)


def status(stage, **extra):
    write_json(OUT / 'status.json', dict(stage=stage, pid=os.getpid(), utc=now().isoformat(), **extra))


def main():
    plan = json.loads((CYCLE / 'plan.json').read_text())
    deadline = dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z', '+00:00'))
    assert now() < deadline
    lock = (ROOT / plan['gpu_lock']).open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not OUT.exists(), 'Do not overwrite a partial extraction; inspect/resume explicitly.'
    OUT.mkdir()
    (OUT / 'parts').mkdir()
    state = json.loads((CYCLE / 'state.json').read_text())
    state.update(stage='extracting_full23_features', active_job='features_v1', active_pid=os.getpid(), updated_utc=now().isoformat())
    write_json(CYCLE / 'state.json', state)
    status('loading')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    model = SurfaceCurveDepthPoseModel(nn.Identity(), DenseHead(), MultiInstanceMouseHead(num_keypoints=23)).cuda().eval()
    initial = OLD / 'training_v1/pilot/initial_heads.pt'
    assert fingerprint(initial) == '9448daa0318eb0c18af2e79ce6f2e320cf82283166ccae6c56e9dbe97df13fc3'
    model.load_state_dict(torch.load(initial, map_location='cpu', weights_only=True), strict=True)
    model.requires_grad_(False)
    golden = torch.load(OLD / 'back_point_probe_inputs_v1/real_inputs.pt', map_location='cpu', weights_only=False)
    gold_by_key = collections.defaultdict(list)
    for i, m in enumerate(golden['metadata']):
        gold_by_key[m['cache_key']].append((i, m))
    paths = sorted((OLD / 'training_v1/pilot/cache').glob('*.pt'))
    assert len(paths) == 192
    rows, seen, counts = [], {}, collections.Counter()
    max_delta, golden_count = 0., 0
    for n, source in enumerate(paths):
        if now() >= deadline:
            raise TimeoutError('Feature extraction reached training deadline')
        item = torch.load(source, map_location='cpu', weights_only=False)
        r = item['record']
        assert r.get('session') not in ('GR2', 'UR2', 'UR3')
        assert all(fingerprint(ROOT / p) == sha for p, sha in item['source_hashes'].items())
        b = move_batch(item['batch'], 'cuda')
        memory, pose = [], []
        a = model.instance_pose_head.instance_decoder.register_forward_pre_hook(lambda m, args: memory.append(args[1].detach()))
        hook = model.instance_pose_head.keypoint_decoder.register_forward_hook(lambda m, args, out: pose.append(out.detach()))
        try:
            with torch.no_grad():
                pred = model.forward_cached(b['images'], [None if x is None else x.cuda() for x in item['cached']], item['patch_start'], coupling='off')
        finally:
            a.remove()
            hook.remove()
        vcount = b['images'].shape[1]
        h, w = b['images'].shape[-2:]
        maps = memory[0].transpose(1, 2).reshape(vcount, 256, h // 16, w // 16).cpu()
        tokens = pose[0].reshape(vcount, 6, 23, 256).cpu()
        assert torch.isfinite(maps).all() and torch.isfinite(tokens).all()
        for i, m in gold_by_key[r['key']]:
            old = tokens[m['cache_view_index'], m['query'], [4, 5, 6, 7]].flatten()
            delta = float((old - golden['tensors']['pose'][i]).abs().max())
            assert delta < 2e-4, (r['key'], delta)
            max_delta = max(max_delta, delta)
            golden_count += 1
        target = instance_targets_from_batch(b)
        match_target = dict(target)
        match_target['keypoint_valid'] = torch.zeros_like(target['keypoint_valid'])
        matches = hungarian_match_instances(pred, match_target)
        observations = []
        for v, (queries, instances) in enumerate(matches[0]):
            count = len(instances)
            for q, m in zip(queries.tolist(), instances.tolist()):
                identity = (r['scene'], r['views'][v], m)
                if identity in seen:
                    assert seen[identity] == r['role'], 'Duplicate source crossed train/validation'
                    continue
                seen[identity] = r['role']
                valid = target['keypoint_valid'][0, v, m].cpu()
                assert valid.shape == (23,)
                if r['kind'] == 'real':
                    allowed = torch.zeros(23, dtype=torch.bool)
                    allowed[[0, 5, 6, 14, 15, 16, 17]] = True
                    assert not (valid & ~allowed).any(), 'Review real label provenance before expansion'
                counts[(r['kind'], r['role'], count)] += 1
                iou = box_iou(pred['instance_boxes_cxcywh'][0, v, q:q+1], target['boxes_cxcywh'][0, v, m:m+1])[0, 0]
                observations.append(dict(
                    metadata=dict(record=r, view=r['views'][v], view_index=v, query=q, instance=m, instance_count=count,
                                  box_iou=float(iou), objectness=float(pred['instance_objectness_logits'][0, v, q].sigmoid())),
                    pose=tokens[v, q].clone(),
                    coarse_xy=pred['instance_keypoint_coarse_xy'][0, v, q].detach().cpu(),
                    original_xy=pred['instance_keypoint_xy'][0, v, q].detach().cpu(),
                    gt_xy=target['keypoint_xy'][0, v, m].cpu(), valid=valid,
                    visible=b['instance_keypoint_visible'][0, v, m].cpu(),
                    visibility_valid=b['instance_keypoint_visibility_valid'][0, v, m].cpu()))
        dest = OUT / 'parts' / source.name
        torch.save(dict(memory=maps.half(), image_hw=[h, w], observations=observations), dest)
        rows.append(dict(key=r['key'], path=dest.relative_to(ROOT).as_posix(), sha256=fingerprint(dest), record=r,
                         count=len(observations), source_hashes=item['source_hashes']))
        status('extracting', completed=n+1, total=len(paths), observations=len(seen), golden_count=golden_count, max_golden_delta=max_delta)
        print('FEATURES', n+1, len(paths), 'observations', len(seen), flush=True)
    assert golden_count == len(golden['metadata'])
    write_json(OUT / 'manifest.json', dict(
        records=rows, observations=len(seen), counts={str(k): v for k, v in counts.items()},
        initial_sha256=fingerprint(initial), script_sha256=fingerprint(Path(__file__)),
        golden_count=golden_count, max_golden_delta=max_delta,
        semantics='All 23 semantic tokens; original coarse AND local-refined readout; prediction-only memory. GT box/objectness assignment for conditional pose evaluation only. First paired context per scene/view/instance, immutable validity; not end-to-end detection.',
        real_supported_ids=[1,6,7,15,16,17,18], no_production_promotion=True))
    status('complete', observations=len(seen))
    state.update(stage='full23_features_ready', active_pid=None, active_job=None, updated_utc=now().isoformat(),
                 next='Run matched Core23SpatialReadout global/spatial strict pilot seeds17/29/43, 1600steps with200step curves. Report exact original full head baseline and all supported-point strata; visually inspect before extending.')
    state['completed'].append('full23_features_and_old_token_goldens_verified')
    write_json(CYCLE / 'state.json', state)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        OUT.mkdir(exist_ok=True)
        status('failed', traceback=__import__('traceback').format_exc())
        raise
