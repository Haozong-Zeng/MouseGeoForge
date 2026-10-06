"""Frozen original/mixed evaluation on seen and nonreplayed dual TRAIN poses."""
from collections import Counter
import fcntl
import json
import time
import numpy as np
from run_depth_pose_ablation import (ROOT, torch, VGGTOmega, DepthPoseModel, restore, tensor_state,
    state_hash, fingerprint, write_json, instance_targets_from_batch, load_scene, load_batch)
from train_cylinder_view_balance import oriented_batch, scene_digest
from training.finetune_blendedmvs import median_camera_baseline
from audit_bottom_view_coverage import cameras
from mousegeoforge.pilot_metrics import evaluate_scene
from mousegeoforge.retention_geometry_audit import matched_views, BODY
from mousegeoforge.replay_coverage import select_unreplayed, summarize_views

OUT = ROOT / 'output/cylinder_replay_coverage_v1'
PARENT = ROOT / 'output/cylinder_source_mix_v1'


def pose_distance(a, b):
    """Same-ID/view centered body shape, pixel-scaled by each GT box diagonal."""
    pixel = torch.tensor(a['record']['image_hw'][::-1])
    ta, tb = a['targets'], b['targets']
    errors = []
    for v in range(2):
        ids_a = ta['instance_ids'][0, v].tolist()
        ids_b = tb['instance_ids'][0, v].tolist()
        for m, identity in enumerate(ids_a):
            if identity < 0 or identity not in ids_b:
                continue
            n = ids_b.index(identity)
            valid = ta['keypoint_valid'][0, v, m, list(BODY)] & tb['keypoint_valid'][0, v, n, list(BODY)]
            shapes = []
            for t, slot in ((ta, m), (tb, n)):
                box = t['boxes_cxcywh'][0, v, slot]
                shapes.append((t['keypoint_xy'][0, v, slot, list(BODY)] - box[:2]) * pixel / (box[2:] * pixel).norm().clamp_min(1))
            errors.extend((shapes[0] - shapes[1]).norm(dim=-1)[valid].tolist())
    if not errors:
        raise ValueError('No comparable identity/label for pose distance')
    return float(np.mean(errors))


def run():
    start = time.monotonic()
    spec = json.loads((PARENT / 'protocol.json').read_text())
    assert json.loads((PARENT / 'status.json').read_text())['state'] == 'completed'
    assert json.loads((PARENT / 'delivery_verification.json').read_text())['passed']
    with (PARENT / 'running.lock').open() as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for p, h in spec['code_sha256'].items():
        assert fingerprint(ROOT / p) == h
    manifest_path = ROOT / spec['config']['manifest']
    assert fingerprint(manifest_path) == spec['manifest_sha256']
    manifest = json.loads(manifest_path.read_text())['records']
    replay = sorted({spec['records'][k[0]]['scene']: spec['records'][k[0]]
                     for k in spec['arms']['source_mix'] if spec['records'][k[0]]['expected_instances'] == 2}.values(), key=lambda r: r['source_frame'])
    assert len(replay) == 8
    selected = select_unreplayed(manifest, {r['scene'] for r in replay})
    cohort = [dict(cohort='replayed', record=r) for r in replay]
    for r in selected:
        scene = ROOT / r['scene']
        roles = {c['view_id']: c['orientation'] for c in cameras(scene)}
        loaded = load_scene(scene)
        scale = float(median_camera_baseline(np.stack([loaded.frames[v].extrinsic for v in r['views']])))
        actual = list(map(int, load_batch(r)['view_ids'][0].split(',')))
        record = dict(r, views=actual, diagnostic_view_order=actual, normalization_scale=scale,
                      physical_roles=[roles[v] for v in actual], source_revision=scene_digest(scene))
        assert actual == replay[0]['views'] and record['physical_roles'] == ['side', 'top']
        cohort.append(dict(cohort='nonreplayed', record=record))
    paths = {'original': ROOT / spec['config']['checkpoint'], 'mixed': PARENT / 'source_mix/final_heads.pt'}
    sources = {e['record']['scene']: e['record']['source_revision'] for e in cohort}
    assert all(scene_digest(ROOT / p) == h for p, h in sources.items())
    code = {p: fingerprint(ROOT / p) for p in ('tools/audit_cylinder_replay_coverage.py',
            'mousegeoforge/replay_coverage.py', 'mousegeoforge/retention_geometry_audit.py')}
    protocol = dict(cohort=cohort, source_sha256=sources, code_sha256={**spec['code_sha256'], **code},
        checkpoints={tag: dict(path=str(p.relative_to(ROOT)), sha256=fingerprint(p)) for tag, p in paths.items()},
        parent_protocol_sha256=fingerprint(PARENT / 'protocol.json'), manifest_sha256=fingerprint(manifest_path),
        selection='16 equally spaced by source_frame among same-source dual TRAIN excluding 8 replay scenes; locked before predictions. No development/test selection.',
        scope='Unreplayed means not in recent 800-step adaptation, NOT unseen by original pretraining. All 24 are synthetic TRAIN records; no new test split.',
        matching='Same production diagnostic GT-box/objectness assignment; no keypoint matching or teacher quality gating. Not end-to-end multi-animal association.',
        candidate_gate='Coverage experiment supported only if mixed nonreplayed Core23 NME > 1.15x replayed AND (>1.05x original nonreplayed NME OR nonreplayed detection recall lower than original by >.05). Evidence gate, not proof of causation.',
        fixed_visual_indices=[4, 16], optimizer_steps=0)
    write_json(OUT / 'protocol.json', protocol)
    torch.set_num_threads(2); torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    encoder = ROOT / 'vggt-omega/vggt_omega_1b_512.pt'
    assert fingerprint(encoder) == spec['encoder_sha256']
    spatial = VGGTOmega(enable_camera=False, enable_depth=True, enable_multi_instance_pose=True, num_keypoints=23)
    weights = torch.load(encoder, weights_only=True, mmap=True)
    loaded = spatial.load_state_dict({k: v for k, v in weights.items() if k.startswith(('aggregator.', 'dense_head.'))}, strict=False)
    assert not loaded.unexpected_keys and all(k.startswith('instance_pose_head.') for k in loaded.missing_keys)
    del weights
    model = DepthPoseModel.from_spatial_model(spatial).cuda().eval(); del spatial
    states = {tag: torch.load(p, weights_only=True, map_location='cpu')['model'] for tag, p in paths.items()}
    hashes = {tag: state_hash(state) for tag, state in states.items()}
    rows = []; metrics = []; target_records = []; projection = []
    for idx, entry in enumerate(cohort):
        r = entry['record']; batch = oriented_batch(r, num_views=2)
        target = instance_targets_from_batch(batch); images = batch['images'].cuda()
        xyz = target['keypoint_world_xyz'][0]
        homo = torch.cat((xyz, torch.ones_like(xyz[..., :1])), dim=-1)
        uvw = torch.einsum('vij,mkj->vmki', target['intrinsics'][0] @ target['extrinsics'][0, :, :3], homo)
        expected = target['keypoint_xy'][0] * torch.tensor(r['image_hw'][::-1])
        projection.extend((uvw[..., :2] / uvw[..., 2:3] - expected).norm(dim=-1)[target['keypoint_valid'][0]].tolist())
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            cached, patch_start = model.aggregator(images)
        for tag, state in states.items():
            restore(model, state)
            with torch.no_grad():
                pred = {k: v.cpu() for k, v in model.forward_cached(images, cached, patch_start, coupling='off').items()}
            metric, assignment = evaluate_scene(pred, batch, target)
            saved = dict(record=r, oracle_assignment=assignment, metrics=metric,
                predictions={k: v for k, v in pred.items() if k.startswith('instance_') and 'mask' not in k},
                targets={k: v for k, v in target.items() if k != 'masks'})
            folder = OUT / tag; folder.mkdir(exist_ok=True)
            torch.save(saved, folder / f'prediction_{idx:03d}.pt')
            metrics.append(dict(checkpoint=tag, record=idx, cohort=entry['cohort'], metrics=metric))
            for row in matched_views(saved):
                valid = row['valid']; body = valid.clone(); body[7:14] = False; tail = valid & ~body
                rows.append(dict(checkpoint=tag, record=idx, cohort=entry['cohort'], scene=r['scene'],
                    **{k: row[k] for k in ('view', 'camera', 'role', 'query', 'instance', 'score', 'iou', 'found')},
                    body_errors=row['errors'][body].tolist(), tail_errors=row['errors'][tail].tolist(), core23_errors=row['errors'][valid].tolist()))
            assert state_hash(tensor_state(model)) == hashes[tag]
            del pred
        target_records.append(dict(record=r, targets=target))
        del cached, images, batch
        if (idx + 1) % 8 == 0:
            write_json(OUT / 'status.json', dict(state='auditing', records=idx + 1, total=24))
            print(f'Checked {idx + 1}/24', flush=True)
    assert max(projection) < .01
    coverage = []
    for idx in range(8, 24):
        distances = [pose_distance(target_records[idx], t) for t in target_records[:8]]
        nearest = int(np.argmin(distances)); r = cohort[idx]['record']
        coverage.append(dict(record=idx, frame=r['frame'], source_frame=r['source_frame'],
            nearest_shape_replay_frame=replay[nearest]['frame'], nearest_shape_distance=min(distances),
            closest_source_frame_gap=min(abs(r['source_frame'] - t['source_frame']) for t in replay)))
    summary = summarize_views(rows)
    lookup = {(r['checkpoint'], r['cohort'], r['role']): r for r in summary}
    seen = lookup['mixed', 'replayed', 'all']; unseen = lookup['mixed', 'nonreplayed', 'all']; original = lookup['original', 'nonreplayed', 'all']
    gate = unseen['core23_nme'] > 1.15 * seen['core23_nme'] and (unseen['core23_nme'] > 1.05 * original['core23_nme'] or unseen['recall'] < original['recall'] - .05)
    for p, h in protocol['code_sha256'].items(): assert fingerprint(ROOT / p) == h
    assert all(scene_digest(ROOT / p) == h for p, h in sources.items())
    assert all(fingerprint(p) == protocol['checkpoints'][tag]['sha256'] for tag, p in paths.items())
    assert fingerprint(manifest_path) == protocol['manifest_sha256']
    report = dict(passed=True, coverage_experiment_supported=bool(gate), encoder_forwards=24, head_forwards=48,
        optimizer_steps=0, source_code_weights_unchanged=True, gt_projection_max_px=max(projection),
        source_groups=dict(Counter(e['record']['source_group'] for e in cohort)),
        action_labels='Manifest provides source_group/source_frame, not verified per-frame action classes.',
        mixed_nonreplay_vs_replay_nme_ratio=unseen['core23_nme'] / seen['core23_nme'],
        mixed_nonreplay_vs_original_nme_ratio=unseen['core23_nme'] / original['core23_nme'],
        summary=summary, elapsed_s=time.monotonic() - start)
    for name, data in [('instance_views', rows), ('metrics', metrics), ('pose_coverage', coverage), ('report', report)]:
        write_json(OUT / (name + '.json'), data)
    write_json(OUT / 'status.json', dict(state='completed'))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'output/cylinder_training.lock').open('a+') as glock, (OUT / 'running.lock').open('a+') as lock:
        fcntl.flock(glock, fcntl.LOCK_EX | fcntl.LOCK_NB); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (OUT / 'protocol.json').exists(): raise RuntimeError('Do not overwrite existing audit')
        try:
            run()
        except BaseException as exc:
            write_json(OUT / 'status.json', dict(state='failed', error=repr(exc), optimizer_steps=0))
            raise
