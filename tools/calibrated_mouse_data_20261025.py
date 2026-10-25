"""Admit only canonical V5 TRAIN/DEV plus already-admitted real manual masks."""
from __future__ import annotations
import argparse
from collections import Counter
import contextlib
import fcntl
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
import torch
from mousegeoforge.semseg_training import digest, load_items, make_encoder, write_json
from v5_render_contract_20261024 import validate_record

OUT = ROOT / 'output/calibrated_mouse_10h_20261025'


@contextlib.contextmanager
def gpu_lock():
    with (ROOT / 'output/teacher_temporal_cycle_20261001/gpu_job.lock').open('a+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def points(z, mask, k, e, stride=1):
    y, x = np.where(mask)
    x, y = x[::stride], y[::stride]
    uv = np.stack([x, y, np.ones_like(x)], -1)
    camera = (uv @ np.linalg.inv(k).T) * z[y, x, None]
    return (camera - e[:3, 3]) @ e[:3, :3]


def audit():
    began = time.monotonic()
    if (OUT / 'data_manifest.json').exists():
        raise FileExistsError('Audit outputs preserved; create a new attempt for changed inputs')
    scenes, semseg = [], []
    real = load_items(ROOT / 'output/vggt_mouse_segmentation_16h_20261020/items.json', ROOT, verify=False)
    real = [r for r in real if r['domain'] == 'real']
    for row in real:
        for key in ('image', 'label'):
            assert digest(ROOT / row[key]) == row[key + '_sha256']
        row = {**row, 'label_policy': 'manual_visible_mouse'}
        semseg.append(row)
    minimum, maximum = np.full(3, np.inf), np.full(3, -np.inf)
    reprojection, views = 0., 0
    for name in sorted((ROOT / 'dataset/BlenderPyMouse').iterdir()):
        if not name.is_dir() or not name.name.endswith(('ComplexBackgroundV5-Train', 'ComplexBackgroundV5-Dev')):
            continue
        assert 'Sealed' not in name.name
        for folder in sorted(name.glob('frame_*')):
            meta_path = folder / 'scene_meta.json'
            meta = json.loads(meta_path.read_text())
            split = 'train' if name.name.endswith('-Train') else 'dev'
            assert meta['split'] == ('train' if split == 'train' else 'val')
            assert not meta['rig'].startswith('sealed')
            source_record = ROOT / meta['source_render_scene'] / 'record.json'
            record = json.loads(source_record.read_text())
            validate_record(ROOT, record)
            row = dict(id=name.name + '/' + folder.name, split=split, rig=meta['rig'],
                       source=meta['motion_id'], pose_frame=meta['source_frame'],
                       environment=meta['environment_id'], condition=meta['dev_condition'],
                       metadata=meta_path.relative_to(ROOT).as_posix(),
                       metadata_sha256=digest(meta_path), views=[])
            for i, view in enumerate(meta['views']):
                assert view['depth_units'] == 'millimeter' and view['depth_semantics'] == 'camera_z'
                image, truth = folder / view['image_lossless'], folder / 'provenance' / f'truth_{i}.npz'
                assert digest(image) == record['views'][i]['file_sha256'][image.name]
                assert digest(truth) == record['views'][i]['file_sha256'][truth.name]
                with np.load(truth, allow_pickle=False) as payload:
                    z, body, k, e = [payload[key] for key in ['depth', 'body', 'K', 'E']]
                    world = points(z, body, k, e, stride=max(1, int(body.sum()) // 2048))
                    if split == 'train':
                        minimum = np.minimum(minimum, world.min(0)); maximum = np.maximum(maximum, world.max(0))
                    camera = world @ e[:3, :3].T + e[:3, 3]
                    uv = camera @ k.T; uv = uv[:, :2] / uv[:, 2:]
                    reprojection = max(reprojection, float(np.abs(uv - np.round(uv)).max()))
                item = dict(image=image.relative_to(ROOT).as_posix(), truth=truth.relative_to(ROOT).as_posix(),
                            image_sha256=digest(image), truth_sha256=digest(truth), bottom=i == 0)
                row['views'].append(item)
                semseg.append(dict(id=digest(image)[:20] + f'_v5_{i}', image=item['image'], label=item['truth'],
                    image_sha256=item['image_sha256'], label_sha256=item['truth_sha256'],
                    split=split, domain='synthetic', bottom=i == 0, group=meta['motion_id'],
                    cohort='v5_clean_skin', rig=meta['rig'], admitted_motion_split='train',
                    label_policy='visible_skin_surrogate_excludes_hair_and_glass'))
                views += 1
            scenes.append(row)
            if len(scenes) % 48 == 0:
                print('AUDIT', len(scenes), views, flush=True)
    assert Counter(s['split'] for s in scenes) == dict(train=576, dev=144)
    assert reprojection < .003
    assert len({(s['source'], s['pose_frame']) for s in scenes}) == 6
    # Bounds learned from TRAIN only and then frozen, never per-image/test GT ranges.
    bounds = [np.floor(minimum - 50).tolist(), np.ceil(maximum + 50).tolist()]
    write_json(OUT / 'data_manifest.json', dict(schema='calibrated-mouse-scenes-v1', scenes=scenes,
        capture_aabb_mm=bounds, aabb_policy='TRAIN sparse skin extent plus50mm, frozen for every scene',
        units='millimeter', geometry='OpenCV world-to-camera; integer pixel centres; camera-Z',
        independent_pose_count=6, sealed_opened=False))
    write_json(OUT / 'semseg_items.json', dict(schema='mouse-semseg-v1', items=semseg))
    load_items(OUT / 'semseg_items.json', ROOT, verify=True)
    write_json(OUT / 'data_audit.json', dict(passed=True, scenes=len(scenes), views=views,
        splits=dict(Counter(s['split'] for s in scenes)), own_reprojection_max_px=reprojection,
        capture_aabb_mm=bounds, semseg_counts=dict(Counter(r['split'] + '/' + r['domain'] for r in semseg)),
        semseg_policy='historically admitted real manual masks plus canonical V5 clean skin; old synthetic masks excluded',
        pose_generalization_claim=False, independent_real_depth_gt=False,
        manifest_sha256=digest(OUT / 'data_manifest.json'), seconds=time.monotonic()-began))


def cache():
    torch.set_num_threads(4)
    manifest = json.loads((OUT / 'data_manifest.json').read_text())
    target = OUT / 'depth_features'; target.mkdir(exist_ok=True)
    encoder = make_encoder(ROOT / 'vggt-omega/vggt_omega_1b_512.pt').cuda().eval().requires_grad_(False)
    mean = torch.tensor([.485, .456, .406], device='cuda')[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device='cuda')[None, :, None, None]
    began, receipt = time.monotonic(), []
    with gpu_lock(), torch.inference_mode():
        for index, row in enumerate(manifest['scenes']):
            path = target / f'{index:06d}.pt'
            if path.exists():
                raise FileExistsError('Existing cache attempt is immutable')
            images, truths = [], []
            for view in row['views']:
                assert digest(ROOT / view['image']) == view['image_sha256']
                assert digest(ROOT / view['truth']) == view['truth_sha256']
                images.append(torch.from_numpy(np.array(Image.open(ROOT / view['image']))).permute(2, 0, 1).float()/255)
                with np.load(ROOT / view['truth']) as data:
                    truths.append({k: torch.from_numpy(data[k].copy()) for k in ['depth', 'body', 'K', 'E']})
            images = torch.stack(images).cuda()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                features = encoder.get_intermediate_layers((images-mean)/std, n=[23], reshape=True, norm=True)[0]
            payload = dict(features=features.cpu().half(), rgb_side=images.shape[-1],
                           scene=row, depth=torch.stack([x['depth'] for x in truths]),
                           body=torch.stack([x['body'] for x in truths]),
                           K=torch.stack([x['K'] for x in truths]), E=torch.stack([x['E'] for x in truths]))
            torch.save(payload, path)
            receipt.append(dict(path=path.relative_to(ROOT).as_posix(), sha256=digest(path), id=row['id']))
            if (index+1) % 12 == 0:
                write_json(OUT / 'cache_progress.json', dict(scenes=index+1, seconds=time.monotonic()-began))
                print('CACHE', index+1, round(time.monotonic()-began, 1), flush=True)
    write_json(OUT / 'cache_receipt.json', dict(passed=True, files=receipt,
        manifest_sha256=digest(OUT / 'data_manifest.json'), encoder_sha256=digest(ROOT / 'vggt-omega/vggt_omega_1b_512.pt'),
        layer=23, shape=[3, 1024, 24, 24], dtype='float16 storage; BF16 extraction',
        seconds=time.monotonic()-began))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=['audit', 'cache'])
    args = parser.parse_args()
    audit() if args.command == 'audit' else cache()
