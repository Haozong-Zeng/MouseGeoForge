"""Portable V5 rendering design and v3-compatible file validation; no model calls."""
from __future__ import annotations
import collections
import hashlib
import json
import math
from pathlib import Path
import random

SCHEMA = 'mousegeoforge-v5-synthetic-v1'
OUT_REL = 'output/bottom_geometry_v5_complex_20261024'
REVIEW_REL = 'output/v5_complex_background_review_20261024'
SOURCE_DESIGN = 'output/bottom_geometry_v3_local12h_20261018/render_design.json'
REVIEW_BLEND = REVIEW_REL + '/MouseGeoForge_V5_complex_background_REVIEW.blend'
REVIEW_COLLECTION = 'V5 REVIEW | shared asymmetric laboratory'
REVIEW_WORLD = 'V5_REVIEW_shared world'
PREFIX = 'V5_REVIEW_'
PIN_FILES = [
    'tools/v5_render_contract_20261024.py',
    'tools/run_v5_complex_render_20261024.py',
    'BlenderScripts/render_v5_complex_20261024.py',
    'BlenderScripts/export_visible_surface_truth_20261007.py',
    'BlenderScripts/audit_visible_surface_geometry_20261007.py',
    'BlenderScripts/build_v5_complex_background_review_20261024.py',
]


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def seed_for(*values):
    return int(hashlib.sha256(':'.join(map(str, ('v5-20261024', *values))).encode()).hexdigest()[:15], 16)


def make_design(old):
    assert len(old['rigs']) == 68 and len(old['sources']) == 2
    assert collections.Counter(r['split'] for r in old['rigs']) == dict(train=48, val=12, sealed=8)
    envs = [dict(id=f'{split}_env{i:02d}', split=split, index=i, seed=seed_for('environment', split, i))
            for split, count in [('train', 24), ('dev', 6), ('sealed', 6)] for i in range(count)]
    pairs = []
    for rig in old['rigs']:
        index = int(rig['id'][-2:])
        if rig['split'] == 'train':
            assignments = [(f'train_env{index % 24:02d}', 'train'), (f'train_env{(index + 12) % 24:02d}', 'train')]
        elif rig['split'] == 'val':
            assignments = [(f'dev_env{index % 6:02d}', 'dev_primary'), (f'train_env{index % 24:02d}', 'dev_attribution')]
        else:
            assignments = [(f'sealed_env{index % 6:02d}', 'sealed')]
        for env_id, condition in assignments:
            pairs.append(dict(rig=rig['id'], environment=env_id, condition=condition,
                              split=rig['split'], shard=0 if rig['split'] == 'train' else 1))
    scenes = []
    for pair in pairs:
        for source in old['sources']:
            assert source['id'] in ('climb', 'moseq') and source['admitted_split'] == 'train'
            assert source['frames'] == [49, 97, 145]
            for frame in source['frames']:
                scenes.append(dict(**pair, source=source['id'], frame=frame,
                                   lighting_seed=seed_for('lighting', pair['environment'], pair['rig'], source['id'], frame)))
    assert len(scenes) == 768
    assert collections.Counter(x['split'] for x in scenes) == dict(train=576, val=144, sealed=48)
    return dict(schema=SCHEMA, sources=old['sources'], rigs=old['rigs'], environments=envs,
                assignments=pairs, scenes=scenes, expected_scenes=768, expected_views=2304,
                resolution=384, cycles_samples=32, color_depth='8',
                visibility=dict(prescreen_side=96, prescreen_min_body=16, final_min_body=100,
                                max_resamples=32, projection_tolerance_px=.003),
                truth_policy='Primary-ray camera-Z opaque skin/mesh truth, pixel-centre K; transparent glass and hair excluded. Opaque new cables are meshes. Not an RGB fur silhouette.',
                rejected_policy='Whole environment candidate rejected if any assigned rig/source/frame/view fails the same geometric rule; never drop hard scenes or change rigs.',
                source_design=SOURCE_DESIGN, template_blend=REVIEW_BLEND,
                template_collection=REVIEW_COLLECTION, template_world=REVIEW_WORLD,
                layout_rule='render_shard{0|1}/{rig}/{environment}/{motion}/frame_{frame:06d}/',
                sealed_policy='Render and geometric QA only; no model inference or scoring.',
                user_authorization='2026-10-24: 可以，开始渲染，注意符合合成数据的数据格式和文件夹安排')


def environment_parameters(env, attempt):
    """Camera-independent candidates. A shared layout is never specialized per rig."""
    rng = random.Random(seed_for(env['seed'], attempt))
    approved = env['id'] == 'train_env00' and attempt == 0
    def off(a, b):
        return [0, 0, 0] if approved else [rng.uniform(-a, a), rng.uniform(-b, b), 0]
    return dict(id=env['id'], split=env['split'], seed=env['seed'], attempt=attempt,
                rotation_z=0 if approved else rng.uniform(-math.pi, math.pi),
                ceiling_offset=off(85, 80), storage_offset=off(40, 25),
                post_offsets=[off(20, 20) for _ in range(3)], controller_offset=off(18, 20),
                wall_panel_offsets=[off(30, 0)[:2] + [0 if approved else rng.uniform(-45, 45)] for _ in range(4)],
                material_gain=1 if approved else rng.uniform(.86, 1.12),
                reference_approved_layout=approved)


def lighting_parameters(row):
    rng = random.Random(row['lighting_seed'])
    reference = row['environment'] == 'train_env00' and row['rig'] == 'train_rig00' and row['source'] == 'climb' and row['frame'] == 49
    base = [('warm key', [-260, -170, 775], 36000000, 12, [1, .87, .72]),
            ('cool soft fill', [320, 110, 660], 24000000, 16, [.79, .89, 1]),
            ('rear bounce', [-60, 410, 510], 16000000, 14, [1, .95, .84])]
    lights = []
    for name, position, energy, radius, color in base:
        kelvin = rng.uniform(3500, 6500)
        # Explicit approximate blackbody RGB. This parameter is not a camera label.
        t = kelvin / 100
        r = 255 if t <= 66 else 329.698727446 * (t - 60) ** -.1332047592
        g = 99.4708025861 * math.log(t) - 161.1195681661 if t <= 66 else 288.1221695283 * (t - 60) ** -.0755148492
        b = 255 if t >= 66 else 0 if t <= 19 else 138.5177312231 * math.log(t - 10) - 305.0447927307
        tint = [max(0, min(255, x)) / 255 for x in (r, g, b)]
        lights.append(dict(name=PREFIX + name, type='SPOT',
                           position=position if reference else [position[k] + rng.uniform(-20, 20) for k in range(3)],
                           target=[0, 0, {'warm key': 340, 'cool soft fill': 330, 'rear bounce': 350}[name]],
                           energy=energy if reference else energy * rng.uniform(.8, 1.25),
                           soft_radius=radius if reference else radius * rng.uniform(.8, 1.2),
                           color=color if reference else tint, temperature_K=None if reference else kelvin,
                           cone_degrees=125, blend=.75))
    return dict(seed=row['lighting_seed'], shared_across_all_views=True,
                world_strength=.18 if reference else rng.uniform(.14, .24), lights=lights,
                reference_approved_lighting=reference)


def scene_path(out_rel, row):
    return f"{out_rel}/render_shard{row['shard']}/{row['rig']}/{row['environment']}/{row['source']}/frame_{row['frame']:06d}"


def validate_record(root, record, expected_resolution=384):
    """Read all payloads and independently check the camera-Z / pixel-centre contract."""
    import numpy as np
    from PIL import Image
    root = Path(root).resolve()
    assert record['kind'] == 'synthetic' and record['admitted_motion_split'] == 'train'
    assert record['split'] in ('train', 'val', 'sealed')
    assert record['motion_id'] in ('climb', 'moseq')
    assert record['rig'] in record['scene'] and record['environment_id'] in record['scene']
    assert len(record['views']) == 3
    for i, view in enumerate(record['views']):
        assert view['id'] == i and view['physical_role'] == ('bottom' if i == 0 else 'side')
        assert view['passed'] and view['foreground_pixels'] >= 100
        assert view['projection_closure_max_px'] < .003
        files = [view['image'], view['truth']]
        assert [Path(x).name for x in files] == [f'rgb_{i}.png', f'truth_{i}.npz']
        for rel in files:
            assert not Path(rel).is_absolute() and '\\' not in rel
            path = (root / rel).resolve()
            assert path.is_relative_to(root) and path.parent == (root / record['scene']).resolve()
            assert sha(path) == view['file_sha256'][Path(rel).name]
        with Image.open(root / view['image']) as im:
            assert im.size == (expected_resolution, expected_resolution) and im.mode == 'RGB'
        with np.load(root / view['truth'], allow_pickle=False) as truth:
            assert set(truth.files) == {'depth', 'body', 'valid', 'K', 'E'}
            z, body, valid = truth['depth'], truth['body'], truth['valid']
            assert z.shape == body.shape == valid.shape == (expected_resolution, expected_resolution)
            assert body.dtype == valid.dtype == np.bool_
            assert np.isfinite(z).all() and (z[valid] > 0).all() and (z[~valid] == 0).all()
            assert not (body & ~valid).any() and int(body.sum()) == view['foreground_pixels']
            np.testing.assert_allclose(truth['K'], view['K'], atol=1e-8)
            np.testing.assert_allclose(truth['E'], view['E'], atol=1e-8)
            k, e = truth['K'], truth['E']
            assert k.shape == (3, 3) and e.shape == (4, 4)
            np.testing.assert_allclose(e[:3, :3] @ e[:3, :3].T, np.eye(3), atol=2e-6)
            np.testing.assert_allclose(e[3], [0, 0, 0, 1], atol=1e-8)
            assert np.linalg.det(e[:3, :3]) > .999
            # Pixel-centre indices are integer coordinates; no extra half pixel here.
            yy, xx = np.where(body)
            uv = np.stack((xx, yy, np.ones_like(xx)), -1)
            camera_xyz = (uv @ np.linalg.inv(k).T) * z[body, None]
            world = (camera_xyz - e[:3, 3]) @ e[:3, :3]
            p = (world @ e[:3, :3].T + e[:3, 3]) @ k.T
            np.testing.assert_allclose(p[:, :2] / p[:, 2:], uv[:, :2], atol=.003)
    return True
