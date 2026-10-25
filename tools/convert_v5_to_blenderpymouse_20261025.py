"""Publish completed V5 renders as standard, self-contained BlenderPyMouse datasets.

Two motion families, each split into Train/Dev/Sealed so root-level frame folders
match existing datasets and no protected scenes enter a training dataset root.
RGB/depth/camera geometry is preserved; Core23 uses read-only source replay.
"""
from __future__ import annotations
import argparse
import collections
import datetime as dt
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from PIL import Image

import v5_render_contract_20261024 as render_contract
from audit_blenderpymouse_keypoint_annotations import audit_dataset
from build_blenderpymouse_metadata import scene_is_complete

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/v5_blenderpymouse_conversion_20261025'
BASE = ROOT / 'dataset/BlenderPyMouse'
IDS = [5, 6, 22]
KP_FORMAT = 'freemoving_core23_surface_v3'
VERSION = 'pymousestudio_manual_surface_core23_v3'
CAMERAS = ['V5_Bottom', 'V5_Side1', 'V5_Side2']
sha, read, write = render_contract.sha, render_contract.read, render_contract.write


def save_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as f:
        np.savez_compressed(f, **arrays)


def write_pfm(path, depth):
    depth = np.asarray(depth, dtype='<f4')
    assert depth.ndim == 2 and np.isfinite(depth).all() and (depth >= 0).all()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as f:
        f.write(f'Pf\n{depth.shape[1]} {depth.shape[0]}\n-1.0\n'.encode('ascii'))
        np.flipud(depth).tofile(f)


def read_pfm(path):
    with path.open('rb') as f:
        assert f.readline().strip() == b'Pf'
        width, height = map(int, f.readline().split())
        scale = float(f.readline())
        assert scale == -1
        data = np.fromfile(f, dtype='<f4')
    assert data.size == width * height
    return np.flipud(data.reshape(height, width))


def write_cam(path, e, k, depth):
    positive = depth[depth > 0]
    low, high = np.percentile(positive, [.1, 99.9])
    high = max(high, low + 1)
    lines = ['extrinsic', *[' '.join(f'{v:.17g}' for v in row) for row in e], '', 'intrinsic',
             *[' '.join(f'{v:.17g}' for v in row) for row in k], '', f'{low:.17g} {(high-low)/191:.17g} 192 {high:.17g}', '']
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines), encoding='utf-8')


def read_cam(path):
    lines = path.read_text().splitlines()
    assert lines[0] == 'extrinsic' and lines[6] == 'intrinsic'
    return np.array([[float(x) for x in line.split()] for line in lines[1:5]]), np.array([[float(x) for x in line.split()] for line in lines[7:10]])


def write_pairs(path, n=3):
    lines = [str(n)]
    for i in range(n):
        sources = [j for j in range(n) if j != i]
        lines.extend([str(i), str(len(sources)) + ' ' + ' '.join(f'{j} {1/(rank+1):.6f}' for rank, j in enumerate(sources))])
    path.write_text('\n'.join(lines) + '\n')


def project(world, e, k):
    camera = world @ e[:3, :3].T + e[:3, 3]
    h = camera @ k.T
    assert np.isfinite(h).all() and (np.abs(h[..., 2]) > 1e-8).all()
    return camera, h[..., :2] / h[..., 2:]


def support_visibility(camera, uv, depth, mask):
    visible = np.zeros(camera.shape[:-1], bool)
    residual = np.full(camera.shape[:-1], np.nan)
    height, width = depth.shape
    for idx in np.ndindex(visible.shape):
        u, v = uv[idx]; z = camera[idx][2]
        if z <= 0 or not np.isfinite([u, v, z]).all():
            continue
        x, y = int(round(u)), int(round(v))
        if not (0 <= x < width and 0 <= y < height):
            continue
        patch = np.s_[max(0, y-1):min(height, y+2), max(0, x-1):min(width, x+2)]
        values = depth[patch][mask[patch] & (depth[patch] > 0)]
        if values.size:
            error = float(np.abs(values.astype(float) - z).min())
            residual[idx] = error
            visible[idx] = error <= max(.05, .01 * abs(z))
    return visible, residual


def label_view(world, surface, support, e, k, depth, mask, bare_visible, self_blocked):
    camera, uv = project(world, e, k)
    surface_camera, surface_uv = project(surface, e, k)
    support_camera, support_uv = project(support, e, k)
    votes, residuals = support_visibility(support_camera, support_uv, depth, mask)
    h, w = depth.shape
    inside = (camera[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    visible = votes.any(axis=1) & inside
    known = inside.copy()
    for index in IDS:
        visible[index] = bool(bare_visible[index] and inside[index])
        if bare_visible[index] and inside[index]:
            # A body-only ray cannot certify visibility through apparatus.
            # Keep conflicts unknown rather than relabeling them as self occlusion.
            anchor_votes, _ = support_visibility(camera[index:index+1], uv[index:index+1], depth, mask)
            if not anchor_votes[0]:
                known[index] = visible[index] = False
    cause = np.full(23, -1, np.int8)
    cause[visible & known] = 0
    cause[inside & known & ~visible & self_blocked.all(axis=-1)] = 1
    for index in IDS:
        if inside[index] and known[index] and not bare_visible[index]:
            cause[index] = 1
    return dict(camera=camera, uv=uv, surface_camera=surface_camera, surface_uv=surface_uv,
                in_frame=inside, visible=visible, visibility_valid=known,
                xy_label_valid=np.isfinite(uv).all(axis=-1) & (camera[:, 2] > 0),
                votes=votes.sum(axis=-1).astype(np.int16), residuals=residuals, cause=cause)


def convert_scene(dataset, record, geometry):
    folder = dataset / record['export_scene']
    folder.mkdir(parents=True, exist_ok=False)
    for name in ('blended_images', 'masks', 'rendered_depth_maps', 'cams', 'annotations', 'provenance'):
        (folder / name).mkdir()
    assert render_contract.validate_record(ROOT, record)
    world = geometry['joint_world_xyz'].astype(float)
    surface = geometry['surface_world_xyz'].astype(float)
    support = geometry['support_world_xyz'].astype(float)
    with np.load(OUT / 'body_ray_visibility' / record['export_dataset'] / (record['export_scene'] + '.npz'), allow_pickle=False) as ray:
        bare, blocked = ray['anchor_body_visible'], ray['self_support_blocked']
    views, arrays = [], []
    for i, old in enumerate(record['views']):
        stem = f'{i:08d}'
        e, k = np.array(old['E']), np.array(old['K'])
        with np.load(ROOT / old['truth'], allow_pickle=False) as truth:
            depth, mask = truth['depth'], truth['body']
            np.testing.assert_array_equal(truth['K'], k)
            np.testing.assert_array_equal(truth['E'], e)
        rgb = np.asarray(Image.open(ROOT / old['image']).convert('RGB'))
        Image.fromarray(rgb).save(folder / 'blended_images' / f'{stem}.jpg', quality=95, subsampling=0)
        Image.fromarray(np.where(mask[..., None], rgb, 0).astype(np.uint8)).save(folder / 'blended_images' / f'{stem}_masked.jpg', quality=95, subsampling=0)
        Image.fromarray(mask.astype(np.uint8) * 255).save(folder / 'masks' / f'{stem}.png')
        write_pfm(folder / 'rendered_depth_maps' / f'{stem}.pfm', depth)
        positive = depth[depth > 0]
        low, high = np.percentile(positive, [.5, 99.5])
        preview = np.where(depth > 0, np.clip((high-depth) / max(high-low, 1e-6), 0, 1) * 255, 0).astype(np.uint8)
        Image.fromarray(preview).save(folder / 'rendered_depth_maps' / f'{stem}_preview.png')
        write_cam(folder / 'cams' / f'{stem}_cam.txt', e, k, depth)
        for key in ('image', 'truth'):
            shutil.copyfile(ROOT / old[key], folder / 'provenance' / Path(old[key]).name)
        labels = label_view(world, surface, support, e, k, depth, mask, bare[i], blocked[i])
        points = []
        for j in range(23):
            residual = labels['residuals'][j]
            finite = residual[np.isfinite(residual)]
            point = dict(id=j, name=str(geometry['names'][j]), semantic_name=str(geometry['semantic_names'][j]),
                group=str(geometry['groups'][j]), bone=str(geometry['bones'][j]), endpoint=str(geometry['endpoints'][j]),
                coordinate_kind=str(geometry['coordinate_kind'][j]), definition_version=VERSION, defined_by_dataset=True,
                joint_world_xyz=world[j].tolist(), surface_world_xyz=surface[j].tolist(),
                camera_xyz=labels['camera'][j].tolist(), surface_camera_xyz=labels['surface_camera'][j].tolist(),
                uv=labels['uv'][j].tolist(), x=float(labels['uv'][j, 0]), y=float(labels['uv'][j, 1]),
                camera_depth=float(labels['camera'][j, 2]), surface_depth=float(labels['surface_camera'][j, 2]),
                in_frame=bool(labels['in_frame'][j]), surface_visible=bool(labels['visible'][j]), visible=bool(labels['visible'][j]),
                xy_label_valid=bool(labels['xy_label_valid'][j]), visibility_valid=bool(labels['visibility_valid'][j]),
                world_valid=True, surface_valid=True, candidate_only=False,
                occluded_but_annotated=bool(labels['in_frame'][j] and labels['visibility_valid'][j] and not labels['visible'][j]),
                confidence=float(labels['xy_label_valid'][j]), support_visible_count=int(labels['votes'][j]), support_vertex_count=16,
                support_min_depth_residual=float(finite.min()) if finite.size else None)
            points.append(point)
        view = dict(frame=record['export_index'], source_frame=record['frame'], view_id=i, camera=CAMERAS[i], physical_role=old['physical_role'],
            image=f'blended_images/{stem}.jpg', masked_image=f'blended_images/{stem}_masked.jpg',
            image_lossless=f'provenance/rgb_{i}.png', mask=f'masks/{stem}.png', depth_pfm=f'rendered_depth_maps/{stem}.pfm',
            depth_preview=f'rendered_depth_maps/{stem}_preview.png', cam=f'cams/{stem}_cam.txt',
            depth_available=True, depth_units='millimeter', depth_semantics='camera_z',
            depth_excludes_transparent_glass=True, depth_excludes_hair_curves=True,
            mask_semantics='visible primary-ray skin; no hair; not amodal; no dilation',
            camera_extrinsic_source='unchanged_v5_render_record', intrinsic=k.tolist(), extrinsic=e.tolist(),
            pixel_center_convention='integer_array_indices', keypoint_format=KP_FORMAT, definition_version=VERSION,
            keypoints=points, split=record['split'], rig=record['rig'], environment_id=record['environment_id'],
            dev_condition=record['dev_condition'])
        write(folder / 'annotations' / f'{stem}.json', view)
        views.append(view)
        arrays.append(dict(labels, intrinsic=k, extrinsic=e))
    write_pairs(folder / 'cams/pair.txt')
    names = geometry['names']
    source = dict(frame=record['export_index'], source_frame=record['frame'], rig=record['rig'], split=record['split'],
                  environment_id=record['environment_id'], environment_split=record['environment_split'],
                  dev_condition=record['dev_condition'], motion_id=record['motion_id'], motion_source_id=record['group'],
                  source_blend=record['source_blend'], source_blend_sha256=record['source_sha256'],
                  source_render_scene=record['scene'], temporal_eligible=False)
    write(folder / 'scene_meta.json', dict(source, keypoint_format=KP_FORMAT, keypoint_names=names.tolist(),
          semantic_names=geometry['semantic_names'].tolist(), definition_version=VERSION, views=views))
    write(folder / 'keypoints.json', dict(source, format=KP_FORMAT, names=names.tolist(), semantic_names=geometry['semantic_names'].tolist(),
          groups=geometry['groups'].tolist(), bones=geometry['bones'].tolist(), endpoints=geometry['endpoints'].tolist(),
          coordinate_system='Blender source millimetre coordinates; OpenCV +Z forward',
          joint_coordinate_rule='Fixed REST barycentric anchors at slots06/07/23; pose bone heads at other slots.',
          joint_world_xyz=world.tolist(), surface_world_xyz=surface.tolist(), support_vertex_ids=geometry['support_vertex_ids'].tolist(),
          definition_version=VERSION, views=[dict(view_id=v['view_id'], camera=v['camera'], keypoints=v['keypoints']) for v in views]))
    flags = dict(xy_label_valid=np.stack([a['xy_label_valid'] for a in arrays]),
                 visibility_valid=np.stack([a['visibility_valid'] for a in arrays]),
                 world_valid=np.ones(23, bool), surface_valid=np.ones(23, bool))
    save_npz(folder / 'keypoints.npz',
        format=np.asarray(KP_FORMAT), frame=np.asarray(record['export_index'], np.int32), source_frame=np.asarray(record['frame'], np.int32),
        **{key: geometry[key] for key in ('names', 'semantic_names', 'groups', 'bones', 'endpoints', 'coordinate_kind', 'definition_version',
                                         'joint_world_xyz', 'surface_world_xyz', 'support_vertex_ids', 'support_world_xyz')},
        view_ids=np.arange(3, dtype=np.int32), camera_names=np.asarray(CAMERAS),
        intrinsics=np.stack([a['intrinsic'] for a in arrays]), extrinsics=np.stack([a['extrinsic'] for a in arrays]),
        joint_camera_xyz=np.stack([a['camera'] for a in arrays]), uv=np.stack([a['uv'] for a in arrays]),
        in_frame=np.stack([a['in_frame'] for a in arrays]), surface_visible=np.stack([a['visible'] for a in arrays]),
        support_visible_count=np.stack([a['votes'] for a in arrays]), depth_available=np.ones(3, bool), **flags,
        source_annotation_sha256=np.asarray(sha(OUT / 'source_geometry' / record['motion_id'] / f"frame_{record['frame']:06d}.npz")),
        surface_anchor_source_sha256=np.asarray(record['source_sha256']))
    save_npz(folder / 'keypoint_validity.npz', schema_version=np.asarray(1), **flags)
    causes = np.stack([a['cause'] for a in arrays])
    save_npz(folder / 'keypoint_occlusion.npz', schema_version=np.asarray(1), names=names, view_ids=np.arange(3, dtype=np.int32),
        annotation_sha256=np.asarray(sha(folder / 'keypoints.npz')), definition_version=np.asarray(VERSION),
        surface_anchor_source_sha256=np.asarray(record['source_sha256']), occlusion_cause=causes, occlusion_cause_valid=causes >= 0,
        self_blocked_fraction=blocked.mean(axis=-1).astype(np.float32))
    raw_record = ROOT / record['scene'] / 'record.json'
    shutil.copyfile(raw_record, folder / 'provenance/render_record.json')
    for key in ('environment_parameters', 'lighting_parameters', 'visibility_receipt'):
        shutil.copyfile(ROOT / record[key], folder / 'provenance' / (key + '.json'))


def validate_dataset(dataset, records):
    sys.path.insert(0, str(ROOT / 'vggt-omega'))
    from training.keypoint_contract import read_validity, read_occlusion
    checked, unknown = 0, 0
    for record in records:
        folder = dataset / record['export_scene']
        complete, ids, missing = scene_is_complete(folder)
        assert complete and ids == [0, 1, 2] and not missing
        meta = read(folder / 'scene_meta.json')
        assert meta['source_frame'] == record['frame'] and meta['split'] == record['split']
        assert meta['rig'] == record['rig'] and meta['environment_id'] == record['environment_id']
        with np.load(folder / 'keypoints.npz', allow_pickle=False) as a, np.load(folder / 'keypoint_validity.npz', allow_pickle=False) as valid:
            assert a['uv'].shape == (3, 23, 2) and a['joint_world_xyz'].shape == (23, 3)
            assert str(a['definition_version']) == VERSION
            for key in ('xy_label_valid', 'visibility_valid', 'world_valid', 'surface_valid'):
                np.testing.assert_array_equal(a[key], valid[key])
            loader_flags = read_validity(a, folder / 'keypoints.npz')
            for key in loader_flags:
                np.testing.assert_array_equal(loader_flags[key], a[key])
            loader_causes = read_occlusion(a, folder / 'keypoints.npz')
            unknown += int((~a['visibility_valid']).sum())
            for i, view in enumerate(record['views']):
                stem = f'{i:08d}'
                with np.load(ROOT / view['truth'], allow_pickle=False) as truth:
                    np.testing.assert_array_equal(read_pfm(folder / 'rendered_depth_maps' / f'{stem}.pfm'), truth['depth'])
                    np.testing.assert_array_equal(np.asarray(Image.open(folder / 'masks' / f'{stem}.png')) > 0, truth['body'])
                    e, k = read_cam(folder / 'cams' / f'{stem}_cam.txt')
                    np.testing.assert_array_equal(e, truth['E'])
                    np.testing.assert_array_equal(k, truth['K'])
                ann = read(folder / 'annotations' / f'{stem}.json')
                assert len(ann['keypoints']) == 23 and ann == meta['views'][i]
                assert ann['frame'] == record['export_index'] and ann['source_frame'] == record['frame']
                np.testing.assert_array_equal(a['intrinsics'][i], k)
                np.testing.assert_array_equal(a['extrinsics'][i], e)
                for j, kp in enumerate(ann['keypoints']):
                    np.testing.assert_allclose(kp['uv'], a['uv'][i, j], atol=0, rtol=0)
                    assert kp['visibility_valid'] == bool(a['visibility_valid'][i, j])
                for path in (folder / 'blended_images' / f'{stem}.jpg', folder / 'blended_images' / f'{stem}_masked.jpg'):
                    with Image.open(path) as im:
                        assert im.mode == 'RGB' and im.size == (384, 384)
                for key in ('image', 'truth'):
                    copy = folder / 'provenance' / Path(view[key]).name
                    assert sha(copy) == view['file_sha256'][copy.name]
                assert sha(folder / 'provenance/render_record.json') == sha(ROOT / record['scene'] / 'record.json')
        with np.load(folder / 'keypoint_occlusion.npz', allow_pickle=False) as occ:
            assert str(occ['annotation_sha256']) == sha(folder / 'keypoints.npz')
            assert set(np.unique(occ['occlusion_cause'])) <= {-1, 0, 1}
            np.testing.assert_array_equal(loader_causes, occ['occlusion_cause'])
        checked += 1
    independent = audit_dataset(dataset, 23, .001)
    assert independent['error_count'] == 0, independent['errors'][:3]
    return dict(passed=True, scenes=checked, views=checked * 3, source_RGB_lossless_copies_verified=True,
        PFM_depth_exact=True, masks_exact=True, camera_matrices_exact=True, split_preserved=True,
        legacy_scene_loader_passed=True, keypoint_audit=independent,
        unknown_or_out_of_frame_visibility_labels=unknown, new_RGB_renders=0)


def metadata_and_provenance(dataset, job, records, source, smoke):
    reference = ROOT / source['reference_dataset']
    final = BASE / job['name']
    frame_numbers = [r['export_index'] for r in records]
    temporal = []
    for r in records:
        temporal.append(dict(local_index=r['export_index']-1, source_frame_id=r['frame'], scene=r['export_scene'],
            segment_id=f"{r['group']}:{r['rig']}:{r['environment_id']}", source_frame_delta=None, transition_valid=False,
            motion_source_id=r['group'], split=r['split'], rig=r['rig'], environment_id=r['environment_id']))
    (dataset / 'temporal_index.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in temporal))
    (dataset / 'scene_index.jsonl').write_text(''.join(json.dumps(dict(x, dev_condition=r['dev_condition'],
        source_render_scene=r['scene'])) + '\n' for x, r in zip(temporal, records)))
    shutil.copyfile(reference / 'surface_keypoint_definition.json', dataset / 'surface_keypoint_definition.json')
    meta = dict(format='blenderpymouse_core23_v1', dataset_name=job['name'],
        dataset_root='\\\\wsl.localhost\\Ubuntu' + str(final).replace('/', '\\'),
        dataset_relative_path=final.relative_to(ROOT).as_posix(), generated_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        source_blend=source['blend'], source_blend_sha256=source['source_sha256'], source_dataset=source['base'],
        source_animation_frames=[49, 97, 145], frame_number_semantics='Export scene ID; source animation frame is separately recorded per scene.',
        frames=[min(frame_numbers), max(frame_numbers)], frame_numbers=frame_numbers, scene_count=len(records), incomplete_scene_count=0,
        scene_folders=[r['export_scene'] for r in records], cameras=CAMERAS, view_counts={'3': len(records)},
        render_resolution=[384, 384], image_format='JPEG', jpeg_quality=95, lossless_original_RGB=True,
        mask_format='PNG binary0/255, primary-ray visible skin, hair excluded', depth_format='PFM',
        depth_units='millimeter', depth_semantics='camera_z', pixel_center_convention='integer_array_indices; K unchanged from raw truth',
        split=job['split'], recommended_split=job['split'], train_allowed=job['split'] == 'train',
        protected_sealed=job['split'] == 'sealed', camera_geometry_varies_per_scene=True,
        environment_ids=sorted({r['environment_id'] for r in records}), rig_ids=sorted({r['rig'] for r in records}),
        dev_conditions=dict(collections.Counter(r['dev_condition'] for r in records)),
        keypoint_annotations=dict(format=KP_FORMAT, keypoint_count=23, names=read(reference / 'metadata.json')['keypoint_annotations']['names'],
            definition_version=VERSION, visibility_rule='Legacy support-depth/mask vote; manual surface slots use body center rays with scene depth/mask conflict validity.'),
        keypoint_definition_version=VERSION, keypoint_definition_file='surface_keypoint_definition.json',
        temporal=dict(dataset_id=job['name'], path='BlenderPyMouse/' + job['name'], temporal_mode='sparse_crossed_rig_environment',
            temporal_eligible=False, frame_count=len(records), motion_source_id=records[0]['group'], recommended_split=job['split'],
            timing='source_frame_order_only', clock_verified=False, evidence='three sparse poses repeated across rigs/environments; scene IDs are not video timestamps',
            index='temporal_index.jsonl', camera_motion='varies_between_scenes', intrinsics_constant=False, views=3),
        layout=dict(images='frame_xxxxxx/blended_images/00000000.jpg', masked_images='frame_xxxxxx/blended_images/00000000_masked.jpg',
            masks='frame_xxxxxx/masks/00000000.png', depths='frame_xxxxxx/rendered_depth_maps/00000000.pfm',
            depth_previews='frame_xxxxxx/rendered_depth_maps/00000000_preview.png', cameras='frame_xxxxxx/cams/00000000_cam.txt',
            pair_file='frame_xxxxxx/cams/pair.txt', annotations='frame_xxxxxx/annotations/00000000.json', keypoints='frame_xxxxxx/keypoints.npz',
            lossless_images='frame_xxxxxx/provenance/rgb_0.png', raw_truth='frame_xxxxxx/provenance/truth_0.npz'),
        engineering_smoke=smoke, provenance='provenance', version='ComplexBackgroundV5')
    write(dataset / 'metadata.json', meta)
    prov = dataset / 'provenance'
    prov.mkdir()
    plan = read(OUT / 'conversion_plan.json')
    # A Train dataset must not bundle the DEV/sealed camera labels embedded in
    # the full render manifest. Keep only this dataset's slice plus parent hashes.
    write(prov / 'conversion_plan.json', dict(schema=plan['schema'], datasets=[job], sources=[source], records=records,
          parent_conversion_plan_sha256=sha(OUT / 'conversion_plan.json'), raw_manifest_sha256=plan['raw_manifest_sha256']))
    shutil.copyfile(OUT / 'source_geometry_receipt.json', prov / 'source_geometry_receipt.json')
    shutil.copytree(OUT / 'source_geometry' / job['source'], prov / 'source_geometry')
    raw = ROOT / read(OUT / 'conversion_plan.json')['raw_output']
    design = read(raw / 'render_design.json')
    rig_ids = {r['rig'] for r in records}
    env_ids = {r['environment_id'] for r in records}
    write(prov / 'render_design.json', dict(schema=design['schema'], parent_render_design_sha256=sha(raw / 'render_design.json'),
          sources=[s for s in design['sources'] if s['id'] == job['source']],
          rigs=[r for r in design['rigs'] if r['id'] in rig_ids],
          environments=[e for e in design['environments'] if e['id'] in env_ids],
          assignments=[a for a in design['assignments'] if a['rig'] in rig_ids and a['environment'] in env_ids],
          expected_scenes=len(records), expected_views=len(records)*3))
    for name in ('protocol.json', 'data_qa.json'):
        shutil.copyfile(raw / name, prov / name)
    for name in ('convert_v5_to_blenderpymouse_20261025.py', 'v5_render_contract_20261024.py'):
        shutil.copyfile(ROOT / 'tools' / name, prov / name)
    shutil.copyfile(ROOT / 'BlenderScripts/extract_v5_core23_for_blenderpymouse_20261025.py', prov / 'extract_v5_core23_for_blenderpymouse_20261025.py')
    (dataset / 'README_数据说明.md').write_text(f'''# {job['name']}

标准 BlenderPyMouse/Core23/BlendedMVS 布局，{len(records)}个场景、每场景3视角。
split={job['split']}；Train/Dev/Sealed分目录。封存集只提供原始合成数据及几何标注，不能用于训练或未经授权的模型评分。
frame_XXXXXX是本次导出的场景编号；真实动画帧49/97/145、机位、环境、DEV条件见scene_index.jsonl和各scene_meta.json。不是连续视频。

RGB使用标准JPEG95、4:4:4；provenance/rgb_i.png保留逐字节相同的无损原图。
PFM是原primary-ray camera-Z float32的精确转换（小端、文件底行先存）；零表示无效。
K/E完全保留原值，像素中心为整数数组下标，禁止再次加减0.5。
mask为可见皮肤的0/255 PNG，不包含毛发、玻璃或被遮挡部分，不是毛发轮廓或amodal标签。
Core23沿用现有23通道及06/07/23手工表面三角形锚点。世界位置从固定源文件回放，按新K/E重投影；不复用旧相机uv或可见性。
visibility_valid=false时不得把未知情况当作负标签。occlusion_cause只使用0可见、1自身遮挡、-1未知，不把装置遮挡伪装成另一只鼠。
provenance下的render_record.json保留原路径用于溯源；实际加载入口为scene_meta、annotations和keypoints。
所有训练所需文件均在本目录内；无须依赖旧output或.blend来读取本数据。
''', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    plan = read(OUT / 'conversion_plan.json')
    assert read(OUT / 'source_geometry_receipt.json')['passed']
    raw = ROOT / plan['raw_output']
    assert sha(raw / 'manifest.json') == plan['raw_manifest_sha256']
    assert read(raw / 'completion.json')['passed']
    if not args.smoke:
        assert (OUT / 'smoke_receipt.json').exists() and read(OUT / 'smoke_receipt.json')['passed']
    receipts = []
    sources = {s['id']: s for s in plan['sources']}
    staging = OUT / ('smoke' if args.smoke else 'staging')
    staging.mkdir(exist_ok=False)
    for job in plan['datasets']:
        records = [r for r in plan['records'] if r['export_dataset'] == job['name']]
        assert len(records) == job['scene_count']
        if args.smoke:
            records = records[:1]
        dataset = staging / job['name']
        dataset.mkdir()
        for n, record in enumerate(records, 1):
            with np.load(OUT / 'source_geometry' / job['source'] / f"frame_{record['frame']:06d}.npz", allow_pickle=False) as z:
                geometry = dict(z)
            convert_scene(dataset, record, geometry)
            if n % 24 == 0 or n == len(records):
                write(OUT / 'conversion_progress.json', dict(state='converting', smoke=args.smoke, dataset=job['name'], completed=n, total=len(records)))
                print('CONVERTED', job['name'], n, '/', len(records), flush=True)
        metadata_and_provenance(dataset, job, records, sources[job['source']], args.smoke)
        result = validate_dataset(dataset, records)
        result.update(dataset=job['name'], split=job['split'], engineering_smoke=args.smoke)
        write(dataset / 'dataset_validation.json', result)
        pins = {p.relative_to(dataset).as_posix(): sha(p) for p in sorted(dataset.rglob('*')) if p.is_file()}
        write(dataset / 'data_file_pins.json', dict(schema='blenderpymouse-v5-export-file-pins-v1', files=pins))
        write(dataset / 'completion.json', dict(passed=True, scenes=len(records), views=len(records)*3,
              data_file_pins_sha256=sha(dataset / 'data_file_pins.json'), schema='blenderpymouse_core23_v1', split=job['split']))
        receipts.append(result)
    if args.smoke:
        write(OUT / 'smoke_receipt.json', dict(passed=True, datasets=receipts, scenes=6, views=18))
        print('SIX_DATASET_SMOKE_PASSED', flush=True)
        return
    assert (OUT / 'smoke_receipt.json').exists() and read(OUT / 'smoke_receipt.json')['passed']
    # Publish only after all six complete datasets pass. Both locations are inside
    # the same workspace/filesystem; do not replace any existing dataset.
    for job in plan['datasets']:
        src = (staging / job['name']).resolve()
        dst = (BASE / job['name']).resolve()
        assert src.is_relative_to(OUT.resolve()) and dst.is_relative_to(BASE.resolve()) and not dst.exists()
    for job in plan['datasets']:
        (staging / job['name']).rename(BASE / job['name'])
    registry = BASE / '_annotation_audit/v5_complex_background_20261025'
    registry.mkdir(parents=True, exist_ok=False)
    summary = dict(passed=True, datasets=receipts, scene_count=sum(x['scenes'] for x in receipts), view_count=sum(x['views'] for x in receipts),
        split_counts=dict(collections.Counter(r['split'] for r in plan['records'])),
        canonical_roots=[(BASE / j['name']).relative_to(ROOT).as_posix() for j in plan['datasets']],
        source_render_archive=plan['raw_output'], original_datasets_modified=False, original_renders_modified=False,
        converter_sha256=sha(Path(__file__)), completed_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    assert summary['scene_count'] == 768 and summary['view_count'] == 2304
    write(registry / 'conversion_receipt.json', summary)
    write(OUT / 'completion.json', summary)
    write(raw / 'blenderpymouse_export.json', dict(passed=True, canonical_roots=summary['canonical_roots'],
        conversion_receipt=(registry / 'conversion_receipt.json').relative_to(ROOT).as_posix()))
    write(OUT / 'conversion_progress.json', dict(state='completed', datasets=6, scenes=768, views=2304))
    print('BLENDERPYMOUSE_PUBLICATION_COMPLETE', 768, 2304, flush=True)


if __name__ == '__main__':
    main()
