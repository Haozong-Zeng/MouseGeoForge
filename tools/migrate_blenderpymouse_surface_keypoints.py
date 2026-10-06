"""Transactional migration of approved Core23 surface slots; run in WSL Python.

Blender staging is separate and read-only. This writer preserves the established
channel names for model compatibility and publishes the anatomical semantic names
separately. Source-replay failures are candidates, NEVER valid supervision.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'output/surface_keypoint_migration_v3'
DEST = ROOT / 'dataset/BlenderPyMouse'
CAND = BASE / 'candidate'
IDS = [5, 6, 22]
SEMANTICS = ['dorsal_thorax_surface', 'dorsal_lumbar_surface', 'ventral_abdomen_surface']
VERSION = 'pymousestudio_manual_surface_core23_v3'
FORMAT = 'freemoving_core23_surface_v3'
sys.path.insert(0, str(ROOT / 'vggt-omega'))
from training.keypoint_contract import read_validity, read_occlusion


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def save_npz(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)


def semantic_names(names):
    result = list(names)
    for k, name in zip(IDS, SEMANTICS):
        result[k] = name
    return result


def update_json(node, a, stage, flags, view=None, instance=0):
    """Update all redundant JSON representations, not only keypoints.json."""
    if not isinstance(node, dict):
        return
    dual = a['joint_world_xyz'].ndim == 3
    if 'view_id' in node:
        view = list(a['view_ids']).index(int(node['view_id']))
    if 'instance_id' in node and dual:
        instance = list(a['instance_ids']).index(int(node['instance_id']))
    for key in ('format', 'keypoint_format'):
        if node.get(key) == 'freemoving_core23_bone_head_v1':
            node[key] = FORMAT
    if 'joint_coordinate_rule' in node:
        node['joint_coordinate_rule'] = 'Slots 06/07/23: fixed REST mesh triangle barycentric surface anchors; other slots unchanged.'
    if 'names' in node or 'keypoint_names' in node:
        node['semantic_names'] = semantic_names(a['names'].tolist())
        node['definition_version'] = VERSION
    for key in ('joint_world_xyz', 'surface_world_xyz', 'support_vertex_ids', 'support_world_xyz', 'endpoints'):
        if key in node and isinstance(node[key], list) and len(node[key]) == 23:
            values = a[key][instance] if dual and key != 'endpoints' else a[key]
            for k in IDS:
                node[key][k] = values[k].tolist()
    if 'id' in node and int(node['id']) in IDS and 'uv' in node:
        k = int(node['id']); j = IDS.index(k)
        def w(key): return a[key][instance, k] if dual else a[key][k]
        def v(key): return a[key][view, instance, k] if dual else a[key][view, k]
        def flag(key):
            arr = flags[key]
            return bool(arr[view, instance, k] if dual else arr[view, k]) if arr.ndim == a['in_frame'].ndim else bool(arr[instance, k] if dual else arr[k])
        xyz = w('joint_world_xyz').tolist(); cam = v('joint_camera_xyz').tolist(); uv = v('uv').tolist()
        vis = bool(v('surface_visible')); known = flag('visibility_valid'); label = flag('xy_label_valid')
        node.update(semantic_name=SEMANTICS[j], definition_version=VERSION,
                    endpoint='mesh_surface', coordinate_kind='fixed_rest_triangle_barycentric',
                    joint_world_xyz=xyz, surface_world_xyz=xyz, camera_xyz=cam,
                    surface_camera_xyz=cam, uv=uv, camera_depth=cam[2], surface_depth=cam[2],
                    in_frame=bool(v('in_frame')), surface_visible=vis, visible=vis,
                    occluded_but_annotated=bool(label and known and not vis and v('in_frame')),
                    confidence=1.0 if label else 0.0, xy_label_valid=label,
                    visibility_valid=known, world_valid=flag('world_valid'), surface_valid=flag('surface_valid'),
                    support_visible_count=int(v('support_visible_count')), support_vertex_count=16,
                    support_min_depth_residual=None,
                    visibility_method='bare_mesh_anchor_ray; raster mask consistency; unknown on replay mismatch',
                    candidate_only=not flag('world_valid'))
        if 'x' in node: node['x'], node['y'] = uv
    for key in ('views', 'instances', 'keypoints'):
        for child in node.get(key, []):
            update_json(child, a, stage, flags, view, instance)


def prepare():
    inventory = json.loads((BASE / 'inventory.json').read_text())
    recipe = json.loads((ROOT / 'output/surface_anchor_demo_v1/rest_surface_anchors.json').read_text())
    manifest = dict(version=VERSION, status='prepared', datasets=[], files=[], protected=[], skipped=inventory['skipped'])
    def record(rel):
        old = DEST / rel; new = CAND / rel
        manifest['files'].append(dict(path=str(rel), old_sha256=sha(old) if old.exists() else None, new_sha256=sha(new)))
    for job in inventory['jobs']:
        name = job['dataset']; ds = DEST / name; stage_dir = BASE / 'staged' / name
        report = json.loads((stage_dir / 'report.json').read_text())
        files = sorted(ds.glob('frame_*/keypoints.npz'))
        assert len(report['frames']) == len(files) == job['frames']
        summary = dict(dataset=name, frames=len(files), instances=0, invalid_world_points=0,
                       unknown_visibility_labels=0, source_mismatch_frames=[], max_local_support_error=report['max_local_support_error'])
        first = None
        for file in files:
            with np.load(file, allow_pickle=False) as z:
                a = dict(z); flags = read_validity(z, file); causes = read_occlusion(z, file)
            assert 'definition_version' not in a, f'Already migrated: {file}'
            old = {k: v.copy() for k, v in a.items()}
            with np.load(stage_dir / (file.parent.name + '.npz'), allow_pickle=False) as z: stage = dict(z)
            assert sha(file) == str(stage['source_annotation_sha256'])
            dual = a['joint_world_xyz'].ndim == 3; mcount = a['joint_world_xyz'].shape[0] if dual else 1
            geometry = stage.get('geometry_valid', np.ones(mcount, dtype=bool))
            verified = bool(stage['full_geometry_verified'])
            for key, src in [('joint_world_xyz','world'), ('surface_world_xyz','world'),
                             ('support_world_xyz','support'), ('support_vertex_ids','support_ids')]:
                if dual: a[key][:, IDS] = stage[src]
                else: a[key][IDS] = stage[src][0]
            for key, src in [('uv','uv'), ('joint_camera_xyz','camera_xyz')]:
                if dual: a[key][:, :, IDS] = stage[src]
                else: a[key][:, IDS] = stage[src][:, 0]
            # Center visibility differs from the historical support-vote heuristic.
            for vi, vid in enumerate(a['view_ids']):
                stem = f'{int(vid):08d}'
                with Image.open(file.parent / 'blended_images' / (stem + '.jpg')) as im: width, height = im.size
                mask_path = file.parent / 'masks' / (stem + '.png')
                mask = np.array(Image.open(mask_path)) if mask_path.exists() else None
                if mask is not None and mask.ndim == 3: mask = mask.max(-1)
                for m in range(mcount):
                    for j, k in enumerate(IDS):
                        idx = (vi,m,k) if dual else (vi,k); wi = (m,k) if dual else (k,)
                        x,y = stage['uv'][vi,m,j]; z = stage['camera_xyz'][vi,m,j,2]
                        inside = bool(np.isfinite([x,y,z]).all() and z>0 and 0<=x<width and 0<=y<height)
                        body_vis = bool(stage['body_visible'][vi,m,j]); known = bool(verified and geometry[m] and inside)
                        # A foreground conflict could be wall/fur/rasterization, not self occlusion.
                        if body_vis and inside and mask is not None:
                            ix,iy = int(round(float(x))),int(round(float(y)))
                            patch=mask[max(0,iy-1):min(height,iy+2),max(0,ix-1):min(width,ix+2)]
                            if not np.any(patch): known=False
                        a['in_frame'][idx]=inside
                        a['surface_visible'][idx]=body_vis and known
                        a['support_visible_count'][idx]=int(stage['support_visible_count'][vi,m,j]) if known else 0
                        flags['xy_label_valid'][idx]=bool(geometry[m] and z>0)
                        flags['visibility_valid'][idx]=known
                        flags['world_valid'][wi]=bool(geometry[m]);flags['surface_valid'][wi]=bool(geometry[m])
                        causes[idx]=int(stage['occlusion_cause'][vi,m,j]) if known else -1
            a.update(flags)
            a['definition_version']=np.asarray(VERSION)
            a['semantic_names']=np.asarray(semantic_names(a['names'].tolist()))
            a['coordinate_kind']=np.asarray(['fixed_rest_triangle_barycentric' if k in IDS else 'legacy_unchanged' for k in range(23)])
            a['source_annotation_sha256']=stage['source_annotation_sha256']
            a['surface_anchor_source_sha256']=stage['source_blend_sha256']
            if str(a['format']) == 'freemoving_core23_bone_head_v1': a['format']=np.asarray(FORMAT)
            if 'keypoint_format' in a: a['keypoint_format']=np.asarray(FORMAT)
            if 'endpoints' in a:
                a['endpoints']=a['endpoints'].astype('U32');a['endpoints'][IDS]='mesh_surface'
            keep=[k for k in range(23) if k not in IDS]
            for key in ('joint_world_xyz','surface_world_xyz','support_world_xyz','support_vertex_ids'):
                assert np.array_equal(a[key][:,keep] if dual else a[key][keep],old[key][:,keep] if dual else old[key][keep])
            for key in ('uv','joint_camera_xyz','in_frame','surface_visible','support_visible_count'):
                assert np.array_equal(a[key][:,:,keep] if dual else a[key][:,keep],old[key][:,:,keep] if dual else old[key][:,keep])
            for key in ('names','bones','intrinsics','extrinsics','frame','view_ids','instance_ids'):
                if key in old: assert np.array_equal(a[key],old[key])
            rel=file.relative_to(DEST); target=CAND/rel
            save_npz(target,a);record(rel)
            validrel=rel.with_name('keypoint_validity.npz')
            save_npz(CAND/validrel,dict(schema_version=np.asarray(1),**flags));record(validrel)
            occpath=file.with_name('keypoint_occlusion.npz')
            occ=dict(np.load(occpath,allow_pickle=False)) if occpath.exists() else {}
            occ.update(schema_version=np.asarray(1), annotation_sha256=np.asarray(sha(target)), names=a['names'],view_ids=a['view_ids'],
                       occlusion_cause=causes,occlusion_cause_valid=causes>=0,definition_version=np.asarray(VERSION),
                       surface_anchor_source_sha256=stage['source_blend_sha256'])
            for fraction,code in [('other_mouse_blocked_fraction',2),('self_blocked_fraction',1)]:
                if fraction in occ:
                    occ[fraction]=occ[fraction].copy()
                    if dual: occ[fraction][:,:,IDS]=np.nan
                    else: occ[fraction][:,IDS]=np.nan
            if dual:occ['instance_ids']=a['instance_ids']
            occrel=rel.with_name('keypoint_occlusion.npz');save_npz(CAND/occrel,occ);record(occrel)
            for jp in [file.with_suffix('.json'), file.with_name('scene_meta.json'), *sorted((file.parent/'annotations').glob('*.json'))]:
                data=json.loads(jp.read_text());update_json(data,a,stage,flags)
                data['definition_version']=VERSION
                jr=jp.relative_to(DEST);save_json(CAND/jr,data);record(jr)
            # Read both contracts from candidates: checks shapes and annotation hashes.
            with np.load(target) as z: read_validity(z,target);read_occlusion(z,target)
            summary['instances']+=mcount
            summary['invalid_world_points']+=int((~geometry).sum())*3
            summary['unknown_visibility_labels']+=int((~flags['visibility_valid'][...,IDS]).sum())
            if not geometry.all():summary['source_mismatch_frames'].append(dict(frame=int(a['frame']),instances=np.flatnonzero(~geometry).tolist()))
            if first is None:first=(a,stage)
        # Versioned definition is authoritative; historical export maps remain labeled as legacy.
        definition=dict(version=VERSION, changed_zero_based_indices=IDS, changed_display_ids=[6,7,23],
                        stable_channel_names=first[0]['names'][IDS].tolist(), semantic_names=SEMANTICS,
                        channel_name_policy='Keep existing model channel names; semantic_names are authoritative anatomical names.',
                        coordinate_rule='Fixed REST triangle barycentric anchors; joint_world_xyz and surface_world_xyz both store these surface positions for changed slots.',
                        invalid_policy='Finite source candidates may remain for inspection; false xy/world/surface validity disables supervision. Never use candidate coordinates without validity masks.',
                        support_rule='16 vertices for local visibility inspection, NOT averaged to define anchor position.',
                        visibility_rule='Bare-mesh center ray including other mice, raster-mask consistency; unknown on source mismatch. Not a fur-strand visibility annotation.',
                        source_blend=job['source_blend'],source_blend_sha256=report['source_blend_sha256'],
                        anchors=recipe['anchors'],geometry_verification=report['geometry'],
                        support_vertex_ids=first[1]['support_ids'].tolist(),summary=summary,
                        historical_export_artifacts=[p.name for p in ds.glob('*support*')],
                        historical_artifact_warning='Original support maps/audits describe v1 for changed slots and must not regenerate v3 coordinates. Use this definition and staged migration script.')
        dr=Path(name)/'surface_keypoint_definition.json';save_json(CAND/dr,definition);record(dr)
        meta=json.loads((ds/'metadata.json').read_text());meta.update(keypoint_definition_version=VERSION,keypoint_definition_file='surface_keypoint_definition.json',
            keypoint_semantic_names=semantic_names(first[0]['names'].tolist()),keypoint_migration_summary=summary,
            keypoint_migration_note='06/07/23 surface anchors replace former bone heads. Other 20 slots, RGB, depth, cameras, poses and temporal indexes unchanged. Legacy export support maps are historical for these slots.')
        if meta.get('keypoint_format')=='freemoving_core23_bone_head_v1':meta['keypoint_format']=FORMAT
        mr=Path(name)/'metadata.json';save_json(CAND/mr,meta);record(mr)
        for p in ds.glob('*temporal*'):
            if p.is_file():manifest['protected'].append(dict(path=str(p.relative_to(DEST)),sha256=sha(p)))
        manifest['datasets'].append(summary)
        print('PREPARED',name,summary,flush=True)
    save_json(BASE/'manifest.json',manifest)


def commit():
    manifest=json.loads((BASE/'manifest.json').read_text());assert manifest['status']=='prepared'
    backup=ROOT/'backup/keypoint_migrations/BlenderPyMouse-before-surface-core23-v3.zip'
    backup.parent.mkdir(parents=True,exist_ok=True)
    assert not backup.exists(),f'Backup already exists; inspect transaction instead of overwriting: {backup}'
    for item in manifest['files']:
        old=DEST/item['path'];new=CAND/item['path']
        assert (sha(old) if old.exists() else None)==item['old_sha256'],str(old)
        assert sha(new)==item['new_sha256'],str(new)
    with zipfile.ZipFile(backup,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        z.writestr('migration_manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
        for item in manifest['files']:
            if item['old_sha256'] is not None:z.write(DEST/item['path'],item['path'])
    with zipfile.ZipFile(backup) as z:assert z.testzip() is None
    manifest['backup']=str(backup);manifest['backup_sha256']=sha(backup);manifest['status']='committing'
    save_json(BASE/'manifest.json',manifest)
    for item in manifest['files']:
        dest=DEST/item['path'];temp=dest.with_name(dest.name+'.surface-v3-tmp')
        with temp.open('xb') as f:f.write((CAND/item['path']).read_bytes())
        os.replace(temp,dest)
    for item in manifest['files']:assert sha(DEST/item['path'])==item['new_sha256']
    for item in manifest['protected']:assert sha(DEST/item['path'])==item['sha256']
    manifest['status']='committed';save_json(BASE/'manifest.json',manifest)
    print('COMMITTED',len(manifest['files']),'files; backup',backup,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','commit']);args=parser.parse_args()
    {'prepare':prepare,'commit':commit}[args.action]()
