"""Finite V5 Cycles renderer, geometric prescreen, v3 payloads and atomic records.

Only the approved procedural environment collection is appended from the review
project. Source mouse/animation/contact geometry and the 68 original rigs remain
unchanged. Never saves a .blend, calls a model, or starts training.
"""
import argparse
import collections
import datetime as dt
import faulthandler
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import traceback

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'BlenderScripts'))
import v5_render_contract_20261024 as contract
from audit_visible_surface_geometry_20261007 import camera_parameters
from export_visible_surface_truth_20261007 import geometry, trace, fixture


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def open_source(source, cfg):
    path = ROOT / source['path']
    assert contract.sha(path) == cfg['source_sha256'][source['id']]
    bpy.ops.wm.open_mainfile(filepath=str(path))
    scene = bpy.context.scene
    hidden = []
    for collection in bpy.data.collections:
        if collection.name.startswith('CameraItems'):
            collection.hide_render = True
    for obj in list(scene.objects):
        if obj.type == 'LIGHT' or obj.name in ('Building_Floor', 'Part_WhiteHexagonBackground', 'Part_TopCover') or obj.name.startswith(('Part_Camera', 'Part_CamLen', 'CameraItems', 'Part_PiCam', 'Part_CNC-Camera', 'Part_BottomCamScrew', 'Part_SideCamScrew')):
            obj.hide_render = True
            hidden.append(obj.name)
            for child in obj.children_recursive:
                child.hide_render = True
                hidden.append(child.name)
        if obj.type == 'CURVES':
            obj.hide_viewport = True
    assert source['mesh'] in scene.objects and not scene.objects[source['mesh']].hide_render
    scene.render.engine = 'CYCLES'
    scene.render.resolution_x = scene.render.resolution_y = cfg['resolution']
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.color_depth = '8'
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    scene.render.film_transparent = False
    scene.render.use_persistent_data = True
    scene.cycles.samples = cfg['cycles_samples']
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 10
    scene.cycles.transmission_bounces = 8
    scene.view_settings.exposure = 0
    return scene, dict(source=source['id'], source_sha256=cfg['source_sha256'][source['id']],
                       hidden_equipment_and_replaced_shell=sorted(set(hidden)),
                       visible_meshes=sorted(o.name for o in scene.objects if o.type == 'MESH' and not o.hide_render))


def install_environment(scene, params, design):
    with bpy.data.libraries.load(str(ROOT / design['template_blend']), link=False) as (src, dst):
        assert design['template_collection'] in src.collections
        assert design['template_world'] in src.worlds
        dst.collections = [design['template_collection']]
        dst.worlds = [design['template_world']]
    coll = dst.collections[0]
    world = dst.worlds[0]
    scene.collection.children.link(coll)
    scene.world = world
    # Newly appended objects have valid local transforms but stale/identity
    # matrix_world until the linked collection enters the evaluated graph.
    # Applying layout transforms before this update collapses the room at zero.
    bpy.context.view_layer.update()
    rotation = Matrix.Rotation(params['rotation_z'], 4, 'Z')
    rows = []
    for obj in coll.all_objects:
        name = obj.name.removeprefix(contract.PREFIX)
        offset = [0, 0, 0]
        if name.startswith('ceiling ') and name != 'ceiling':
            offset = params['ceiling_offset']
        elif name.startswith(('storage ', 'bin lid')):
            offset = params['storage_offset']
        elif name.startswith(('post base', 'instrument post', 'post clamp', 'post cross arm')):
            offset = params['post_offsets'][int(name[-2:])]
        elif name in ('above floor housing', 'rear hanging service'):
            offset = params['post_offsets'][0]
        elif name.startswith(('below floor ', 'controller vent')):
            offset = params['controller_offset']
        elif name.startswith('back wall panel'):
            offset = params['wall_panel_offsets'][int(name[-2:])]
        obj.matrix_world = rotation @ Matrix.Translation(Vector(offset)) @ obj.matrix_world
        if obj.type == 'MESH':
            rows.append(obj)
    materials = {m for o in rows for m in o.data.materials if m is not None}
    for mat in materials:
        if not mat.use_nodes:
            continue
        for node in mat.node_tree.nodes:
            if node.type == 'BSDF_PRINCIPLED':
                color = node.inputs['Base Color'].default_value
                color[:3] = [min(1, v * params['material_gain']) for v in color[:3]]
            elif node.type == 'VALTORGB':
                for element in node.color_ramp.elements:
                    color = element.color
                    color[:3] = [min(1, v * params['material_gain']) for v in color[:3]]
    bpy.context.view_layer.update()
    actual = dict(parameters=params, template_blend_sha256=contract.sha(ROOT / design['template_blend']),
                  world_color=[.63, .69, .78],
                  objects=[dict(name=o.name, type=o.type, matrix_world=[list(r) for r in o.matrix_world],
                                dimensions=list(o.dimensions), materials=[m.name for m in o.data.materials]) for o in sorted(rows, key=lambda x: x.name)])
    assert all(o.type == 'MESH' for o in rows)
    if params['reference_approved_layout']:
        reference = contract.read(ROOT / contract.REVIEW_REL / 'scene_review_manifest.json')
        expected = {o['name']: o for o in reference['environment']}
        assert set(expected) == {o.name for o in rows}
        for obj in rows:
            np.testing.assert_allclose(np.array(obj.matrix_world), expected[obj.name]['matrix_world'], atol=2e-5, rtol=1e-6,
                                       err_msg='Imported template transform differs from the approved scene: ' + obj.name)
    return actual, rotation


def setup_lighting(scene, row, rotation):
    params = contract.lighting_parameters(row)
    bg = next(n for n in scene.world.node_tree.nodes if n.type == 'BACKGROUND')
    bg.inputs['Color'].default_value = (.63, .69, .78, 1)
    bg.inputs['Strength'].default_value = params['world_strength']
    for light in params['lights']:
        obj = bpy.data.objects[light['name']]
        obj.location = rotation @ Vector(light['position'])
        target = rotation @ Vector(light['target'])
        obj.rotation_euler = (target - obj.location).to_track_quat('-Z', 'Y').to_euler()
        obj.data.energy = light['energy']
        obj.data.color = light['color']
        obj.data.shadow_soft_size = light['soft_radius']
        obj.data.spot_size = math.radians(light['cone_degrees'])
        obj.data.spot_blend = light['blend']
        light['world_position'] = list(obj.location)
        light['world_target'] = list(target)
    scene.cycles.seed = row['lighting_seed'] % 2147483647
    return params


def cameras_for(scene, rig, source):
    anchor = Vector(source['anchor'])
    cameras = []
    for i, relative in enumerate(rig['relative_positions']):
        d = bpy.data.cameras.new(f'V5_{rig["id"]}_{i}')
        d.type = 'PERSP'
        d.lens = rig['lens_mm']
        d.sensor_width = 36
        d.sensor_fit = 'HORIZONTAL'
        d.clip_start = .1
        d.clip_end = 4000
        cam = bpy.data.objects.new(d.name, d)
        scene.collection.objects.link(cam)
        cam.location = anchor + Vector(relative)
        cam.rotation_mode = 'QUATERNION'
        cam.rotation_quaternion = (anchor - cam.location).to_track_quat('-Z', 'Y') @ Quaternion((0, 0, 1), math.radians(rig['roll_deg'][i]))
        cameras.append(cam)
    bpy.context.view_layer.update()
    return cameras


def remove_cameras(cameras):
    for cam in cameras:
        data = cam.data
        bpy.data.objects.remove(cam, do_unlink=True)
        bpy.data.cameras.remove(data)


def ray_payload(scene, cam, side, tree, instances):
    yy, xx = np.mgrid[:side, :side]
    uv = np.stack((xx + .5, yy + .5, np.ones_like(xx)), -1).reshape(-1, 3)
    k, e = camera_parameters(scene, cam, side, side)
    camera_rays = uv @ np.linalg.inv(k).T
    rays = camera_rays @ e[:3, :3]
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    near_camera = camera_rays * (cam.data.clip_start / camera_rays[:, 2, None])
    origins = (near_camera - e[:3, 3]) @ e[:3, :3]
    points, _, faces = trace(tree, origins, rays, cam.data.clip_end)
    z = (points @ e[:3, :3].T + e[:3, 3])[:, 2]
    valid = (faces >= 0) & (z >= cam.data.clip_start) & (z < cam.data.clip_end)
    z[~valid] = 0
    body = np.zeros(len(z), bool)
    body[valid] = instances[faces[valid]] > 0
    k[:2, 2] -= .5
    closure = None
    if body.any():
        projected = (points[body] @ e[:3, :3].T + e[:3, 3]) @ k.T
        closure = float(np.abs(projected[:, :2] / projected[:, 2:] - np.stack((xx, yy), -1).reshape(-1, 2)[body]).max())
    return dict(depth=z.reshape(side, side).astype(np.float32), body=body.reshape(side, side),
                valid=valid.reshape(side, side), K=k, E=e), closure


class RenderJob:
    def __init__(self, cfg, design):
        self.cfg, self.design = cfg, design
        self.out = ROOT / cfg['output']
        self.records = {0: [], 1: []}
        self.accepted = []
        self.bytes_written = 0
        self.start = time.monotonic()

    def guard(self):
        if (ROOT / self.cfg['stop']).exists():
            raise InterruptedError('Explicit STOP file')
        if dt.datetime.now(dt.timezone.utc) >= dt.datetime.fromisoformat(self.cfg['deadline_utc']):
            raise InterruptedError('Finite 48-hour deadline')
        if self.bytes_written > self.cfg['max_payload_bytes']:
            raise RuntimeError('Finite payload byte budget exceeded')

    def stage(self, stage, **kw):
        self.guard()
        faulthandler.cancel_dump_traceback_later()
        faulthandler.dump_traceback_later(300, repeat=True)
        value = dict(state='running', stage=stage, utc=utc(), windows_pid=os.getpid(),
                     job_token=self.cfg['job_token'], scenes_completed=sum(map(len, self.records.values())),
                     scenes_expected=len(self.cfg['scenes']), elapsed_seconds=time.monotonic() - self.start, **kw)
        contract.write(self.out / 'progress.json', value)
        print('V5_STAGE', stage, kw, flush=True)

    def precheck(self, env):
        rows = [r for r in self.cfg['scenes'] if r['environment'] == env['id']]
        sources = {s['id']: s for s in self.design['sources']}
        rigs = {r['id']: r for r in self.design['rigs']}
        for attempt in range(self.design['visibility']['max_resamples'] + 1):
            params = contract.environment_parameters(env, attempt)
            cache, checks, inventory, actual = {}, [], {}, None
            for source_id in sorted({r['source'] for r in rows}):
                source = sources[source_id]
                self.stage('prescreen_source', environment=env['id'], attempt=attempt, source=source_id)
                scene, inv = open_source(source, self.cfg)
                actual_here, _ = install_environment(scene, params, self.design)
                actual = actual or actual_here
                inventory[source_id] = inv
                for frame in sorted({r['frame'] for r in rows if r['source'] == source_id}):
                    self.stage('prescreen_geometry', environment=env['id'], attempt=attempt, source=source_id, frame=frame)
                    scene.frame_set(frame)
                    bpy.context.view_layer.update()
                    tree, _, _, _, instances, _, _ = geometry(scene, dict(meshes=[source['mesh']], source_depth_policy={}), 'V5_PRESCREEN')
                    cache[source_id, frame] = (tree, instances)
                    for row in [r for r in rows if r['source'] == source_id and r['frame'] == frame]:
                        cams = cameras_for(scene, rigs[row['rig']], source)
                        for view, cam in enumerate(cams):
                            self.guard()
                            payload, closure = ray_payload(scene, cam, 96, tree, instances)
                            count = int(payload['body'].sum())
                            checks.append(dict(rig=row['rig'], source=source_id, frame=frame, view=view, body_pixels=count,
                                               projection_closure_max_px=closure, passed=count >= 16 and closure is not None and closure < .003))
                        remove_cameras(cams)
            passed = all(x['passed'] for x in checks)
            receipt = dict(environment_id=env['id'], candidate_parameters=params, views=checks, passed=passed,
                           model_calls=0, RGB_renders=0, rule=self.design['visibility'],
                           source_sha256=self.cfg['source_sha256'], template_sha256=self.cfg['template_sha256'])
            if passed:
                folder = self.out / 'environments' / env['id']
                contract.write(folder / 'visibility_receipt.json', receipt)
                contract.write(folder / 'environment_parameters.json', actual)
                for source_id, inv in inventory.items():
                    contract.write(folder / f'{source_id}_inventory.json', inv)
                self.accepted.append(dict(environment=env['id'], attempt=attempt, checks=len(checks),
                                          receipt_sha256=contract.sha(folder / 'visibility_receipt.json'),
                                          parameters_sha256=contract.sha(folder / 'environment_parameters.json')))
                contract.write(self.out / 'visibility_prescreen.json', dict(state='in_progress', passed=False,
                               accepted_environments=self.accepted, complete=False, model_calls=0))
                return rows, params, cache, folder
            contract.write(self.out / 'rejected_configurations' / env['id'] / f'attempt_{attempt:02d}.json', receipt)
            print('V5_ENVIRONMENT_REJECTED', env['id'], attempt, flush=True)
        raise RuntimeError('No admissible shared environment within 32 deterministic resamples: ' + env['id'])

    def render_environment(self, rows, params, cache, folder):
        sources = {s['id']: s for s in self.design['sources']}
        rigs = {r['id']: r for r in self.design['rigs']}
        for source_id in sorted({r['source'] for r in rows}):
            source = sources[source_id]
            self.stage('render_source_open', environment=params['id'], source=source_id)
            scene, _ = open_source(source, self.cfg)
            _, rotation = install_environment(scene, params, self.design)
            pref = bpy.context.preferences.addons['cycles'].preferences
            pref.compute_device_type = 'OPTIX'
            pref.get_devices()
            for device in pref.devices:
                device.use = device.type == 'OPTIX'
            assert any(d.use for d in pref.devices), 'OPTIX GPU unavailable; no implicit CPU fallback'
            scene.cycles.device = 'GPU'
            for frame in sorted({r['frame'] for r in rows if r['source'] == source_id}):
                scene.frame_set(frame)
                bpy.context.view_layer.update()
                tree, instances = cache[source_id, frame]
                for row in [r for r in rows if r['source'] == source_id and r['frame'] == frame]:
                    rig = rigs[row['rig']]
                    rel = contract.scene_path(self.cfg['output'], row)
                    dest = ROOT / rel
                    dest.mkdir(parents=True, exist_ok=True)
                    lighting = setup_lighting(scene, row, rotation)
                    light_path = self.out / 'lighting' / row['environment'] / row['rig'] / source_id / f'frame_{frame:06d}.json'
                    contract.write(light_path, lighting)
                    rec = dict(schema=contract.SCHEMA, rig=rig['id'], split=rig['split'], dev_tier=rig.get('dev_tier'),
                               environment_id=row['environment'], environment_split=params['split'], dev_condition=row['condition'],
                               source_blend=source['path'], source_sha256=self.cfg['source_sha256'][source_id],
                               producer_script_sha256=contract.sha(Path(__file__)),
                               rig_config_sha256=hashlib.sha256(json.dumps(rig, sort_keys=True).encode()).hexdigest(),
                               environment_parameters= (folder / 'environment_parameters.json').relative_to(ROOT).as_posix(),
                               environment_parameters_sha256=contract.sha(folder / 'environment_parameters.json'),
                               visibility_receipt=(folder / 'visibility_receipt.json').relative_to(ROOT).as_posix(),
                               visibility_receipt_sha256=contract.sha(folder / 'visibility_receipt.json'),
                               lighting_parameters=light_path.relative_to(ROOT).as_posix(), lighting_sha256=contract.sha(light_path),
                               frame=frame, kind='synthetic', group=source['motion_group'], motion_id=source_id,
                               views=[], keypoints=None, apparatus_ablation=False, admitted_motion_split='train', scene=rel,
                               depth_unit='mm', depth_semantics='camera_z', mask_semantics='primary-ray opaque skin, excludes glass and hair',
                               design_sha256=self.cfg['design_sha256'], engineering_smoke=self.cfg['engineering_smoke'])
                    cams = cameras_for(scene, rig, source)
                    # Committed records can only be reused under the same immutable job pins.
                    old = contract.read(dest / 'record.json') if (dest / 'record.json').exists() else None
                    if old is not None:
                        for key in ('design_sha256', 'producer_script_sha256', 'source_sha256', 'environment_parameters_sha256', 'lighting_sha256'):
                            assert old[key] == rec[key], ('Resume pin mismatch', key)
                    for i, cam in enumerate(cams):
                        self.stage('final_truth', environment=row['environment'], rig=rig['id'], source=source_id, frame=frame, view=i)
                        start = time.monotonic()
                        payload, closure = ray_payload(scene, cam, self.cfg['resolution'], tree, instances)
                        count = int(payload['body'].sum())
                        if count < 100 or closure is None or closure >= .003:
                            contract.write(self.out / 'rejected_configurations' / 'final_failure.json', dict(row=row, view=i, body_pixels=count, closure=closure))
                            raise RuntimeError('384px geometric QA failure; refusing incomplete scene or silent resample')
                        image = dest / f'rgb_{i}.png'
                        truth = dest / f'truth_{i}.npz'
                        if old is None:
                            self.stage('rgb_render', environment=row['environment'], rig=rig['id'], source=source_id, frame=frame, view=i)
                            scene.camera = cam
                            tmp_image = dest / f'rgb_{i}.partial.png'
                            scene.render.filepath = str(tmp_image)
                            bpy.ops.render.render(write_still=True)
                            tmp_image.replace(image)
                            tmp_truth = truth.with_suffix('.partial')
                            with tmp_truth.open('wb') as f:
                                np.savez_compressed(f, **payload)
                            tmp_truth.replace(truth)
                        else:
                            for file in (image, truth):
                                assert contract.sha(file) == old['views'][i]['file_sha256'][file.name]
                        self.bytes_written += image.stat().st_size + truth.stat().st_size
                        rec['views'].append(dict(id=i, image=image.relative_to(ROOT).as_posix(), truth=truth.relative_to(ROOT).as_posix(),
                                                 K=payload['K'].tolist(), E=payload['E'].tolist(), physical_role='bottom' if i == 0 else 'side',
                                                 depth=None, mask=None, foreground_pixels=count, projection_closure_max_px=closure,
                                                 passed=True, seconds=time.monotonic() - start,
                                                 file_sha256={image.name: contract.sha(image), truth.name: contract.sha(truth)}))
                    if old is None:
                        contract.write(dest / 'record.json', rec)
                    else:
                        rec = old
                    remove_cameras(cams)
                    self.records[row['shard']].append(rec)
                    shard = self.out / f'render_shard{row["shard"]}'
                    contract.write(shard / 'records_partial.json', self.records[row['shard']])
                    contract.write(shard / 'progress.json', dict(state='rendering', records=len(self.records[row['shard']]), last_scene=rel, utc=utc()))
                    self.stage('scene_committed', last_scene=rel)
            assert contract.sha(ROOT / source['path']) == self.cfg['source_sha256'][source_id]

    def run(self):
        contract.write(self.out / 'analytic_fixture.json', fixture())
        selected = {r['environment'] for r in self.cfg['scenes']}
        for env in self.design['environments']:
            if env['id'] not in selected:
                continue
            rows, params, cache, folder = self.precheck(env)
            self.render_environment(rows, params, cache, folder)
            del cache
        assert sum(map(len, self.records.values())) == len(self.cfg['scenes'])
        contract.write(self.out / 'visibility_prescreen.json', dict(state='completed', passed=True, complete=True,
                       accepted_environments=self.accepted, checked_scenes=len(self.cfg['scenes']),
                       rule=self.design['visibility'], model_calls=0, sealed_model_calls=0))
        for shard_id in sorted({r['shard'] for r in self.cfg['scenes']}):
            config = self.out / f'render_config_shard{shard_id}.json'
            records = self.records[shard_id]
            assert len(records) == len([r for r in self.cfg['scenes'] if r['shard'] == shard_id])
            contract.write(self.out / f'render_shard{shard_id}' / 'completion.json', dict(records=records, rejected=[],
                           config_sha256=contract.sha(config), script_sha256=contract.sha(Path(__file__)),
                           code_pins=self.cfg['code_pins'], design_sha256=self.cfg['design_sha256'],
                           visibility_prescreen_sha256=contract.sha(self.out / 'visibility_prescreen.json'),
                           blender=bpy.app.version_string, policy=self.design['truth_policy'],
                           source_sha256=self.cfg['source_sha256'], template_sha256=self.cfg['template_sha256'],
                           schema=contract.SCHEMA, completed_utc=utc(), engineering_smoke=self.cfg['engineering_smoke']))
        self.stage('completed')
        progress = contract.read(self.out / 'progress.json')
        progress['state'] = 'completed'
        contract.write(self.out / 'progress.json', progress)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--job-token', required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    cfg = contract.read(ROOT / args.config)
    assert args.job_token == cfg['job_token']
    for rel, pin in cfg['code_pins'].items():
        assert contract.sha(ROOT / rel) == pin, ('Source changed', rel)
    for rel, pin in cfg['template_provenance_pins'].items():
        assert contract.sha(ROOT / rel) == pin, ('Template provenance changed', rel)
    assert contract.sha(ROOT / cfg['template_blend']) == cfg['template_sha256']
    assert contract.sha(ROOT / cfg['design']) == cfg['design_sha256']
    design = contract.read(ROOT / cfg['design'])
    job = RenderJob(cfg, design)
    faulthandler.enable()
    try:
        job.run()
    except BaseException as exc:
        contract.write(job.out / 'progress.json', dict(state='stopped' if isinstance(exc, InterruptedError) else 'failed',
                       utc=utc(), error=str(exc), traceback=traceback.format_exc(), windows_pid=os.getpid(),
                       scenes_completed=sum(map(len, job.records.values())), job_token=cfg['job_token']))
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()


if __name__ == '__main__':
    main()
