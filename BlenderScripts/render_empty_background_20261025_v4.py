"""CPU-only empty-background diagnostics; source .blend files are never saved."""
import argparse
import datetime as dt
import json
from pathlib import Path
import sys
import time

import bpy
import numpy as np
from mathutils import Vector

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'));sys.path.insert(0,str(ROOT/'BlenderScripts'))
import v5_render_contract_20261024 as contract
from render_v5_complex_20261024 import open_source,install_environment,setup_lighting,cameras_for,remove_cameras
from audit_visible_surface_geometry_20261007 import camera_parameters


def mouse_family(scene,source):
    body=scene.objects[source['mesh']]
    bodies=[o for o in scene.objects if o.type=='MESH' and o.name.startswith('MS_Mouse')]
    rigs={mod.object for mod in body.modifiers if mod.type=='ARMATURE' and mod.object}
    for candidate in bodies:
        rigs.update(mod.object for mod in candidate.modifiers if mod.type=='ARMATURE' and mod.object)
        if candidate.parent and candidate.parent.type=='ARMATURE':rigs.add(candidate.parent)
    reasons={body.name:'declared source mouse mesh'}
    for candidate in bodies:
        for collection in candidate.users_collection:
            if collection.name.startswith('Mouse_'):
                assert len(collection.all_objects)<=64,'Unusually broad mouse collection'
                for obj in collection.all_objects:reasons[obj.name]='member of explicit mouse collection '+collection.name
    for owner in [*bodies,*rigs]:
        reasons[owner.name]='body or mouse armature'
        for child in owner.children_recursive:reasons[child.name]='descendant of mouse mesh/armature'
    for obj in scene.objects:
        if obj.type=='CURVE' and obj.name.startswith(('Spine_NURBSCurve','Tail_NURBSCurve')):
            reasons[obj.name]='mouse pose guide curve'
        if any(mod.type=='ARMATURE' and mod.object in rigs for mod in obj.modifiers):
            reasons[obj.name]='bound to same mouse armature'
        surface=getattr(obj.data,'surface',None) if obj.data else None
        if surface in bodies:reasons[obj.name]='hair surface refers to source mouse'
    return reasons


def bounds(obj):
    xyz=np.array([obj.matrix_world@Vector(p) for p in obj.bound_box])
    return xyz.min(0),xyz.max(0)


def inspect(scene,source):
    family=mouse_family(scene,source);body=scene.objects[source['mesh']]
    low,high=bounds(body);near=[]
    for obj in scene.objects:
        if obj.type!='MESH' or obj.hide_render or obj.name in family:continue
        a,b=bounds(obj)
        if ((b>=low-3)&(a<=high+3)).all():near.append(obj.name)
    return dict(source=source['id'],mouse=source['mesh'],mouse_family=family,
        collections={c.name:[o.name for o in c.all_objects] for c in body.users_collection},
        curves=[dict(name=o.name,parent=o.parent.name if o.parent else None,
            surface=getattr(getattr(o.data,'surface',None),'name',None),hide_render=o.hide_render)
                for o in scene.objects if o.type in ['CURVES','CURVE']],
        nearby_nonfamily_meshes=near,
        body_world_bounds=[low.tolist(),high.tolist()])


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--job-token',required=True)
    p.add_argument('--command',choices=['inspect','render'],required=True)
    args=p.parse_args(sys.argv[sys.argv.index('--')+1:])
    cfg=contract.read(ROOT/args.config);assert cfg['job_token']==args.job_token
    for path,pin in cfg['code_pins'].items():assert contract.sha(ROOT/path)==pin
    design=contract.read(ROOT/cfg['design']);out=ROOT/cfg['output'];out.mkdir(exist_ok=True)
    deadline=dt.datetime.fromisoformat(cfg['deadline_utc'])
    def guard():
        if dt.datetime.now(dt.timezone.utc)>=deadline or (out/'STOP').exists():raise InterruptedError('Finite diagnostic deadline/STOP')
    inspection=[];records=[]
    for source in design['sources']:
        guard();scene,_=open_source(source,cfg);scene.frame_set(49);bpy.context.view_layer.update()
        info=inspect(scene,source);inspection.append(info)
        if args.command=='inspect':continue
        allowed=cfg['approved_mouse_family'][source['id']]
        assert set(info['mouse_family'])==set(allowed),'Source mouse family changed; no automatic broad hiding'
        # Only mouse-owned objects change visibility; equipment remains as in V5.
        for name in allowed:scene.objects[name].hide_render=True
        bpy.context.view_layer.update()
        assert all(scene.objects[name].hide_render for name in allowed)
        assert not [o.name for o in scene.objects if o.type=='MESH' and o.name.startswith('MS_Mouse') and not o.hide_render]
        source_rows=[r for r in cfg['scenes'] if r['source']==source['id']]
        for row in source_rows:
            guard()
            # Reopen for each environment so transforms/material gains cannot accumulate.
            scene,_=open_source(source,cfg);scene.frame_set(row['frame'])
            for name in allowed:scene.objects[name].hide_render=True
            params=contract.read(ROOT/row['environment_parameters'])['parameters']
            actual,rotation=install_environment(scene,params,design)
            lighting=setup_lighting(scene,row,rotation)
            scene.cycles.device='CPU';scene.render.threads_mode='FIXED';scene.render.threads=cfg['cpu_threads']
            scene.render.use_persistent_data=True
            cameras=cameras_for(scene,next(r for r in design['rigs'] if r['id']==row['rig']),source)
            folder=out/row['id'];folder.mkdir(exist_ok=False);views=[]
            before_visible={o.name for o in scene.objects if o.type=='MESH' and not o.hide_render}
            assert source['mesh'] not in before_visible
            for i,camera in enumerate(cameras):
                guard();scene.camera=camera
                k,e=camera_parameters(scene,camera,cfg['resolution'],cfg['resolution']);k[:2,2]-=.5
                old=contract.read(ROOT/row['positive_record'])['views'][i]
                np.testing.assert_allclose(k,old['K'],atol=1e-8);np.testing.assert_allclose(e,old['E'],atol=1e-8)
                scene.render.filepath=str(folder/f'rgb_{i}.png');began=time.monotonic()
                contract.write(out/'progress.json',dict(stage='rendering',scene=row['id'],view=i,windows_pid=__import__('os').getpid(),job_token=args.job_token,device='CPU'))
                bpy.ops.render.render(write_still=True)
                views.append(dict(view=i,image=(folder/f'rgb_{i}.png').relative_to(ROOT).as_posix(),
                    sha256=contract.sha(folder/f'rgb_{i}.png'),K=k.tolist(),E=e.tolist(),
                    seconds=time.monotonic()-began,device=scene.cycles.device))
            rec=dict(id=row['id'],split='diagnostic_dev',views=views,positive_record=row['positive_record'],
                source_sha256=cfg['source_sha256'][source['id']],hidden_mouse_family=allowed,
                remaining_visible_meshes=sorted(before_visible),mouse_instance_count=0,
                zero_foreground_policy='All source mouse-family objects hidden for rendering; no inpainting/cropping/erasure',
                environment_sha256=contract.sha(ROOT/row['environment_parameters']),lighting=lighting,
                original_blend_saved=False,training_admission=False)
            contract.write(folder/'record.json',rec);records.append(rec);remove_cameras(cameras)
    if args.command=='inspect':contract.write(out/'source_inspection.json',dict(sources=inspection))
    else:contract.write(out/'completion.json',dict(passed=True,records=records,device='CPU',
        blender_version=bpy.app.version_string,job_token=args.job_token,model_calls=0,
        code_pins=cfg['code_pins'],diagnostic_only=True,training_admission=False))


if __name__=='__main__':main()
