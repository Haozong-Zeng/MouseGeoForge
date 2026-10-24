"""Finite render-only v3 dataset. Original .blend files are never saved.

Configuration explicitly admits sources/splits. Camera geometry is the same
relative rig across admitted motion scenes, translated to a fixed scene anchor.
Sealed records are rendered and geometrically checked, never model-scored here.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import bpy
import faulthandler
import numpy as np
from mathutils import Vector,Quaternion

sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit_visible_surface_geometry_20261007 import camera_parameters
from export_visible_surface_truth_20261007 import geometry,trace,fixture


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');tmp.replace(p)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--config',required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);root=Path(args.root);cfg=json.loads(Path(args.config).read_text())
    out=root/cfg['output'];out.mkdir(parents=True,exist_ok=True);assert not (out/'completion.json').exists()
    faulthandler.enable()
    def stage(name,**kw):
        faulthandler.cancel_dump_traceback_later();faulthandler.dump_traceback_later(120,repeat=True)
        write(out/'progress.json',dict(stage=name,utc=dt.datetime.now(dt.timezone.utc).isoformat(),**kw))
        print('V3_STAGE',name,kw,flush=True)
    stage('startup')
    deadline=dt.datetime.fromisoformat(cfg['deadline_utc']);records=[];rejects=[]
    write(out/'analytic_fixture.json',fixture())
    for source in cfg['sources']:
        assert source['admitted_split'] in ('train','val')
        if source['admitted_split']=='train':
            assert 'WalkAround' not in source['path'] and 'WashFace' not in source['path']
        path=root/source['path'];stage('source_hash',source=source['id']);digest=sha(path);stage('source_open',source=source['id']);bpy.ops.wm.open_mainfile(filepath=str(path));scene=bpy.context.scene
        for collection in bpy.data.collections:
            if collection.name.startswith('CameraItems'):collection.hide_render=True
        hidden=[]
        for obj in bpy.data.objects:
            if obj.type=='CURVES':obj.hide_viewport=True
            if obj.name.startswith(('Part_Camera','Part_CamLen','CameraItems','Part_PiCam','Part_CNC-Camera','Part_BottomCamScrew','Part_SideCamScrew')):
                obj.hide_render=True;hidden.append(obj.name)
                for child in obj.children_recursive:child.hide_render=True;hidden.append(child.name)
        if source.get('apparatus_ablation',False):
            # Registered synthetic variant: retain admitted animated body/fur,
            # lighting and world; remove opaque apparatus instead of pretending
            # its occluded pixels contain observed bottom-view body.
            for obj in scene.objects:
                if obj.type=='MESH' and obj.name!=source['mesh']:
                    obj.hide_render=True;hidden.append(obj.name)
        assert source['mesh'] in bpy.data.objects and not bpy.data.objects[source['mesh']].hide_render
        scene.render.engine='CYCLES';scene.cycles.samples=8;scene.cycles.use_denoising=True
        pref=bpy.context.preferences.addons['cycles'].preferences;pref.compute_device_type='OPTIX';pref.get_devices()
        for device in pref.devices:device.use=device.type=='OPTIX'
        assert any(d.use for d in pref.devices);scene.cycles.device='GPU'
        side=cfg['resolution'];scene.render.resolution_x=side;scene.render.resolution_y=side;scene.render.resolution_percentage=100
        scene.render.pixel_aspect_x=1;scene.render.pixel_aspect_y=1;scene.render.use_compositing=False;scene.render.use_sequencer=False
        scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGB';scene.render.image_settings.color_depth='8';scene.render.use_persistent_data=True
        target=Vector(source['anchor']);cameras=[]
        write(out/(source['id']+'_inventory.json'),dict(source_sha256=digest,hidden_equipment=hidden,body_mesh=source['mesh'],
            visible_meshes=[o.name for o in scene.objects if o.type=='MESH' and not o.hide_render]))
        for rig in cfg['rigs']:
            if source['admitted_split']=='val' and rig['split']=='train':continue
            if source['admitted_split']=='train' and rig.get('holdout_motion_only',False):continue
            if (root/cfg['stop']).exists() or dt.datetime.now(dt.timezone.utc)>=deadline:raise RuntimeError('Finite render deadline or STOP')
            cams=[]
            for i,relative in enumerate(rig['relative_positions']):
                data=bpy.data.cameras.new(f'V3_{rig["id"]}_{i}');data.type='PERSP';data.lens=rig['lens_mm'];data.sensor_width=36;data.sensor_fit='HORIZONTAL';data.clip_start=.1;data.clip_end=3000
                camera=bpy.data.objects.new(data.name,data);scene.collection.objects.link(camera);camera.location=target+Vector(relative)
                camera.rotation_mode='QUATERNION';camera.rotation_quaternion=(target-camera.location).to_track_quat('-Z','Y')@Quaternion((0,0,1),math.radians(rig['roll_deg'][i]));cams.append(camera)
            bpy.context.view_layer.update()
            rig_digest=hashlib.sha256(json.dumps(rig,sort_keys=True).encode()).hexdigest()
            for frame in source['frames']:
                if (root/cfg['stop']).exists() or dt.datetime.now(dt.timezone.utc)>=deadline:raise RuntimeError('Finite render deadline or STOP')
                folder=out/rig['id']/source['id']/f'frame_{frame:06d}'
                receipt=folder/'record.json'
                if receipt.exists():
                    row=json.loads(receipt.read_text());assert row['source_sha256']==digest
                    if 'rig_config_sha256'in row:assert row['rig_config_sha256']==rig_digest
                    for view,cam in zip(row['views'],cams):
                        kk,ee=camera_parameters(scene,cam,side,side);kk[:2,2]-=.5
                        np.testing.assert_allclose(view['K'],kk,atol=.0001,rtol=1e-6)
                        np.testing.assert_allclose(view['E'],ee,atol=.0002,rtol=1e-6)
                    row.setdefault('producer_script_sha256',cfg['accepted_legacy_script_sha256'])
                    row['rig_config_sha256']=rig_digest;records.append(row);continue
                folder.mkdir(parents=True,exist_ok=True);stage('frame_set',rig=rig['id'],source=source['id'],frame=frame);scene.frame_set(frame);bpy.context.view_layer.update();stage('record_setup',rig=rig['id'],source=source['id'],frame=frame)
                rec=dict(rig=rig['id'],split=rig['split'],dev_tier=rig.get('dev_tier'),source_blend=source['path'],source_sha256=digest,
                    producer_script_sha256=sha(Path(__file__)),rig_config_sha256=rig_digest,
                    frame=frame,kind='synthetic',group=source['motion_group'],motion_id=source['id'],views=[],keypoints=None,
                    apparatus_ablation=source.get('apparatus_ablation',False),admitted_motion_split=source['admitted_split'],scene=folder.relative_to(root).as_posix())
                stage('geometry',rig=rig['id'],source=source['id'],frame=frame)
                tree,_,_,_,instances,_,_=geometry(scene,dict(meshes=[source['mesh']],source_depth_policy={}),cams[0].name)
                yy,xx=np.mgrid[:side,:side];uv=np.stack((xx+.5,yy+.5,np.ones_like(xx)),-1).reshape(-1,3)
                for i,cam in enumerate(cams):
                    stage('camera_setup',rig=rig['id'],source=source['id'],frame=frame,view=i)
                    started=time.monotonic();scene.camera=cam;bpy.context.view_layer.update();K,E=camera_parameters(scene,cam,side,side)
                    image=folder/f'rgb_{i}.png';scene.render.filepath=str(image);stage('rgb_render',rig=rig['id'],source=source['id'],frame=frame,view=i);bpy.ops.render.render(write_still=True)
                    stage('truth_trace',rig=rig['id'],source=source['id'],frame=frame,view=i)
                    rays=(uv@np.linalg.inv(K).T)@E[:3,:3];rays/=np.linalg.norm(rays,axis=1,keepdims=True)
                    center=-E[:3,:3].T@E[:3,3];points,_,faces=trace(tree,center,rays,cam.data.clip_end)
                    z=(points@E[:3,:3].T+E[:3,3])[:,2];valid=(faces>=0)&(z>0);z[~valid]=0
                    iid=np.zeros(len(z),np.int16);iid[valid]=instances[faces[valid]];body=iid>0
                    K[:2,2]-=.5
                    if body.any():
                        projected=(points[body]@E[:3,:3].T+E[:3,3])@K.T;projected=projected[:,:2]/projected[:,2:]
                        error=float(np.abs(projected-np.stack((xx,yy),-1).reshape(-1,2)[body]).max())
                    else:error=None
                    passed=int(body.sum())>=cfg.get('min_body_pixels',100) and error is not None and error<.003
                    truth=folder/f'truth_{i}.npz';np.savez_compressed(truth,depth=z.reshape(side,side),body=body.reshape(side,side),valid=valid.reshape(side,side),K=K,E=E)
                    view=dict(id=i,image=image.relative_to(root).as_posix(),truth=truth.relative_to(root).as_posix(),K=K.tolist(),E=E.tolist(),
                        physical_role='bottom' if i==0 else 'side',depth=None,mask=None,foreground_pixels=int(body.sum()),projection_closure_max_px=error,passed=passed,seconds=time.monotonic()-started)
                    rec['views'].append(view)
                    write(out/'status.json',dict(state='rendering',source=source['id'],rig=rig['id'],frame=frame,view=i,completed_scenes=len(records),rejected_scenes=len(rejects),last_view=view))
                    print('V3_RENDER',source['id'],rig['id'],frame,i,int(body.sum()),round(view['seconds'],2),flush=True)
                if all(v['passed'] for v in rec['views']):write(receipt,rec);records.append(rec)
                else:
                    write(folder/'rejected.json',rec);rejects.append(rec)
                    if not cfg.get('smoke_allow_rejected',False):raise RuntimeError(f'Geometry QA failed: {source["id"]}/{rig["id"]}/{frame}')
                write(out/'records_partial.json',records)
            for cam in cams:bpy.data.objects.remove(cam,do_unlink=True)
        assert sha(path)==digest
    write(out/'completion.json',dict(records=records,rejected=rejects,config_sha256=sha(Path(args.config)),script_sha256=sha(Path(__file__)),blender=bpy.app.version_string,
        policy='Cycles RGB; primary-ray camera-Z skin truth; glass/hair excluded from truth; only camera equipment hidden in RGB; original scenes never saved.'))
    write(out/'status.json',dict(state='completed',scenes=len(records),rejected=len(rejects)));faulthandler.cancel_dump_traceback_later()


if __name__=='__main__':main()
