"""Local quadruped head elevation and shallow tail-floor correction, no body scale changes."""
import bpy,sys,json,math,time,datetime as dt
import numpy as np
from pathlib import Path
from mathutils import Matrix,Vector,Quaternion
root=Path(sys.argv[sys.argv.index('--')+1]);base=root/'output/pymousestudio_realism_cycle_20261002';out=base/'v026_quadruped_head';out.mkdir(exist_ok=False)
s=bpy.context.scene;s.frame_set(111);arm=bpy.data.objects['DeformationSystem'];mouse=bpy.data.objects['MS_Mouse'];inv=arm.matrix_world.inverted();source=bpy.data.filepath
names=[f'Tail{i}_M' for i in range(12)];world={n:arm.matrix_world@arm.pose.bones[n].matrix for n in names};old=np.array([world[n].translation for n in names]);lengths=np.linalg.norm(np.diff(old,axis=0),axis=1);points=old.copy()
for i in range(11):
    z=old[i+1,2]+.5;dz=z-points[i,2];assert abs(dz)<lengths[i];d=old[i+1,:2]-old[i,:2];d/=np.linalg.norm(d)
    points[i+1,:2]=points[i,:2]+d*math.sqrt(lengths[i]**2-dz**2);points[i+1,2]=z
for i,n in enumerate(names):
    a=old[min(i+1,11)]-old[max(i-1,0)];b=points[min(i+1,11)]-points[max(i-1,0)];rot=Vector(a).rotation_difference(Vector(b)).to_matrix().to_4x4()
    arm.pose.bones[n].matrix=inv@Matrix.Translation(Vector(points[i]))@rot@Matrix.Translation(Vector(-old[i]))@world[n];bpy.context.view_layer.update()
head=arm.pose.bones['Head1_M'];pivot=arm.matrix_world@head.head;forward=arm.matrix_world@arm.pose.bones['HeadEnd_M'].head-pivot;axis=forward.cross(Vector((0,0,1))).normalized();rotation=Quaternion(axis,math.radians(12)).to_matrix().to_4x4()
head.matrix=inv@Matrix.Translation(pivot)@rotation@Matrix.Translation(-pivot)@arm.matrix_world@head.matrix;bpy.context.view_layer.update()
ev=mouse.evaluated_get(bpy.context.evaluated_depsgraph_get());me=ev.to_mesh();p=np.array([ev.matrix_world@v.co for v in me.vertices]);ev.to_mesh_clear()
groups={g.index:g.name for g in mouse.vertex_groups};feet={}
for side in 'LR':
    for kind,parts in [('fore',['Finger','Wrist']),('hind',['Toe','Ankle'])]:
        ids=[v.index for v in mouse.data.vertices if sum(g.weight for g in v.groups if groups[g.group].endswith('_'+side) and any(t in groups[g.group] for t in parts))>.65];feet[kind+side]=float(p[ids,2].min())
after=np.array([arm.matrix_world@arm.pose.bones[n].head for n in names]);geometry=dict(min_z=float(p[:,2].min()),floor_z=339.5,vertices_below_floor=int((p[:,2]<339.5).sum()),wall_vertices=int((np.linalg.norm(p[:,:2],axis=1)>56.470588).sum()),feet_min_z=feet,tail_length_error_mm=float(np.max(np.abs(np.linalg.norm(np.diff(after,axis=0),axis=1)-lengths))))
assert geometry['wall_vertices']==0 and geometry['vertices_below_floor']==0,geometry
prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='OPTIX';prefs.get_devices();devices=[]
for d in prefs.devices:
    d.use=d.type=='OPTIX' and 'RTX PRO 6000' in d.name
    if d.use:devices.append(d.name)
assert devices;s.cycles.device='GPU';s.cycles.samples=128;s.cycles.use_denoising=True;s.render.resolution_percentage=100;s.render.use_persistent_data=True
target=root/'BlenderPyMouseProjects/Cylinder/PyMouseStudio-RealCapture-Replica-20261002/PyMouseStudio-RealCapture-Replica-v026-quadruped-head.blend';assert not target.exists();bpy.ops.wm.save_as_mainfile(filepath=str(target))
m=dict(source=source,blend=str(target),frame=111,pose='independent static quadruped, not continuous animation',head_pitch_degrees=12,tail_clearance_lift_mm=.5,geometry=geometry,devices=devices,size=[1080,1200],samples=128,renders=[])
def write():(out/'manifest.json').write_text(json.dumps(m,indent=2))
write();print('V026_GEOMETRY',json.dumps(geometry),flush=True)
for c in 'ABDF':
    assert dt.datetime.now(dt.timezone.utc)<dt.datetime(2026,10,3,6,59,16,tzinfo=dt.timezone.utc)
    s.camera=bpy.data.objects['Replica_FR2_cam'+c];r=float(s.camera['replica_pixel_aspect_y_over_x']);s.render.pixel_aspect_x=max(1.,1/r);s.render.pixel_aspect_y=max(1.,r);s.render.filepath=str(out/f'cam{c}.png');t=time.monotonic();bpy.ops.render.render(write_still=True);m['renders'].append(dict(camera=c,file=s.render.filepath,seconds=time.monotonic()-t));write()
