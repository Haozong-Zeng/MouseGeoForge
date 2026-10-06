"""Transfer a validated static rig pose into current optics/lighting with regression checks."""
import bpy,json,sys,time,datetime as dt,hashlib
from pathlib import Path
import numpy as np
from mathutils import Matrix
root=Path(sys.argv[sys.argv.index('--')+1]);base=root/'output/pymousestudio_realism_cycle_20261002';tag=globals().get('POSE_TAG','quadruped');payload_dir=base/'v041_pose_integration'/tag;payload=json.loads((payload_dir/'pose.json').read_text(encoding='utf-8'));out=payload_dir/'integrated';out.mkdir(exist_ok=False);s=bpy.context.scene;source=bpy.data.filepath
assert hashlib.sha256(Path(payload['source']).read_bytes()).hexdigest()==payload['source_sha256']
s.frame_set(payload['frame']);arm=bpy.data.objects['DeformationSystem'];mouse=bpy.data.objects['MS_Mouse']
for name,record in payload['objects'].items():
    ob=bpy.data.objects[name];assert (ob.parent.name if ob.parent else None)==record['parent'];assert ob.parent_type==record['parent_type'] and ob.parent_bone==record['parent_bone']
    if ob.animation_data:
        ob.animation_data.action=None
        for track in ob.animation_data.nla_tracks:track.mute=True
    ob.rotation_mode=record['rotation_mode'];ob.matrix_parent_inverse=Matrix(record['matrix_parent_inverse']);ob.matrix_basis=Matrix(record['matrix_basis'])
for k,v in payload['arm_props'].items():arm[k]=v
for name,record in payload['bones'].items():
    pb=arm.pose.bones[name];assert len(pb.constraints)==len(record['constraints']);pb.rotation_mode=record['rotation_mode']
    for k,v in record['props'].items():pb[k]=v
    pb.matrix_basis=Matrix(record['basis'])
for name,mods in payload['armature_modifiers'].items():
    for key,record in mods.items():
        mod=bpy.data.objects[name].modifiers[key];mod.use_deform_preserve_volume=record['preserve_volume'];mod.show_viewport=record['show_viewport'];mod.show_render=record['show_render']
bpy.context.view_layer.update()
bone_error={n:float(np.linalg.norm(np.array(arm.matrix_world@arm.pose.bones[n].head)-r['head_world'])) for n,r in payload['bones'].items()}
transform_error={n:max(abs(bpy.data.objects[n].matrix_world[i][j]-r['matrix_world'][i][j]) for i in range(4) for j in range(4)) for n,r in payload['objects'].items()}
ev=mouse.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=ev.to_mesh();points=np.array([ev.matrix_world@v.co for v in mesh.vertices]);ev.to_mesh_clear();ref=np.load(payload_dir/'skin_reference.npz')['xyz_mm'];assert points.shape==ref.shape;errors=np.linalg.norm(points-ref,axis=1)
groups={g.index:g.name for g in mouse.vertex_groups};feet={}
for side in 'LR':
    for kind,parts in [('fore',['Finger','Wrist']),('hind',['Toe','Ankle'])]:
        ids=[v.index for v in mouse.data.vertices if sum(g.weight for g in v.groups if groups[g.group].endswith('_'+side) and any(t in groups[g.group] for t in parts))>.65];feet[kind+side]=dict(min_z_mm=float(points[ids,2].min()),floor_gap_mm=float(points[ids,2].min()-339.5))
geometry=dict(skin_vertex_count=len(points),skin_max_error_mm=float(errors.max()),skin_mean_error_mm=float(errors.mean()),bone_head_max_error_mm=max(bone_error.values()),object_matrix_max_error=max(transform_error.values()),min_skin_z_mm=float(points[:,2].min()),vertices_below_floor=int((points[:,2]<339.5-.001).sum()),wall_vertices=int((np.linalg.norm(points[:,:2],axis=1)>56.470588+.001).sum()),feet=feet)
m=dict(scene_source=source,pose_source=payload['source'],pose_source_sha256=payload['source_sha256'],tag=tag,frame=payload['frame'],geometry=geometry,bone_errors_mm=bone_error,object_transform_errors=transform_error,armature_modifiers=payload['armature_modifiers'],renders=[],limitations=['Independent static pose, not a temporal sequence or exact real-frame reconstruction.','Geometry regression checks all evaluated skin vertices but do not certify every hair/whisker intersection.','Current scene optics and lighting retained; original camera matrices/calibration not changed.'])
def write():(out/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2))
write();assert geometry['skin_max_error_mm']<.003 and geometry['bone_head_max_error_mm']<.003 and geometry['object_matrix_max_error']<.003,geometry
assert geometry['vertices_below_floor']==0 and geometry['wall_vertices']==0,geometry
s.frame_start=payload['frame'];s.frame_end=payload['frame'];s['replica_pose_reference']=payload['scene_props'].get('replica_pose_reference',tag+' independent static pose');s['replica_source_pose_frame']=payload['frame'];s['replica_integration_pose_source']=payload['source'];s['replica_static_not_temporal']=True
prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='OPTIX';prefs.get_devices();devices=[]
for d in prefs.devices:
    d.use=d.type=='OPTIX' and 'RTX PRO 6000' in d.name
    if d.use:devices.append(d.name)
assert devices;s.cycles.device='GPU';s.cycles.samples=128;s.cycles.use_denoising=True;s.render.resolution_x=1080;s.render.resolution_y=1200;s.render.resolution_percentage=100;s.render.use_persistent_data=True;m['devices']=devices;m['resolution']=[1080,1200];m['samples']=128
target=root/f'BlenderPyMouseProjects/Cylinder/PyMouseStudio-RealCapture-Replica-20261002/PyMouseStudio-RealCapture-Replica-v041-{tag}-integrated.blend';assert not target.exists();bpy.ops.wm.save_as_mainfile(filepath=str(target),compress=True);m['blend']=str(target);write();print('POSE_INTEGRATION',json.dumps(geometry),flush=True)
for c in 'ABDF':
    assert dt.datetime.now(dt.timezone.utc)<dt.datetime(2026,10,3,6,59,16,tzinfo=dt.timezone.utc)
    s.camera=bpy.data.objects['Replica_FR2_cam'+c];r=float(s.camera['replica_pixel_aspect_y_over_x']);s.render.pixel_aspect_x=max(1.,1/r);s.render.pixel_aspect_y=max(1.,r);s.render.filepath=str(out/f'cam{c}.png');t=time.monotonic();bpy.ops.render.render(write_still=True);m['renders'].append(dict(camera=c,file=s.render.filepath,seconds=time.monotonic()-t));write()
