"""Integrate accepted dust and registered bottom-camera proxy with exact pose regression."""
import bpy,json,sys,time,datetime as dt,hashlib
from pathlib import Path
import numpy as np
root=Path(sys.argv[sys.argv.index('--')+1]);base=root/'output/pymousestudio_realism_cycle_20261002';project=root/'BlenderPyMouseProjects/Cylinder/PyMouseStudio-RealCapture-Replica-20261002';tag=globals().get('POSE_TAG','quadruped');out=base/'v056_common_scene'/tag;out.mkdir(parents=True,exist_ok=False);s=bpy.context.scene;source=bpy.data.filepath;mouse=bpy.data.objects['MS_Mouse'];dust_source=project/'PyMouseStudio-RealCapture-Replica-v053-a_sparse004.blend';proxy_source=project/'PyMouseStudio-RealCapture-Replica-v055-bottom-receiver.blend'
def points():
    bpy.context.view_layer.update();ev=mouse.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=ev.to_mesh();q=np.array([ev.matrix_world@v.co for v in mesh.vertices]);ev.to_mesh_clear();return q
before=points();cams={c:[list(r) for r in bpy.data.objects['Replica_FR2_cam'+c].matrix_world] for c in 'ABCDEFG'};lights={o.name:dict(energy=o.data.energy,matrix=[list(r) for r in o.matrix_world]) for o in s.objects if o.type=='LIGHT'}
if bpy.data.objects['Part_AntiReflectiveGlass'].material_slots[0].material.name!='Replica_SparseFloorDust_v053':
    with bpy.data.libraries.load(str(dust_source),link=False) as (src,dst):
        assert 'Replica_SparseFloorDust_v053' in src.materials;dst.materials=['Replica_SparseFloorDust_v053']
    bpy.data.objects['Part_AntiReflectiveGlass'].material_slots[0].material=dst.materials[0]
assert 'Replica_BottomCamera_VisualProxy' not in bpy.data.collections
with bpy.data.libraries.load(str(proxy_source),link=False) as (src,dst):
    assert 'Replica_BottomCamera_VisualProxy' in src.collections;dst.collections=['Replica_BottomCamera_VisualProxy']
co=dst.collections[0];s.collection.children.link(co);renamed=[]
for o in co.all_objects:
    new=o.name.replace('Replica_Aase','Replica_Case').replace('Replica_Aable','Replica_Cable')
    if new!=o.name:renamed.append([o.name,new]);o.name=new
bpy.data.objects['Part_PiCamBottom'].hide_render=True;after=points();error=float(np.linalg.norm(after-before,axis=1).max());assert error==0
for c,x in cams.items():assert x==[list(r) for r in bpy.data.objects['Replica_FR2_cam'+c].matrix_world]
for n,x in lights.items():assert x==dict(energy=bpy.data.objects[n].data.energy,matrix=[list(r) for r in bpy.data.objects[n].matrix_world])
prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='OPTIX';prefs.get_devices();devices=[]
for d in prefs.devices:
    d.use=d.type=='OPTIX' and 'RTX PRO 6000' in d.name
    if d.use:devices.append(d.name)
assert devices;s.cycles.device='GPU';s.cycles.samples=128;s.cycles.use_denoising=True;s.render.resolution_x=1080;s.render.resolution_y=1200;s.render.resolution_percentage=100;s.render.use_persistent_data=True
target=project/f'PyMouseStudio-RealCapture-Replica-v056-{tag}-common-scene.blend';assert not target.exists();s['replica_common_scene_version']='v056: sparse floor dust +registered bottom camera receiver';bpy.ops.wm.save_as_mainfile(filepath=str(target),compress=True)
m=dict(source=source,blend=str(target),tag=tag,frame=s.frame_current,skin_max_change_mm=error,skin_min_z=float(after[:,2].min()),floor_vertices=int((after[:,2]<339.5-.001).sum()),wall_vertices=int((np.linalg.norm(after[:,:2],axis=1)>56.470588+.001).sum()),camera_matrices_unchanged=True,lights_unchanged=True,source_assets={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [dust_source,proxy_source]},renamed=renamed,proxy_meshes=len([o for o in co.all_objects if o.type=='MESH']),proxy_caustic_receivers=sum(o.cycles.is_caustics_receiver for o in co.all_objects if o.type=='MESH'),devices=devices,renders=[],limitations=['Three independent static poses; no temporal continuity implied.','Skin vertex checks do not certify complete hair/whisker/polygon collisions.','Bottom camera housing and dust appearance are physical scene proxies, not exact recovered hardware.'])
def write():(out/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2))
write()
for c in 'ABDF':
    assert dt.datetime.now(dt.timezone.utc)<dt.datetime(2026,10,3,6,59,16,tzinfo=dt.timezone.utc)
    s.camera=bpy.data.objects['Replica_FR2_cam'+c];r=float(s.camera['replica_pixel_aspect_y_over_x']);s.render.pixel_aspect_x=max(1.,1/r);s.render.pixel_aspect_y=max(1.,r);s.render.filepath=str(out/f'cam{c}.png');t=time.monotonic();bpy.ops.render.render(write_still=True);m['renders'].append(dict(camera=c,file=s.render.filepath,seconds=time.monotonic()-t));write()
