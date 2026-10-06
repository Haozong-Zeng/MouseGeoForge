"""Copy accepted backdrop/floor material data only into a static pose project."""
import bpy,json,sys,time,datetime as dt,hashlib
from pathlib import Path
import numpy as np
root=Path(sys.argv[sys.argv.index('--')+1]);base=root/'output/pymousestudio_realism_cycle_20261002';state=json.loads((base/'state.json').read_text(encoding='utf-8-sig'));appearance=Path(state['best_blend']);tag=globals().get('POSE_TAG','quadruped');out=base/'v046_pose_appearance'/(tag+'_r1');out.mkdir(parents=True,exist_ok=False);s=bpy.context.scene;source=bpy.data.filepath;mouse=bpy.data.objects['MS_Mouse']
def points():
    bpy.context.view_layer.update();ev=mouse.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=ev.to_mesh();q=np.array([ev.matrix_world@v.co for v in mesh.vertices]);ev.to_mesh_clear();return q
before=points();cams={c:[list(r) for r in bpy.data.objects['Replica_FR2_cam'+c].matrix_world] for c in 'ABCDEFG'}
names=['Replica_Backdrop_v044_b_matte_principled'];is_smudged='v045-' in appearance.name
if is_smudged:names.append('Replica_FloorSmudges_v045')
with bpy.data.libraries.load(str(appearance),link=False) as (data_from,data_to):
    assert all(n in data_from.materials for n in names),(names,data_from.materials);data_to.materials=list(names)
loaded=dict(zip(names,data_to.materials));targets={'Part_WhiteHexagonBackground':'Replica_Backdrop_v044_b_matte_principled'}
if is_smudged:targets['Part_AntiReflectiveGlass']='Replica_FloorSmudges_v045'
for obj,name in targets.items():
    for slot in bpy.data.objects[obj].material_slots:slot.material=loaded[name]
after=points();error=float(np.linalg.norm(after-before,axis=1).max());assert error==0
for c,matrix in cams.items():assert matrix==[list(r) for r in bpy.data.objects['Replica_FR2_cam'+c].matrix_world]
prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='OPTIX';prefs.get_devices();devices=[]
for d in prefs.devices:
    d.use=d.type=='OPTIX' and 'RTX PRO 6000' in d.name
    if d.use:devices.append(d.name)
assert devices;s.cycles.device='GPU';s.cycles.samples=128;s.cycles.use_denoising=True;s.render.resolution_x=1080;s.render.resolution_y=1200;s.render.resolution_percentage=100;s.render.use_persistent_data=True
m=dict(source=source,appearance_source=str(appearance),appearance_source_sha256=hashlib.sha256(appearance.read_bytes()).hexdigest(),tag=tag,frame=s.frame_current,materials=targets,skin_max_change_mm=error,skin_min_z=float(after[:,2].min()),wall_vertices=int((np.linalg.norm(after[:,:2],axis=1)>56.470588+.001).sum()),floor_vertices=int((after[:,2]<339.5-.001).sum()),camera_matrices_unchanged=True,devices=devices,renders=[],limitations=['Independent static pose. Materials only, no calibration/pose/lighting changes.'])
s['replica_appearance_sync_source']=str(appearance);target=root/f'BlenderPyMouseProjects/Cylinder/PyMouseStudio-RealCapture-Replica-20261002/PyMouseStudio-RealCapture-Replica-v046-{tag}-appearance.blend';assert not target.exists();bpy.ops.wm.save_as_mainfile(filepath=str(target),compress=True);m['blend']=str(target)
def write():(out/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2))
write()
for c in 'ABDF':
    assert dt.datetime.now(dt.timezone.utc)<dt.datetime(2026,10,3,6,59,16,tzinfo=dt.timezone.utc)
    s.camera=bpy.data.objects['Replica_FR2_cam'+c];r=float(s.camera['replica_pixel_aspect_y_over_x']);s.render.pixel_aspect_x=max(1.,1/r);s.render.pixel_aspect_y=max(1.,r);s.render.filepath=str(out/f'cam{c}.png');t=time.monotonic();bpy.ops.render.render(write_still=True);m['renders'].append(dict(camera=c,file=s.render.filepath,seconds=time.monotonic()-t));write()
