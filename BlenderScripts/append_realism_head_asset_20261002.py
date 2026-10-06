"""Attach detached, head-local hardware only; never import source animation."""
import bpy,json,sys,time,datetime as dt
from pathlib import Path
from mathutils import Matrix
root=Path(sys.argv[sys.argv.index('--')+1]);base=root/'output/pymousestudio_realism_cycle_20261002';out=base/'v015_head_mount';out.mkdir(exist_ok=False)
s=bpy.context.scene;s.frame_set(31);source=bpy.data.filepath;arm=bpy.data.objects['DeformationSystem'];bone='Head1_M'
library=base/'head_asset/head_mount_local.blend'
with bpy.data.libraries.load(str(library),link=False) as (data_from,data_to):
    data_to.objects=[n for n in data_from.objects if n.startswith('ReplicaAsset_')]
objects=data_to.objects;assert len(objects)==2
records=[]
for ob in objects:
    s.collection.objects.link(ob);assert ob.animation_data is None and not ob.constraints
    ob.parent=arm;ob.parent_type='BONE';ob.parent_bone=bone;ob.matrix_parent_inverse=Matrix.Identity(4);ob.matrix_basis=Matrix.Identity(4)
    bpy.context.view_layer.update();parent=ob.matrix_world.copy();desired=arm.matrix_world @ arm.pose.bones[bone].matrix
    ob.matrix_basis=parent.inverted() @ desired;ob.hide_render=False;ob.hide_set(False)
    if hasattr(ob.cycles,'is_caustics_receiver'):ob.cycles.is_caustics_receiver=True
    bpy.context.view_layer.update();records.append(dict(name=ob.name,dimensions_mm=list(ob.dimensions),vertices=len(ob.data.vertices),faces=len(ob.data.polygons),materials=[m.name for m in ob.data.materials if m]))
checks=[]
for frame in [30,31,32]:
    s.frame_set(frame);bpy.context.view_layer.update();desired=arm.matrix_world @ arm.pose.bones[bone].matrix
    for ob in objects:
        error=max(abs(ob.matrix_world[i][j]-desired[i][j]) for i in range(4) for j in range(4));assert error<.0002,(frame,ob.name,error)
        checks.append(dict(frame=frame,object=ob.name,max_matrix_error=error))
s.frame_set(31)
prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='OPTIX';prefs.get_devices();devices=[]
for d in prefs.devices:
    d.use=d.type=='OPTIX' and 'RTX PRO 6000' in d.name
    if d.use:devices.append(d.name)
assert devices;s.cycles.device='GPU';s.cycles.samples=32;s.cycles.use_denoising=True;s.render.resolution_percentage=50;s.render.use_persistent_data=True
m=dict(source_blend=source,asset_library=str(library),frame=31,objects=records,attachment_checks=checks,devices=devices,renders=[],limitations=['Existing cranial window and bar are a hardware proxy, not measured real implant geometry.','No source animation imported; parent tracking alone tested at frames30/31/32. Other poses not validated.'])
def write():(out/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2))
write()
for c in 'ABDF':
    assert dt.datetime.now(dt.timezone.utc)<dt.datetime(2026,10,3,6,59,16,tzinfo=dt.timezone.utc)
    s.camera=bpy.data.objects['Replica_FR2_cam'+c];r=float(s.camera['replica_pixel_aspect_y_over_x']);s.render.pixel_aspect_x=max(1.,1/r);s.render.pixel_aspect_y=max(1.,r)
    s.render.filepath=str(out/f'cam{c}.png');t=time.monotonic();bpy.ops.render.render(write_still=True);m['renders'].append(dict(camera=c,file=s.render.filepath,seconds=time.monotonic()-t));write()
s.render.resolution_percentage=100;s.cycles.samples=64
target=root/'BlenderPyMouseProjects/Cylinder/PyMouseStudio-RealCapture-Replica-20261002/PyMouseStudio-RealCapture-Replica-v015-head-mount.blend';assert not target.exists()
bpy.ops.wm.save_as_mainfile(filepath=str(target));m['candidate_blend']=str(target);write()
