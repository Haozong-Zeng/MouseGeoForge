"""Register visible bottom-camera housing behind frozen optical center; reflection test."""
import bpy,json,sys,time,datetime as dt
from pathlib import Path
from mathutils import Matrix,Vector
root=Path(sys.argv[sys.argv.index('--')+1]);out=root/'output/pymousestudio_realism_cycle_20261002/v054_bottom_camera';out.mkdir(exist_ok=False);s=bpy.context.scene;before={c:[list(r) for r in bpy.data.objects['Replica_FR2_cam'+c].matrix_world] for c in 'ABCDEFG'}
old=bpy.data.objects['Part_PiCamBottom'];old_visibility=dict(hide_render=old.hide_render,visible_camera=old.visible_camera);old.hide_render=True
co=bpy.data.collections.new('Replica_BottomCamera_VisualProxy');s.collection.children.link(co);holder=bpy.data.objects.new('Replica_CameraProxy_A',None);co.objects.link(holder);holder.matrix_world=bpy.data.objects['Replica_FR2_camA'].matrix_world;created=[]
for src in list(bpy.data.objects['Replica_CameraProxy_C'].children):
    o=src.copy()
    if src.data:o.data=src.data.copy()
    co.objects.link(o);o.name=src.name.replace('_C','_A');o.parent=holder;o.matrix_parent_inverse=src.matrix_parent_inverse.copy();o.matrix_basis=src.matrix_basis.copy();o.hide_render=False;o.visible_camera=True;created.append(o.name)
# Backplate is entirely behind the lens, not a foreground occluder.
bpy.ops.mesh.primitive_cube_add(size=1);plate=bpy.context.object;plate.name='Replica_BottomCameraBackingPlate';plate.data.transform(Matrix.Diagonal(Vector((62,105,5,1))));plate.parent=holder;plate.location=(7,18,48)
for c in list(plate.users_collection):c.objects.unlink(plate)
co.objects.link(plate);plate.data.materials.append(bpy.data.materials['Replica_CameraBrushedMetal']);bevel=plate.modifiers.new('Plate bevel','BEVEL');bevel.width=.6;bevel.segments=3;created.append(plate.name)
bpy.context.view_layer.update()
for c,m in before.items():assert m==[list(r) for r in bpy.data.objects['Replica_FR2_cam'+c].matrix_world]
prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='OPTIX';prefs.get_devices();devices=[]
for d in prefs.devices:
    d.use=d.type=='OPTIX' and 'RTX PRO 6000' in d.name
    if d.use:devices.append(d.name)
assert devices;s.cycles.device='GPU';s.cycles.samples=128;s.cycles.use_denoising=True;s.render.resolution_percentage=100;s.render.use_persistent_data=True
target=root/'BlenderPyMouseProjects/Cylinder/PyMouseStudio-RealCapture-Replica-20261002/PyMouseStudio-RealCapture-Replica-v054-bottom-camera.blend';assert not target.exists();bpy.ops.wm.save_as_mainfile(filepath=str(target),compress=True)
m=dict(source=bpy.data.filepath,blend=str(target),frame=s.frame_current,created=created,old_bottom_camera_visibility=old_visibility,old_bottom_camera_hidden=True,camera_matrices_unchanged=True,mouse_pose_materials_unchanged=True,lights_glass_unchanged=True,devices=devices,renders=[],hypothesis='Real bottom-view central equipment may be floor reflection of bottom camera. Old CAD lensZ~140 differs from calibrated cameraZ162.96. Replace visual housing only with same procedural camera as side views registered behind optical center.',limitations=['Physical housing/backplate proxy, not measured CAD. Reflection hypothesis still needs visual validation.','No ray visibility filtering on new geometry; entirely behind camera optical center.','No source/label/calibration edits.'])
def write():(out/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2))
write()
for c in 'AF':
    assert dt.datetime.now(dt.timezone.utc)<dt.datetime(2026,10,3,6,59,16,tzinfo=dt.timezone.utc)
    s.camera=bpy.data.objects['Replica_FR2_cam'+c];r=float(s.camera['replica_pixel_aspect_y_over_x']);s.render.pixel_aspect_x=max(1.,1/r);s.render.pixel_aspect_y=max(1.,r);s.render.filepath=str(out/f'cam{c}.png');t=time.monotonic();bpy.ops.render.render(write_still=True);m['renders'].append(dict(camera=c,file=s.render.filepath,seconds=time.monotonic()-t));write()
