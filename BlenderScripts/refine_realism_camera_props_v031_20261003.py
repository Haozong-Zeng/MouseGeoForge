"""Mirror housing asymmetry, remove unobserved LED fixtures; keep calibrated cameras."""
import bpy,sys,json,time,datetime as dt
from pathlib import Path
from mathutils import Vector
root=Path(sys.argv[sys.argv.index('--')+1]);out=root/'output/pymousestudio_realism_cycle_20261002/v031_camera_refinement';out.mkdir(exist_ok=False);s=bpy.context.scene;source=bpy.data.filepath
collection=bpy.data.collections['Replica_CalibratedCamera_VisualProxies'];mirrored=[]
for o in collection.objects:
    if o.parent and o.parent.name.startswith('Replica_CameraProxy_'):
        if o.type=='CURVE':
            for sp in o.data.splines:
                for b in sp.bezier_points:b.co.x*=-1
        else:o.location.x*=-1
        mirrored.append(o.name)
for c in 'BCDEFG':
    holder=bpy.data.objects['Replica_CameraProxy_'+c];p=holder.matrix_world@Vector((7,0,45))
    for prefix in ['Replica_Post_','Replica_PostCollar_']:o=bpy.data.objects[prefix+c];o.location.x=p.x;o.location.y=p.y
bs=next(n for n in bpy.data.materials['Replica_CameraBrushedMetal'].node_tree.nodes if n.type=='BSDF_PRINCIPLED')
# Separate lens rings from steel posts; real reference has a pale mounting ring.
ring=bpy.data.materials['Replica_CameraBrushedMetal'].copy();ring.name='Replica_PaleLensRing';bs=next(n for n in ring.node_tree.nodes if n.type=='BSDF_PRINCIPLED');bs.inputs['Base Color'].default_value=(.65,.65,.65,1);bs.inputs['Metallic'].default_value=.15;bs.inputs['Roughness'].default_value=.38
for c in 'BCDEFG':bpy.data.objects['Replica_LensRing_'+c].data.materials[0]=ring
hidden=[]
for name in ['LED.001','LED.002','LED.003']:
    co=bpy.data.collections[name];co.hide_render=True;hidden.extend(o.name for o in co.all_objects)
bpy.context.view_layer.update()
prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='OPTIX';prefs.get_devices();devices=[]
for d in prefs.devices:
    d.use=d.type=='OPTIX' and 'RTX PRO 6000' in d.name
    if d.use:devices.append(d.name)
assert devices;s.cycles.device='GPU';s.cycles.samples=64;s.cycles.use_denoising=True;s.render.use_persistent_data=True;s.render.resolution_percentage=100
target=root/'BlenderPyMouseProjects/Cylinder/PyMouseStudio-RealCapture-Replica-20261002/PyMouseStudio-RealCapture-Replica-v031-camera-refinement.blend';assert not target.exists();bpy.ops.wm.save_as_mainfile(filepath=str(target))
m=dict(source=source,blend=str(target),mirrored_props=mirrored,hidden_LED_meshes=hidden,remaining_lighting='Existing sun and Replica spot lights unchanged. Removing old emissive fixtures may change indirect illumination.',camera_calibration_unchanged=True,mouse_pose_materials_unchanged=True,devices=devices,renders=[])
def write():(out/'manifest.json').write_text(json.dumps(m,indent=2))
write()
for c in 'ABDF':
    assert dt.datetime.now(dt.timezone.utc)<dt.datetime(2026,10,3,6,59,16,tzinfo=dt.timezone.utc)
    s.camera=bpy.data.objects['Replica_FR2_cam'+c];r=float(s.camera['replica_pixel_aspect_y_over_x']);s.render.pixel_aspect_x=max(1.,1/r);s.render.pixel_aspect_y=max(1.,r);s.render.resolution_percentage=50;s.render.filepath=str(out/f'cam{c}.png');t=time.monotonic();bpy.ops.render.render(write_still=True);m['renders'].append(dict(camera=c,file=s.render.filepath,seconds=time.monotonic()-t));write()
