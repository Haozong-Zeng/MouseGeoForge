"""Diagnostic only: disable shadow-caustics reception on the head bar, not mouse skin."""
import bpy,json,sys,time,datetime as dt
from pathlib import Path
root=Path(sys.argv[sys.argv.index('--')+1]);out=root/'output/pymousestudio_realism_cycle_20261002/v062_implant_caustics';out.mkdir(exist_ok=False);s=bpy.context.scene;ob=bpy.data.objects['ReplicaAsset_立方体'];mouse=bpy.data.objects['MS_Mouse'];original=ob.cycles.is_caustics_receiver
m=dict(source=bpy.data.filepath,object=ob.name,original_receiver=original,mouse_receiver=mouse.cycles.is_caustics_receiver,world_scale=[ob.matrix_world.to_3x3().col[i].length for i in range(3)],world_dimensions=list(ob.dimensions),frame=s.frame_current,geometry_unchanged=True,lights_unchanged=True,cameras_unchanged=True,not_saved_as_blend=True,renders=[],limitations=['Diagnostic ablation only. Disabling a receiver is not a physically validated material correction.', 'Do not automatically adopt this image as final RGB or ground truth.'])
(out/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2));assert original and mouse.cycles.is_caustics_receiver
ob.cycles.is_caustics_receiver=False
prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='OPTIX';prefs.get_devices();devices=[]
for d in prefs.devices:
    d.use=d.type=='OPTIX' and 'RTX PRO 6000' in d.name
    if d.use:devices.append(d.name)
assert devices;s.cycles.device='GPU';s.cycles.samples=128;s.cycles.use_denoising=True;s.render.resolution_x=1080;s.render.resolution_y=1200;s.render.resolution_percentage=100
s.camera=bpy.data.objects['Replica_FR2_camF'];r=float(s.camera['replica_pixel_aspect_y_over_x']);s.render.pixel_aspect_x=max(1.,1/r);s.render.pixel_aspect_y=max(1.,r);s.render.filepath=str(out/'bar_receiver_off_camF.png');assert dt.datetime.now(dt.timezone.utc)<dt.datetime(2026,10,3,6,59,16,tzinfo=dt.timezone.utc);t=time.monotonic();bpy.ops.render.render(write_still=True);m['devices']=devices;m['renders'].append(dict(file=s.render.filepath,seconds=time.monotonic()-t));(out/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2))
