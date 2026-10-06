"""Original-scene RGB baseline with explicit GPU; no source save or material edit."""
import bpy,json,sys,time,datetime as dt
from pathlib import Path
out=Path(sys.argv[sys.argv.index('--')+1]);out.mkdir(exist_ok=False)
s=bpy.context.scene;prefs=bpy.context.preferences.addons['cycles'].preferences
prefs.compute_device_type='OPTIX';prefs.get_devices()
selected=[]
for device in prefs.devices:
    device.use=device.type=='OPTIX' and 'RTX PRO 6000' in device.name
    if device.use:selected.append(device.name)
assert selected,'No requested GPU; refuse CPU fallback'
s.render.engine='CYCLES';s.cycles.device='GPU';s.cycles.samples=16;s.cycles.use_denoising=True
s.render.resolution_x=512;s.render.resolution_y=512;s.render.resolution_percentage=100
s.render.use_compositing=False;s.render.use_sequencer=False;s.render.use_persistent_data=True
assert 'PNG' in [i.identifier for i in s.render.image_settings.bl_rna.properties['file_format'].enum_items]
s.render.image_settings.file_format='PNG';s.frame_set(31)
records=[]
for name in ['Camera.007','Camera.001','Camera.002','Camera.003']:
    assert dt.datetime.now(dt.timezone.utc)<dt.datetime(2026,10,3,6,59,16,tzinfo=dt.timezone.utc)
    s.camera=bpy.data.objects[name];s.render.filepath=str(out/(name+'.png'));t=time.monotonic()
    bpy.ops.render.render(write_still=True)
    records.append(dict(camera=name,frame=31,file=s.render.filepath,seconds=time.monotonic()-t))
    (out/'manifest.json').write_text(json.dumps(dict(devices=selected,resolution=[512,512],samples=16,compositing=False,source_saved=False,original_camera_layout=True,real_camera_mapping_not_yet_established=True,renders=records),indent=2))
    print('BASELINE_RENDER_COMPLETE',name,flush=True)
