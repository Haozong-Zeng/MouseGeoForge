"""Native-resolution check, including raw-vs-denoised fur appearance."""
import bpy,json,sys,time,datetime as dt
from pathlib import Path
root=Path(sys.argv[sys.argv.index('--')+1]);out=root/globals().get('OUTPUT_RELATIVE','output/pymousestudio_realism_cycle_20261002/v014_room_fill/native128');out.mkdir(exist_ok=False)
s=bpy.context.scene;s.frame_set(globals().get('FRAME',31));s.render.resolution_percentage=100;s.cycles.samples=128;s.render.use_persistent_data=True
prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='OPTIX';prefs.get_devices();devices=[]
for d in prefs.devices:
    d.use=d.type=='OPTIX' and 'RTX PRO 6000' in d.name
    if d.use:devices.append(d.name)
assert devices;s.cycles.device='GPU';rows=[]
for c,denoise in globals().get('RENDER_CASES',[('F',True),('F',False),('A',True)]):
    assert dt.datetime.now(dt.timezone.utc)<dt.datetime(2026,10,3,6,59,16,tzinfo=dt.timezone.utc)
    s.camera=bpy.data.objects['Replica_FR2_cam'+c];r=float(s.camera['replica_pixel_aspect_y_over_x']);s.render.pixel_aspect_x=max(1.,1/r);s.render.pixel_aspect_y=max(1.,r);s.cycles.use_denoising=denoise
    s.render.filepath=str(out/f'cam{c}_{"denoised" if denoise else "raw"}.png');t=time.monotonic();bpy.ops.render.render(write_still=True)
    rows.append(dict(camera=c,denoised=denoise,file=s.render.filepath,seconds=time.monotonic()-t));(out/'manifest.json').write_text(json.dumps(dict(blend=bpy.data.filepath,frame=s.frame_current,samples=128,size=[1080,1200],devices=devices,renders=rows),indent=2))
