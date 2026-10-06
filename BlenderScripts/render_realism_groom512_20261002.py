"""One high-sample native view to separate groom shape from denoiser artifacts."""
import bpy,json,sys,time,datetime as dt
from pathlib import Path
root=Path(sys.argv[sys.argv.index('--')+1]);out=root/'output/pymousestudio_realism_cycle_20261002/v021_groom_variation/native512';out.mkdir(exist_ok=False)
s=bpy.context.scene;s.frame_set(31);s.render.resolution_percentage=100;s.cycles.samples=512;s.cycles.adaptive_threshold=.002;s.cycles.use_adaptive_sampling=True;s.cycles.use_denoising=True
prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='OPTIX';prefs.get_devices();devices=[]
for d in prefs.devices:
    d.use=d.type=='OPTIX' and 'RTX PRO 6000' in d.name
    if d.use:devices.append(d.name)
assert devices;s.cycles.device='GPU';s.camera=bpy.data.objects['Replica_FR2_camF'];r=float(s.camera['replica_pixel_aspect_y_over_x']);s.render.pixel_aspect_x=max(1.,1/r);s.render.pixel_aspect_y=max(1.,r)
assert dt.datetime.now(dt.timezone.utc)<dt.datetime(2026,10,3,6,59,16,tzinfo=dt.timezone.utc)
s.render.filepath=str(out/'camF_denoised.png');t=time.monotonic();bpy.ops.render.render(write_still=True)
(out/'manifest.json').write_text(json.dumps(dict(blend=bpy.data.filepath,frame=31,camera='F',samples=512,adaptive_threshold=.002,size=[1080,1200],devices=devices,seconds=time.monotonic()-t,file=s.render.filepath),indent=2))
