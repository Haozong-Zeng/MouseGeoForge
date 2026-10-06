"""Unmodified repeat-render controls for the largest reopened-image outliers."""
import bpy,json,sys,time,datetime as dt
from pathlib import Path
root=Path(sys.argv[sys.argv.index('--')+1]);base=root/'output/pymousestudio_realism_cycle_20261002';out=base/'v065_repeatability';out.mkdir(exist_ok=False);m=json.loads((base/'v064_final_delivery/manifest.json').read_text());report=dict(renders=[],no_scene_edits_except_camera_and_output=True)
for tag,c in [('climb','A'),('quadruped','F')]:
    assert dt.datetime.now(dt.timezone.utc)<dt.datetime(2026,10,3,6,59,16,tzinfo=dt.timezone.utc)
    bpy.ops.wm.open_mainfile(filepath=m['poses'][tag]['delivery_blend'],load_ui=False);s=bpy.context.scene;prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='OPTIX';prefs.get_devices();devices=[]
    for d in prefs.devices:
        d.use=d.type=='OPTIX' and 'RTX PRO 6000' in d.name
        if d.use:devices.append(d.name)
    assert devices and s.cycles.device=='GPU' and s.cycles.samples==128
    s.camera=bpy.data.objects['Replica_FR2_cam'+c];r=float(s.camera['replica_pixel_aspect_y_over_x']);s.render.pixel_aspect_x=max(1.,1/r);s.render.pixel_aspect_y=max(1.,r);s.render.filepath=str(out/f'{tag}_cam{c}.png');t=time.monotonic();bpy.ops.render.render(write_still=True);report['renders'].append(dict(pose=tag,camera=c,file=s.render.filepath,seed=s.cycles.seed,animated_seed=s.cycles.use_animated_seed,seconds=time.monotonic()-t));(out/'manifest.json').write_text(json.dumps(report,indent=2))
