"""Re-test extended emitters after removal of the confounding backdrop texture."""
import bpy,json,sys,time,datetime as dt
from pathlib import Path
root=Path(sys.argv[sys.argv.index('--')+1]);out=root/'output/pymousestudio_realism_cycle_20261002/v047_key_radius';out.mkdir(exist_ok=False);s=bpy.context.scene
lights={n:bpy.data.objects[n].data for n in ['Replica_KeyPanel','Replica_RimPanel']};original={n:dict(energy=d.energy,radius=d.shadow_soft_size,type=d.type) for n,d in lights.items()}
prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='OPTIX';prefs.get_devices();devices=[]
for d in prefs.devices:
    d.use=d.type=='OPTIX' and 'RTX PRO 6000' in d.name
    if d.use:devices.append(d.name)
assert devices;s.cycles.device='GPU';s.cycles.use_denoising=True;s.render.use_persistent_data=True;s.cycles.samples=128;s.render.resolution_percentage=100
m=dict(source=bpy.data.filepath,frame=s.frame_current,devices=devices,original=original,candidates={},renders=[],hypothesis='Larger key/rim emitter radii may soften the headbar saturated reflection without reducing underside illumination. Revisited now inherited backdrop roughness texture is removed.',limitations=['Numeric scene unit is millimeter although Blender unit_scale is1; radius values are scene millimeters.','Total power, positions, bottom light, all glass/caustics, materials, poses and cameras retained. Not measured physical lamp reconstruction.'])
def write():(out/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2))
for tag,radius in [('a_radius6',6.),('b_radius12',12.)]:
    for d in lights.values():d.shadow_soft_size=radius
    target=root/f'BlenderPyMouseProjects/Cylinder/PyMouseStudio-RealCapture-Replica-20261002/PyMouseStudio-RealCapture-Replica-v047-{tag}.blend';assert not target.exists();bpy.ops.wm.save_as_mainfile(filepath=str(target),compress=True);m['candidates'][tag]=dict(blend=str(target),radius_mm=radius);write()
    for c in 'AF':
        assert dt.datetime.now(dt.timezone.utc)<dt.datetime(2026,10,3,6,59,16,tzinfo=dt.timezone.utc)
        s.camera=bpy.data.objects['Replica_FR2_cam'+c];r=float(s.camera['replica_pixel_aspect_y_over_x']);s.render.pixel_aspect_x=max(1.,1/r);s.render.pixel_aspect_y=max(1.,r);s.render.filepath=str(out/f'{tag}_cam{c}.png');t=time.monotonic();bpy.ops.render.render(write_still=True);m['renders'].append(dict(tag=tag,camera=c,file=s.render.filepath,seconds=time.monotonic()-t));write()
