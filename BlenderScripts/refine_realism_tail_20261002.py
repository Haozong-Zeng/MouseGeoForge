"""Restore fine short tail hairs, then isolate lower ear albedo."""
import bpy,json,sys,time,datetime as dt
from pathlib import Path
root=Path(sys.argv[sys.argv.index('--')+1]);base=root/'output/pymousestudio_realism_cycle_20261002';out=base/'v017_fine_tail';out.mkdir(exist_ok=False)
s=bpy.context.scene;s.frame_set(31);source=bpy.data.filepath;tail=bpy.data.objects['TailFur.002'];tail.data=tail.data.copy();tail.hide_render=False
for curve in tail.data.curves:
    root_co=curve.points[0].position.copy()
    for point in curve.points:point.position=root_co+.35*(point.position-root_co)
profile=tail.modifiers['Set Hair Curve Profile'];old_radius=float(profile['Input_3']);profile['Input_3']=old_radius*.30;tail.update_tag()
prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='OPTIX';prefs.get_devices();devices=[]
for d in prefs.devices:
    d.use=d.type=='OPTIX' and 'RTX PRO 6000' in d.name
    if d.use:devices.append(d.name)
assert devices;s.cycles.device='GPU';s.cycles.samples=48;s.cycles.use_denoising=True;s.render.resolution_percentage=50;s.render.use_persistent_data=True
m=dict(source_blend=source,frame=31,devices=devices,tail=dict(guide_length_factor=.35,profile_radius_factor=.30,old_radius=old_radius,new_radius=float(profile['Input_3']),curves=len(tail.data.curves),roots_preserved=True),renders=[],candidate_blends={},limitations=['Tail appearance scalars are visual hypotheses, not measured hair dimensions.','Mouse skin topology, pose and cameras unchanged.','Hardware approximate; only frame31 visually validated.'])
def write():(out/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2))
write()
for stage,views in [('a_fine_tail','ABDF'),('b_ear_gray020','AF')]:
    if stage=='b_ear_gray020':
        mouse=bpy.data.objects['MS_Mouse'];mat=mouse.data.materials[4].copy();mat.name='Replica_Ear_Gray020';mouse.data.materials[4]=mat
        shader=next(n for n in mat.node_tree.nodes if n.type=='BSDF_PRINCIPLED');shader.inputs['Base Color'].default_value=(.20,.20,.20,1)
    for c in views:
        assert dt.datetime.now(dt.timezone.utc)<dt.datetime(2026,10,3,6,59,16,tzinfo=dt.timezone.utc)
        s.camera=bpy.data.objects['Replica_FR2_cam'+c];r=float(s.camera['replica_pixel_aspect_y_over_x']);s.render.pixel_aspect_x=max(1.,1/r);s.render.pixel_aspect_y=max(1.,r)
        s.render.filepath=str(out/f'{stage}_cam{c}.png');t=time.monotonic();bpy.ops.render.render(write_still=True);m['renders'].append(dict(stage=stage,camera=c,file=s.render.filepath,seconds=time.monotonic()-t));write()
    target=root/f'BlenderPyMouseProjects/Cylinder/PyMouseStudio-RealCapture-Replica-20261002/PyMouseStudio-RealCapture-Replica-v017-{stage}.blend';assert not target.exists()
    s.render.resolution_percentage=100;s.cycles.samples=64;bpy.ops.wm.save_as_mainfile(filepath=str(target));m['candidate_blends'][stage]=str(target);write();s.render.resolution_percentage=50;s.cycles.samples=48
