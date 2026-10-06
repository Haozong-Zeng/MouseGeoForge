"""Physical backdrop-albedo ablation, with all scene geometry and lights frozen."""
import bpy,json,sys,time,datetime as dt
from pathlib import Path
import numpy as np
root=Path(sys.argv[sys.argv.index('--')+1]);out=root/'output/pymousestudio_realism_cycle_20261002/v058_backdrop_variation';out.mkdir(exist_ok=False);s=bpy.context.scene;source=bpy.data.filepath;ob=bpy.data.objects['Part_WhiteHexagonBackground'];mouse=bpy.data.objects['MS_Mouse']
def points():
    bpy.context.view_layer.update();ev=mouse.evaluated_get(bpy.context.evaluated_depsgraph_get());me=ev.to_mesh();q=np.array([ev.matrix_world@v.co for v in me.vertices]);ev.to_mesh_clear();return q
before=points();cameras={c:[list(r) for r in bpy.data.objects['Replica_FR2_cam'+c].matrix_world] for c in 'ABCDEFG'};lights={o.name:dict(energy=o.data.energy,matrix=[list(r) for r in o.matrix_world]) for o in s.objects if o.type=='LIGHT'}
old=ob.material_slots[0].material;mat=old.copy();mat.name='Replica_BackdropVariation_v058'
for slot in ob.material_slots:slot.material=mat
nt=mat.node_tree;bs=next(n for n in nt.nodes if n.type=='BSDF_PRINCIPLED');assert not bs.inputs['Base Color'].is_linked
pos=nt.nodes.new('ShaderNodeNewGeometry');stretch=nt.nodes.new('ShaderNodeVectorMath');stretch.operation='MULTIPLY';stretch.inputs[1].default_value=(1,1,.12);nt.links.new(pos.outputs['Position'],stretch.inputs[0]);noise=nt.nodes.new('ShaderNodeTexNoise');noise.noise_dimensions='3D';noise.inputs['Scale'].default_value=.018;noise.inputs['Detail'].default_value=2.;noise.inputs['Roughness'].default_value=.6;nt.links.new(stretch.outputs['Vector'],noise.inputs['Vector']);ramp=nt.nodes.new('ShaderNodeMapRange');ramp.clamp=True;ramp.inputs['From Min'].default_value=.25;ramp.inputs['From Max'].default_value=.75;ramp.inputs['To Min'].default_value=.04;ramp.inputs['To Max'].default_value=.36;nt.links.new(noise.outputs['Fac'],ramp.inputs['Value'])
assert float(np.linalg.norm(points()-before,axis=1).max())==0
assert all(cameras[c]==[list(r) for r in bpy.data.objects['Replica_FR2_cam'+c].matrix_world] for c in cameras)
assert all(lights[o.name]==dict(energy=o.data.energy,matrix=[list(r) for r in o.matrix_world]) for o in s.objects if o.type=='LIGHT')
prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='OPTIX';prefs.get_devices();devices=[]
for d in prefs.devices:
    d.use=d.type=='OPTIX' and 'RTX PRO 6000' in d.name
    if d.use:devices.append(d.name)
assert devices;s.cycles.device='GPU';s.cycles.samples=128;s.cycles.use_denoising=True;s.render.resolution_x=1080;s.render.resolution_y=1200;s.render.resolution_percentage=100;s.render.use_persistent_data=True
m=dict(source=source,frame=s.frame_current,devices=devices,previous_material=old.name,skin_max_change_mm=0,cameras_unchanged=True,lights_unchanged=True,candidates={},renders=[],limitations=['Background reflectance proxy, not measured curtain material or recovered illumination.','Unchanged lights do not imply unchanged mouse radiance: backdrop indirect bounce can change.','No geometry displacement, image overlay, external assets or source dataset edits.'])
def write():(out/'manifest.json').write_text(json.dumps(m,indent=2))
for tag in ['a_uniform020','b_vertical_variation']:
    bs.inputs['Base Color'].default_value=(.2,.2,.2,1)
    if tag.startswith('b_'):nt.links.new(ramp.outputs['Result'],bs.inputs['Base Color'])
    s.camera=bpy.data.objects['Replica_FR2_camF'];r=float(s.camera['replica_pixel_aspect_y_over_x']);s.render.pixel_aspect_x=max(1.,1/r);s.render.pixel_aspect_y=max(1.,r)
    target=root/f'BlenderPyMouseProjects/Cylinder/PyMouseStudio-RealCapture-Replica-20261002/PyMouseStudio-RealCapture-Replica-v058-{tag}.blend';assert not target.exists();bpy.ops.wm.save_as_mainfile(filepath=str(target),compress=True);m['candidates'][tag]=dict(blend=str(target),albedo=.2 if tag.startswith('a_') else dict(min=.04,max=.36,noise_world_scale=.018,vertical_scale=.12));write()
    for c in 'FA':
        assert dt.datetime.now(dt.timezone.utc)<dt.datetime(2026,10,3,6,59,16,tzinfo=dt.timezone.utc)
        s.camera=bpy.data.objects['Replica_FR2_cam'+c];r=float(s.camera['replica_pixel_aspect_y_over_x']);s.render.pixel_aspect_x=max(1.,1/r);s.render.pixel_aspect_y=max(1.,r);s.render.filepath=str(out/f'{tag}_cam{c}.png');t=time.monotonic();bpy.ops.render.render(write_still=True);m['renders'].append(dict(tag=tag,camera=c,file=s.render.filepath,seconds=time.monotonic()-t));write()
