"""Rig-following tail microtexture A/B; no geometry displacement or camera edits."""
import bpy, sys, json, time, math, datetime as dt
from pathlib import Path
import numpy as np
root=Path(sys.argv[sys.argv.index('--')+1]);base=root/'output/pymousestudio_realism_cycle_20261002';out=base/'v057_tail_rings';out.mkdir(exist_ok=False)
s=bpy.context.scene;mouse=bpy.data.objects['MS_Mouse'];source=bpy.data.filepath
def points():
    bpy.context.view_layer.update();ev=mouse.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=ev.to_mesh();v=np.array([ev.matrix_world@v.co for v in mesh.vertices]);ev.to_mesh_clear();return v
before=points();cams={c:[list(r) for r in bpy.data.objects['Replica_FR2_cam'+c].matrix_world] for c in 'ABCDEFG'}
assert mouse.data.attributes.get('replica_tail_rest_z_v057') is None
attr=mouse.data.attributes.new('replica_tail_rest_z_v057','FLOAT','POINT')
for v,d in zip(mouse.data.vertices,attr.data):d.value=v.co.z
mat=mouse.data.materials[5].copy();mat.name='Replica_TailAnnularMicrotexture_v057';mouse.data.materials[5]=mat;nt=mat.node_tree
shader=next(n for n in nt.nodes if n.type=='BSDF_PRINCIPLED');original=list(shader.inputs['Base Color'].default_value)
assert not shader.inputs['Base Color'].is_linked and not shader.inputs['Normal'].is_linked
assert abs(original[0]-.3)<1e-5 and original[:3]==[original[0]]*3
a=nt.nodes.new('ShaderNodeAttribute');a.attribute_name=attr.name
def mathnode(operation,first,second=None):
    n=nt.nodes.new('ShaderNodeMath');n.operation=operation
    nt.links.new(first,n.inputs[0])
    if second is not None:n.inputs[1].default_value=second
    return n
phase=mathnode('MULTIPLY',a.outputs['Fac'],2*math.pi/.025)
wave=mathnode('SINE',phase.outputs[0]);amplitude=mathnode('MULTIPLY',wave.outputs[0],.015);bias=mathnode('ADD',amplitude.outputs[0],original[0]);nt.links.new(bias.outputs[0],shader.inputs['Base Color'])
bump=nt.nodes.new('ShaderNodeBump');bump.inputs['Strength'].default_value=0.;bump.inputs['Distance'].default_value=.015;nt.links.new(wave.outputs[0],bump.inputs['Height']);nt.links.new(bump.outputs['Normal'],shader.inputs['Normal'])
after=points();error=float(np.linalg.norm(after-before,axis=1).max());assert error==0
assert all(cams[c]==[list(r) for r in bpy.data.objects['Replica_FR2_cam'+c].matrix_world] for c in cams)
prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='OPTIX';prefs.get_devices();devices=[]
for d in prefs.devices:
    d.use=d.type=='OPTIX' and 'RTX PRO 6000' in d.name
    if d.use:devices.append(d.name)
assert devices;s.cycles.device='GPU';s.cycles.samples=128;s.cycles.use_denoising=True;s.render.resolution_x=1080;s.render.resolution_y=1200;s.render.resolution_percentage=100;s.render.use_persistent_data=True
m=dict(source=source,frame=s.frame_current,devices=devices,skin_max_change_mm=error,cameras_unchanged=True,mapping='Named point attribute original undeformed vertex Z, linearly interpolated across polygons and carried through armature. Period .025 local units (~.35mm before pose strain).',original_color=original,tail_face_count=sum(p.material_index==5 for p in mouse.data.polygons),candidates={},renders=[],limitations=['Appearance proxy, not measured histological scale pattern or complete irregular tail scales.','Rest-Z assumes original tail is aligned along local Z; not a general curved-rest centerline parameterization.','Color sine has unchanged nominal mean, not guaranteed unchanged rendered mean.','Bump changes shading normal only, not depth/silhouette or geometric ground truth.'])
def write():(out/'manifest.json').write_text(json.dumps(m,indent=2))
for tag,amp,strength in [('a_faint_color',.015,0.),('b_color_bump',.03,.2)]:
    amplitude.inputs[1].default_value=amp;bump.inputs['Strength'].default_value=strength
    s.camera=bpy.data.objects['Replica_FR2_camA'];r=float(s.camera['replica_pixel_aspect_y_over_x']);s.render.pixel_aspect_x=max(1.,1/r);s.render.pixel_aspect_y=max(1.,r)
    target=root/f'BlenderPyMouseProjects/Cylinder/PyMouseStudio-RealCapture-Replica-20261002/PyMouseStudio-RealCapture-Replica-v057-{tag}.blend';assert not target.exists();bpy.ops.wm.save_as_mainfile(filepath=str(target),compress=True)
    m['candidates'][tag]=dict(blend=str(target),color_amplitude=amp,bump_strength=strength,bump_distance_mm=.015);write()
    for c in 'AF':
        assert dt.datetime.now(dt.timezone.utc)<dt.datetime(2026,10,3,6,59,16,tzinfo=dt.timezone.utc)
        s.camera=bpy.data.objects['Replica_FR2_cam'+c];r=float(s.camera['replica_pixel_aspect_y_over_x']);s.render.pixel_aspect_x=max(1.,1/r);s.render.pixel_aspect_y=max(1.,r);s.render.filepath=str(out/f'{tag}_cam{c}.png');t=time.monotonic();bpy.ops.render.render(write_still=True);m['renders'].append(dict(tag=tag,camera=c,file=s.render.filepath,seconds=time.monotonic()-t));write()
