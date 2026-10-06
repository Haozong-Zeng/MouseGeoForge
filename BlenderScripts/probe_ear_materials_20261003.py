"""Independent ear-only hypotheses, immutable scene and source controls."""
import bpy,datetime as dt,hashlib,json,sys,time
from pathlib import Path
import numpy as np

root=Path(sys.argv[-1]);base=root/'output/pymousestudio_ear_cycle_20261003';version=globals().get('VERSION','v001');out=base/('probe_'+version);out.mkdir(exist_ok=False)
plan=json.loads((base/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['deadline_utc'])
project=root/'BlenderPyMouseProjects/Cylinder/PyMouseStudio-RealCapture-EarStudy-20261003'
source=project/'EarStudy-v000-baseline-climb.blend'
def val(v):
    try:return list(v)
    except TypeError:return v if isinstance(v,(str,int,float,bool)) else str(v)
def mat_signature(m):
    nt=m.node_tree
    return dict(name=m.name,nodes=[dict(type=n.bl_idname,name=n.name,inputs=[(i.name,val(i.default_value)) for i in n.inputs if hasattr(i,'default_value')]) for n in nt.nodes],links=[(l.from_node.name,l.from_socket.identifier,l.to_node.name,l.to_socket.identifier) for l in nt.links]) if m.use_nodes else dict(name=m.name,diffuse=list(m.diffuse_color))
def snapshot():
    mouse=bpy.data.objects['MS_Mouse'];bpy.context.view_layer.update();ev=mouse.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=ev.to_mesh();world=np.array([ev.matrix_world@v.co for v in mesh.vertices]);ev.to_mesh_clear()
    objects={o.name:[list(r) for r in o.matrix_world] for o in bpy.data.objects}
    bones={b.name:[list(r) for r in b.matrix] for b in bpy.data.objects['DeformationSystem'].pose.bones}
    used={m for o in bpy.data.objects if hasattr(o.data,'materials') for i,m in enumerate(o.data.materials) if m and not(o.name=='MS_Mouse' and i==4)}
    mats={m.name:mat_signature(m) for m in used}
    return world,json.dumps(dict(objects=objects,bones=bones,materials=mats),sort_keys=True)

candidates=globals().get('CANDIDATES',[('a_albedo030',dict(gray=.30)),('b_albedo060',dict(gray=.60)),('c_soft_spec',dict(gray=.30,roughness=.50,specular=.25)),('d_translucent',dict(gray=.30,translucency=.30)),('e_subsurface',dict(gray=.30,sss=.15,scale=.30))])
manifest=dict(source=str(source),complete=False,candidates={},renders=[],hypothesis='Ear slot4 only, independent reset per candidate; a/b brightness. c/d/e compared with a to isolate specular response, thin diffuse transmission, or subsurface scattering.',limitations=['Appearance parameters, not measured IR spectral or physiological constants.','Pose mismatch prevents pixel-aligned real-image error scoring.','Existing ear geometry and all other scene settings held fixed.'])
def write():(out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
for tag,params in candidates:
    assert dt.datetime.now(dt.timezone.utc)<deadline
    bpy.ops.wm.open_mainfile(filepath=str(source));s=bpy.context.scene;mouse=bpy.data.objects['MS_Mouse'];before,signature=snapshot()
    original=mouse.data.materials[4];assert original.name=='Replica_BalancedSkin_4'
    mat=original.copy();mat.name='EarStudy_'+version+'_'+tag;mouse.data.materials[4]=mat;nt=mat.node_tree;bs=next(n for n in nt.nodes if n.type=='BSDF_PRINCIPLED')
    assert not bs.inputs['Base Color'].is_linked
    bs.inputs['Base Color'].default_value=(params['gray'],)*3+(1,)
    if 'roughness' in params:bs.inputs['Roughness'].default_value=params['roughness']
    if 'specular' in params:bs.inputs['Specular IOR Level'].default_value=params['specular']
    if 'sss' in params:
        bs.inputs['Subsurface Weight'].default_value=params['sss'];bs.inputs['Subsurface Scale'].default_value=params['scale']
        bs.inputs['Subsurface Radius'].default_value=(1,1,1)
    if 'translucency' in params:
        output=next(n for n in nt.nodes if n.type=='OUTPUT_MATERIAL');trans=nt.nodes.new('ShaderNodeBsdfTranslucent');trans.inputs['Color'].default_value=(.6,.6,.6,1)
        mix=nt.nodes.new('ShaderNodeMixShader');mix.inputs[0].default_value=params['translucency'];nt.links.new(bs.outputs[0],mix.inputs[1]);nt.links.new(trans.outputs[0],mix.inputs[2]);nt.links.new(mix.outputs[0],output.inputs['Surface'])
    after,check=snapshot();assert np.array_equal(before,after);assert signature==check
    prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='OPTIX';prefs.get_devices();devices=[]
    for d in prefs.devices:
        d.use=d.type=='OPTIX'
        if d.use:devices.append(d.name)
    assert devices;s.cycles.device='GPU';s.cycles.samples=128;s.cycles.use_denoising=True;s.render.resolution_percentage=100
    target=project/f'EarStudy-{version}-{tag}-climb.blend';assert not target.exists()
    bpy.ops.wm.save_as_mainfile(filepath=str(target),compress=True)
    manifest['candidates'][tag]=dict(params=params,blend=str(target),skin_max_change_mm=0,object_bone_non_ear_material_snapshot_unchanged=True,ear_faces=sum(p.material_index==4 for p in mouse.data.polygons),devices=devices)
    write()
    for c in 'CF':
        assert dt.datetime.now(dt.timezone.utc)<deadline
        s.camera=bpy.data.objects['Replica_FR2_cam'+c];r=float(s.camera['replica_pixel_aspect_y_over_x']);s.render.pixel_aspect_x=max(1.,1/r);s.render.pixel_aspect_y=max(1.,r)
        s.render.filepath=str(out/f'{tag}_cam{c}.png');t=time.monotonic();bpy.ops.render.render(write_still=True)
        manifest['renders'].append(dict(candidate=tag,camera=c,seconds=time.monotonic()-t));write();print('EAR_PROBE',tag,c,flush=True)
manifest['complete']=True;write()
print('EAR_PROBE_COMPLETE')
