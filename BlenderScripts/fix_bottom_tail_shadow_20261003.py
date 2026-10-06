"""Clear-floor shadow transmission approximation, with immutable source audit."""
import bpy,sys,json,hashlib,time
from pathlib import Path
import numpy as np
root=Path(sys.argv[-1]);base=root/'output/bottom_tail_fix_20261003';out=base/globals().get('OUTPUT_VERSION','final');out.mkdir(exist_ok=False)
source_dir=root/'BlenderPyMouseProjects/Cylinder/PyMouseStudio-RealCapture-EarStudy-20261003/DELIVERY'
delivery=root/'BlenderPyMouseProjects/Cylinder/PyMouseStudio-RealCapture-BottomTailFix-20261003'/globals().get('DELIVERY_FOLDER','DELIVERY');delivery.mkdir(parents=True,exist_ok=False)
def sha(p):
    with p.open('rb') as h:return hashlib.file_digest(h,'sha256').hexdigest()
def signature():
    mouse=bpy.data.objects['MS_Mouse'];bpy.context.view_layer.update();ev=mouse.evaluated_get(bpy.context.evaluated_depsgraph_get());me=ev.to_mesh();v=np.array([ev.matrix_world@p.co for p in me.vertices]);ev.to_mesh_clear()
    objects={o.name:[list(r) for r in o.matrix_world] for o in bpy.data.objects}
    bones={b.name:[list(r) for r in b.matrix] for b in bpy.data.objects['DeformationSystem'].pose.bones}
    return v,json.dumps(dict(objects=objects,bones=bones),sort_keys=True)
def fix(floor):
    old=floor.data.materials[0];mat=old.copy();mat.name='BottomTailFix_Floor_ShadowTransmission';floor.data.materials[0]=mat;tree=mat.node_tree;glass=next(n for n in tree.nodes if n.type=='BSDF_GLASS');links=list(glass.outputs[0].links)
    path=tree.nodes.new('ShaderNodeLightPath');path.name='BottomTailFix_LightPath'
    trans=tree.nodes.new('ShaderNodeBsdfTransparent');trans.name='BottomTailFix_ShadowTransmission';trans.inputs[0].default_value=(.945,.945,.945,1)
    mix=tree.nodes.new('ShaderNodeMixShader');mix.name='BottomTailFix_ShadowOnly';mix.label='Shadow approximation only; optical glass retained for camera/reflection rays'
    tree.links.new(path.outputs['Is Shadow Ray'],mix.inputs[0]);tree.links.new(glass.outputs[0],mix.inputs[1]);tree.links.new(trans.outputs[0],mix.inputs[2])
    for link in links:tree.links.new(mix.outputs[0],link.to_socket)
    for node in tree.nodes:
        if node.type=='BSDF_DIFFUSE':
            old=list(node.inputs['Color'].default_value);node.inputs['Color'].default_value=tuple(v*globals().get('DUST_FACTOR',1.) for v in old[:3])+(old[3],)
    floor.cycles.is_caustics_caster=False
    floor['bottom_tail_fix_note']='Clear glass uses .945 per-interface straight shadow transmission. Approximation, not full caustic simulation. Original optical BSDF and dust branches retained. No mouse geometry/material edits.'
manifest=dict(complete=False,method='Clear-floor-only Is Shadow Ray transmission .945 per interface; floor MNEE caster disabled to avoid duplicate/incompatible shadow paths.',limitations=['Approximate straight-through direct shadow transmission, not complete physical caustics.','Floor dust and animal underside illumination can change; camera/reflection/refraction branch and dust coverage retained.','No data exports or existing datasets altered.'],sources=['https://docs.blender.org/manual/en/2.80/render/cycles/optimizations/reducing_noise.html'],poses={})
manifest['dust_diffuse_color_factor']=globals().get('DUST_FACTOR',1.)
def write():(out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
for pose,label in [('headraise','HeadRaise'),('climb','Climb'),('quadruped','Quadruped')]:
    source=source_dir/f'PyMouseStudio-EarStudy-{label}-Static.blend';source_sha=sha(source);bpy.ops.wm.open_mainfile(filepath=str(source));s=bpy.context.scene;v,sig=signature();mouse=bpy.data.objects['MS_Mouse'];mat_names=[m.name for m in mouse.data.materials];floor=bpy.data.objects['Part_AntiReflectiveGlass'];fix(floor)
    vv,ss=signature();assert np.array_equal(v,vv) and sig==ss;assert mat_names==[m.name for m in mouse.data.materials]
    prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='OPTIX';prefs.get_devices()
    for d in prefs.devices:d.use=d.type=='OPTIX'
    assert any(d.use for d in prefs.devices);s.cycles.device='GPU';s.cycles.samples=128;s.render.resolution_percentage=100
    target=delivery/f'PyMouseStudio-BottomTailFix-{label}-Static.blend';s.render.filepath='//renders/'+pose+'/camF.png';bpy.ops.wm.save_as_mainfile(filepath=str(target),compress=True)
    bpy.ops.wm.open_mainfile(filepath=str(target));s=bpy.context.scene;vv,ss=signature();assert np.array_equal(v,vv) and sig==ss
    record=dict(source=str(source),source_sha256=source_sha,blend=str(target),blend_sha256=sha(target),skin_max_change_mm=float(np.abs(v-vv).max()),object_bone_camera_matrices_unchanged=True,mouse_materials_unchanged=True,frame=s.frame_current,renders=[]);manifest['poses'][pose]=record;write()
    for c in 'AF':
        s.camera=bpy.data.objects['Replica_FR2_cam'+c];r=float(s.camera['replica_pixel_aspect_y_over_x']);s.render.pixel_aspect_x=max(1.,1/r);s.render.pixel_aspect_y=max(1.,r)
        folder=delivery/'renders'/pose;folder.mkdir(parents=True,exist_ok=True);s.render.filepath=str(folder/f'cam{c}.png');t=time.monotonic();bpy.ops.render.render(write_still=True);record['renders'].append(dict(view=c,seconds=time.monotonic()-t));write()
    assert sha(source)==source_sha and sha(target)==record['blend_sha256']
manifest['complete']=True;write();print('BOTTOM_TAIL_FIX_VALIDATED')
