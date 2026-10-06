"""Apply one ear parameter to three independent copies, reopen and verify."""
import bpy,datetime as dt,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector
root=Path(sys.argv[-1]);base=root/'output/pymousestudio_ear_cycle_20261003';plan=json.loads((base/'plan.json').read_text());out=base/'validate_v003';out.mkdir(exist_ok=False)
project=root/'BlenderPyMouseProjects/Cylinder/PyMouseStudio-RealCapture-EarStudy-20261003/DELIVERY';project.mkdir(exist_ok=False)
deadline=dt.datetime.fromisoformat(plan['deadline_utc'])
def host(p):return root/Path(p).relative_to('/home/haozong/projects/MouseGeoForge')
def val(v):
    try:return list(v)
    except TypeError:return v if isinstance(v,(str,int,float,bool)) else str(v)
def mat_signature(m,omit_color=False):
    return dict(name=m.name if not omit_color else 'EAR',nodes=[dict(type=n.bl_idname,name=n.name,inputs=[(i.identifier,val(i.default_value)) for i in n.inputs if hasattr(i,'default_value') and not(omit_color and n.type=='BSDF_PRINCIPLED' and i.name=='Base Color')]) for n in m.node_tree.nodes],links=[(l.from_node.name,l.from_socket.identifier,l.to_node.name,l.to_socket.identifier) for l in m.node_tree.links]) if m.use_nodes else dict(name=m.name,color=list(m.diffuse_color))
def snapshot():
    mouse=bpy.data.objects['MS_Mouse'];bpy.context.view_layer.update();ev=mouse.evaluated_get(bpy.context.evaluated_depsgraph_get());me=ev.to_mesh();world=np.array([ev.matrix_world@v.co for v in me.vertices]);ev.to_mesh_clear()
    objects={o.name:[list(r) for r in o.matrix_world] for o in bpy.data.objects};bones={b.name:[list(r) for r in b.matrix] for b in bpy.data.objects['DeformationSystem'].pose.bones}
    mats={m for o in bpy.data.objects if hasattr(o.data,'materials') for i,m in enumerate(o.data.materials) if m and not(o.name=='MS_Mouse' and i==4)}
    values=dict(objects=objects,bones=bones,non_ear_materials={m.name:mat_signature(m) for m in mats},ear_except_basecolor=mat_signature(mouse.data.materials[4],True),lights={o.name:dict(energy=o.data.energy,color=list(o.data.color),type=o.data.type) for o in bpy.data.objects if o.type=='LIGHT'})
    return world,json.dumps(values,sort_keys=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=dict(selected_basecolor_linear=.60,only_appearance_parameter_changed='MS_Mouse material slot4 Principled Base Color .12 to .60 neutral gray',poses={},complete=False)
def write():(out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
for pose,label in [('climb','Climb'),('quadruped','Quadruped'),('headraise','HeadRaise')]:
    source=host(plan['sources'][pose]);assert sha(source)==plan['source_sha256'][pose]
    bpy.ops.wm.open_mainfile(filepath=str(source));s=bpy.context.scene;before,reference=snapshot();mouse=bpy.data.objects['MS_Mouse'];original=mouse.data.materials[4]
    assert original.name=='Replica_BalancedSkin_4'
    mat=original.copy();mat.name='EarStudy_Final_Gray060';mouse.data.materials[4]=mat;bs=next(n for n in mat.node_tree.nodes if n.type=='BSDF_PRINCIPLED');bs.inputs['Base Color'].default_value=(.6,.6,.6,1)
    after,check=snapshot();assert np.array_equal(before,after) and reference==check
    prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='OPTIX';prefs.get_devices();devices=[]
    for d in prefs.devices:
        d.use=d.type=='OPTIX'
        if d.use:devices.append(d.name)
    assert devices;s.cycles.device='GPU';s.cycles.samples=128;s.render.resolution_percentage=100
    s.camera=bpy.data.objects['Replica_FR2_camF'];r=float(s.camera['replica_pixel_aspect_y_over_x']);s.render.pixel_aspect_x=max(1.,1/r);s.render.pixel_aspect_y=max(1.,r);s.render.filepath='//renders/'+pose+'/camF_'
    target=project/f'PyMouseStudio-EarStudy-{label}-Static.blend';assert not target.exists();bpy.ops.wm.save_as_mainfile(filepath=str(target),compress=True)
    bpy.ops.wm.open_mainfile(filepath=str(target));s=bpy.context.scene;after,check=snapshot();error=float(np.abs(before-after).max());assert error==0 and reference==check
    mouse=bpy.data.objects['MS_Mouse'];ids=sorted({i for p in mouse.data.polygons if p.material_index==4 for i in p.vertices})
    record=dict(source=str(source),source_sha256=sha(source),delivery_blend=str(target),delivery_sha256=sha(target),frame=s.frame_current,skin_max_change_mm=error,scene_bone_camera_light_non_ear_material_unchanged=True,ear_other_inputs_unchanged=True,devices=devices,ear_rois={},renders=[])
    manifest['poses'][pose]=record;write()
    for c in 'ACEF':
        assert dt.datetime.now(dt.timezone.utc)<deadline
        camera=bpy.data.objects['Replica_FR2_cam'+c];s.camera=camera;r=float(camera['replica_pixel_aspect_y_over_x']);s.render.pixel_aspect_x=max(1.,1/r);s.render.pixel_aspect_y=max(1.,r)
        q=[world_to_camera_view(s,camera,Vector(after[i])) for i in ids];record['ear_rois'][c]=[max(0,int(min(v.x for v in q)*1080)-25),max(0,int((1-max(v.y for v in q))*1200)-25),min(1080,int(max(v.x for v in q)*1080)+26),min(1200,int((1-min(v.y for v in q))*1200)+26)]
        folder=out/pose;folder.mkdir(exist_ok=True);s.render.filepath=str(folder/f'cam{c}.png');t=time.monotonic();bpy.ops.render.render(write_still=True);record['renders'].append(dict(camera=c,seconds=time.monotonic()-t));write()
    assert sha(source)==plan['source_sha256'][pose] and sha(target)==record['delivery_sha256']
manifest['complete']=True;write();print('EAR_FINALIST_VALIDATED')
