import bpy,json,sys,hashlib,time
from pathlib import Path
from bpy_extras.object_utils import world_to_camera_view
import numpy as np
root=Path(sys.argv[-1]);base=root/'output/pymousestudio_ear_cycle_20261003';out=base/'audit_baseline';out.mkdir(exist_ok=True)
project=root/'BlenderPyMouseProjects/Cylinder/PyMouseStudio-RealCapture-EarStudy-20261003'
s=bpy.context.scene;mouse=bpy.data.objects['MS_Mouse'];s.frame_set(31);bpy.context.view_layer.update()
def value(v):
    try:return list(v)
    except TypeError:return v if isinstance(v,(str,float,int,bool)) else str(v)
def mat(m):return dict(name=m.name,users=m.users,nodes=[dict(name=n.name,type=n.bl_idname,inputs={i.name:dict(value=value(i.default_value),links=[l.from_node.name+':'+l.from_socket.name for l in i.links]) for i in n.inputs if hasattr(i,'default_value')}) for n in m.node_tree.nodes])
slots=[dict(slot=i,name=m.name,faces=sum(p.material_index==i for p in mouse.data.polygons)) for i,m in enumerate(mouse.data.materials)]
print('MOUSE_MATERIAL_SLOTS='+json.dumps(slots),flush=True)
(out/'slots.json').write_text(json.dumps(slots,indent=2))
ear=mouse.data.materials[4];assert ear.name=='Replica_BalancedSkin_4',ear.name
ids=sorted({i for p in mouse.data.polygons if p.material_index==4 for i in p.vertices});assert len(ids)>10
ev=mouse.evaluated_get(bpy.context.evaluated_depsgraph_get());me=ev.to_mesh();world=np.array([ev.matrix_world@v.co for v in me.vertices]);ev.to_mesh_clear()
np.save(out/'skin_world_before.npy',world)
audit=dict(source=bpy.data.filepath,ear_material=mat(ear),slots=slots,ear_vertex_ids=ids,ear_faces=sum(p.material_index==4 for p in mouse.data.polygons),ear_material_users=[o.name for o in bpy.data.objects if hasattr(o.data,'materials') and ear in list(o.data.materials)],geometry_sha256=hashlib.sha256(world.tobytes()).hexdigest(),frame=s.frame_current,camera_matrices={c:[list(r) for r in bpy.data.objects['Replica_FR2_cam'+c].matrix_world] for c in 'ABCDEFG'},ear_rois={},renders=[])
prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='OPTIX';prefs.get_devices();enabled=[]
for d in prefs.devices:
    d.use=d.type=='OPTIX'
    if d.use:enabled.append(d.name)
assert enabled;audit['devices']=enabled
s.render.engine='CYCLES';s.cycles.device='GPU';s.cycles.samples=128;s.cycles.use_denoising=True;s.render.resolution_percentage=100
for c in 'ABCDEFG':
    camera=bpy.data.objects['Replica_FR2_cam'+c];s.camera=camera;r=float(camera['replica_pixel_aspect_y_over_x']);s.render.pixel_aspect_x=max(1.,1/r);s.render.pixel_aspect_y=max(1.,r)
    from mathutils import Vector
    q=[world_to_camera_view(s,camera,Vector(world[i])) for i in ids]
    audit['ear_rois'][c]=[max(0,int(min(v.x for v in q)*1080)-30),max(0,int((1-max(v.y for v in q))*1200)-30),min(1080,int(max(v.x for v in q)*1080)+31),min(1200,int((1-min(v.y for v in q))*1200)+31)]
    if c in 'CF':
        s.render.filepath=str(out/f'cam{c}.png');t=time.monotonic();bpy.ops.render.render(write_still=True);audit['renders'].append(dict(camera=c,seconds=time.monotonic()-t))
s.camera=bpy.data.objects['Replica_FR2_camF'];r=float(s.camera['replica_pixel_aspect_y_over_x']);s.render.pixel_aspect_x=max(1.,1/r);s.render.pixel_aspect_y=max(1.,r)
target=project/'EarStudy-v000-baseline-climb.blend';assert not target.exists();bpy.ops.wm.save_as_mainfile(filepath=str(target),compress=True);audit['baseline_blend']=str(target);audit['complete']=True
(out/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
print('EAR_AUDIT_BASELINE_COMPLETE')
