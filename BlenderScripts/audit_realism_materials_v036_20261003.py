"""Read-only audit of current implant/glass shaders and lights."""
import bpy,json,sys
from pathlib import Path
from mathutils import Vector
root=Path(sys.argv[sys.argv.index('--')+1]);out=root/'output/pymousestudio_realism_cycle_20261002/v036_material_audit';out.mkdir(exist_ok=False)
def val(v):
    try:return list(v)
    except TypeError:return v if isinstance(v,(str,int,float,bool)) else str(v)
def mat(m):
    return dict(name=m.name,nodes=[dict(name=n.name,type=n.bl_idname,inputs={i.name:dict(value=val(i.default_value),links=[l.from_node.name+':'+l.from_socket.name for l in i.links]) for i in n.inputs if hasattr(i,'default_value')}) for n in m.node_tree.nodes] if m.use_nodes else [])
objects={}
for ob in bpy.data.objects:
    if ob.name.startswith('ReplicaAsset_') or ob.name in ['Part_GlassCylinder160mm','Part_AntiReflectiveGlass','Part_TopCover']:
        points=[ob.matrix_world@Vector(x) for x in ob.bound_box];objects[ob.name]=dict(type=ob.type,parent=ob.parent.name if ob.parent else None,parent_bone=ob.parent_bone,bounds=[[min(x[i] for x in points) for i in range(3)],[max(x[i] for x in points) for i in range(3)]],materials=[mat(m) for m in ob.data.materials if m],hide_render=ob.hide_render)
lights={ob.name:dict(type=ob.data.type,energy=ob.data.energy,position=list(ob.matrix_world.translation),hide_render=ob.hide_render,settings={p:getattr(ob.data,p) for p in ['shadow_soft_size','specular_factor','diffuse_factor','transmission_factor','volume_factor','spot_size','spot_blend'] if hasattr(ob.data,p)}) for ob in bpy.data.objects if ob.type=='LIGHT'}
(out/'audit.json').write_text(json.dumps(dict(source=bpy.data.filepath,frame=bpy.context.scene.frame_current,objects=objects,lights=lights),ensure_ascii=False,indent=2))
