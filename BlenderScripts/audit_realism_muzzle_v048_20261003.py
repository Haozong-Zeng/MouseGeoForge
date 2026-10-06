"""Read-only geometry/rig audit. No save, no scene edits, no label writes."""
import bpy,json,sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[sys.argv.index('--')+1]);out=root/'output/pymousestudio_realism_cycle_20261002/v048_muzzle_audit';out.mkdir(exist_ok=False);s=bpy.context.scene;ob=bpy.data.objects['MS_Mouse'];arm=bpy.data.objects['DeformationSystem'];bpy.context.view_layer.update()
rest=np.array([v.co[:] for v in ob.data.vertices]);ev=ob.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=ev.to_mesh();world=np.array([ev.matrix_world@v.co for v in mesh.vertices]);ev.to_mesh_clear()
weights=np.zeros((len(ob.data.vertices),len(ob.vertex_groups)),np.float32)
for v in ob.data.vertices:
    for g in v.groups:weights[v.index,g.group]=g.weight
faces=[list(p.vertices) for p in ob.data.polygons];groups=[g.name for g in ob.vertex_groups];selected=[n for n in groups if any(x in n.lower() for x in ['jaw','head','lip','mouth','nose','cheek','tongue','tooth'])]
np.savez_compressed(out/'mesh.npz',rest=rest,world=world,weights=weights,group_names=np.array(groups),material_indices=np.array([p.material_index for p in ob.data.polygons]))
bones={}
for b in arm.pose.bones:
    if any(x in b.name.lower() for x in ['jaw','head','lip','mouth','nose','cheek','tongue','tooth']):
        bones[b.name]=dict(head_world=list(arm.matrix_world@b.head),tail_world=list(arm.matrix_world@b.tail),head_rest=list(b.bone.head_local),tail_rest=list(b.bone.tail_local),matrix_basis=[list(r) for r in b.matrix_basis],matrix=[list(r) for r in b.matrix],constraints=[dict(name=c.name,type=c.type,mute=c.mute,influence=c.influence,target=getattr(getattr(c,'target',None),'name',None),subtarget=getattr(c,'subtarget',None)) for c in b.constraints],parent=b.parent.name if b.parent else None)
objects=[]
for o in bpy.data.objects:
    if any(t in o.name.lower() for t in ['ms_','teeth','tooth','tongue','mouth','jaw']):
        objects.append(dict(name=o.name,type=o.type,hide_render=o.hide_render,parent=o.parent.name if o.parent else None,dimensions=list(o.dimensions),materials=[slot.material.name if slot.material else None for slot in o.material_slots],modifiers=[dict(name=m.name,type=m.type,show_render=m.show_render) for m in o.modifiers]))
keys=[]
if ob.data.shape_keys:
    for k in ob.data.shape_keys.key_blocks:keys.append(dict(name=k.name,value=k.value,mute=k.mute))
m=dict(source=bpy.data.filepath,frame=s.frame_current,vertices=len(rest),evaluated_vertices=len(world),faces=faces,groups=selected,weight_counts={n:int((weights[:,groups.index(n)]>.01).sum()) for n in selected},shape_keys=keys,bones=bones,objects=objects,mesh_world_matrix=[list(r) for r in ob.matrix_world],arm_world_matrix=[list(r) for r in arm.matrix_world],readonly=True)
(out/'audit.json').write_text(json.dumps(m,indent=2));print(json.dumps({k:v for k,v in m.items() if k not in ['faces','bones']},ensure_ascii=False))
