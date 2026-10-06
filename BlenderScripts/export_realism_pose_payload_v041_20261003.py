"""Read-only pose payload and evaluated-skin reference for exact scene integration."""
import bpy,json,sys,hashlib
from pathlib import Path
import numpy as np
root=Path(sys.argv[sys.argv.index('--')+1]);base=root/'output/pymousestudio_realism_cycle_20261002';tag=globals().get('POSE_TAG','quadruped');out=base/'v041_pose_integration'/tag;out.mkdir(parents=True,exist_ok=False)
s=bpy.context.scene;arm=bpy.data.objects['DeformationSystem'];mouse=bpy.data.objects['MS_Mouse'];group=bpy.data.objects['Group.004'];bpy.context.view_layer.update()
objects={x.name:x for x in [group,*group.children_recursive]}
for obj in [arm,mouse]:
    while obj is not None:objects[obj.name]=obj;obj=obj.parent
def rows(m):return [list(r) for r in m]
def props(x):return {k:v for k,v in x.items() if isinstance(v,(float,int,bool,str))}
ev=mouse.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=ev.to_mesh();points=np.array([ev.matrix_world@v.co for v in mesh.vertices],dtype=float);ev.to_mesh_clear()
np.savez_compressed(out/'skin_reference.npz',xyz_mm=points)
m=dict(source=bpy.data.filepath,source_sha256=hashlib.sha256(Path(bpy.data.filepath).read_bytes()).hexdigest(),frame=s.frame_current,pose_tag=tag,objects={n:dict(type=o.type,parent=o.parent.name if o.parent else None,parent_type=o.parent_type,parent_bone=o.parent_bone,matrix_basis=rows(o.matrix_basis),matrix_parent_inverse=rows(o.matrix_parent_inverse),matrix_world=rows(o.matrix_world),rotation_mode=o.rotation_mode) for n,o in objects.items()},arm_props=props(arm),bones={b.name:dict(basis=rows(b.matrix_basis),pose_matrix=rows(b.matrix),head_world=list(arm.matrix_world@b.head),rotation_mode=b.rotation_mode,props=props(b),constraints=[dict(name=c.name,type=c.type,mute=c.mute,influence=c.influence) for c in b.constraints]) for b in arm.pose.bones},armature_modifiers={o.name:{mod.name:dict(preserve_volume=mod.use_deform_preserve_volume,show_viewport=mod.show_viewport,show_render=mod.show_render) for mod in o.modifiers if mod.type=='ARMATURE'} for o in objects.values() if hasattr(o,'modifiers')},skin_vertex_count=len(points),scene_props=props(s),limitations=['Static pose payload only. No camera/material/lighting copied. Evaluated skin is a geometry regression reference, not real groundtruth.'])
(out/'pose.json').write_text(json.dumps(m,ensure_ascii=False,indent=2));print(json.dumps(dict(tag=tag,frame=s.frame_current,objects=len(objects),bones=len(m['bones']),skin_vertices=len(points)),ensure_ascii=False))
