import bpy, json, sys
from pathlib import Path
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
root=Path(sys.argv[-1]);out=root/'output/bottom_tail_fix_20261003';out.mkdir(exist_ok=True)
s=bpy.context.scene;mouse=bpy.data.objects['MS_Mouse'];s.camera=bpy.data.objects['Replica_FR2_camA'];r=float(s.camera['replica_pixel_aspect_y_over_x']);s.render.pixel_aspect_x=max(1.,1/r);s.render.pixel_aspect_y=max(1.,r)
ev=mouse.evaluated_get(bpy.context.evaluated_depsgraph_get());me=ev.to_mesh();v=np.array([ev.matrix_world@v.co for v in me.vertices]);tail_ids=sorted({i for p in me.polygons if p.material_index==5 for i in p.vertices});np.save(out/'skin_before.npy',v)
objects=[]
for o in bpy.data.objects:
    if o.type!='MESH':continue
    from mathutils import Vector
    bb=np.array([o.matrix_world@Vector(c) for c in o.bound_box])
    objects.append(dict(name=o.name,hide_render=o.hide_render,bounds=[bb.min(0).tolist(),bb.max(0).tolist()],materials=[m.name if m else None for m in o.data.materials]))
tail=v[tail_ids];rows=[]
for i in tail_ids:
    p=world_to_camera_view(s,s.camera,ev.matrix_world@me.vertices[i].co)
    rows.append([i,*v[i].tolist(),float(p.x*1080),float((1-p.y)*1200)])
mat=mouse.data.materials[5]
report=dict(objects=objects,tail_bounds=[tail.min(0).tolist(),tail.max(0).tolist()],tail_vertices=rows,tail_material=mat.name,nodes=[dict(name=n.name,type=n.type,inputs={i.name:str(i.default_value) for i in n.inputs if hasattr(i,'default_value')}) for n in mat.node_tree.nodes],modifiers=[dict(name=m.name,type=m.type) for m in mouse.modifiers])
(out/'audit.json').write_text(json.dumps(report,indent=2));ev.to_mesh_clear();print('TAIL_AUDIT_COMPLETE')
