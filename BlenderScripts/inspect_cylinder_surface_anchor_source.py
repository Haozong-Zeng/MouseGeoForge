"""Read-only Blender inspection for the surface-landmark demo."""
import bpy
import json
import sys
from pathlib import Path

out = Path(sys.argv[sys.argv.index('--')+1])
mesh = bpy.data.objects['MS_Mouse']
arm = bpy.data.objects['DeformationSystem']
names = ['Head1_M', 'HeadEnd_M', 'Neck3_M', 'Spine3_M', 'Spine1_M', 'Tail0_M']
report = dict(source=bpy.data.filepath, frame=bpy.context.scene.frame_current,
    mesh=dict(vertices=len(mesh.data.vertices), polygons=len(mesh.data.polygons), matrix=[list(r) for r in mesh.matrix_world],
        modifiers=[dict(name=m.name,type=m.type,viewport=m.show_viewport,render=m.show_render) for m in mesh.modifiers],
        bounds=[[min(v.co[i] for v in mesh.data.vertices),max(v.co[i] for v in mesh.data.vertices)] for i in range(3)]),
    armature_matrix=[list(r) for r in arm.matrix_world],
    bones={n:dict(head=list(arm.data.bones[n].head_local),tail=list(arm.data.bones[n].tail_local)) for n in names},
    render=dict(engine=bpy.context.scene.render.engine, x=bpy.context.scene.render.resolution_x,y=bpy.context.scene.render.resolution_y),
    objects=[dict(name=o.name,type=o.type,hide_render=o.hide_render) for o in bpy.data.objects])
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='objects'},indent=2))
