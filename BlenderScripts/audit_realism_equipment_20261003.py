import bpy,json,sys
from pathlib import Path
from mathutils import Vector
root=Path(sys.argv[sys.argv.index('--')+1]);out=root/'output/pymousestudio_realism_cycle_20261002/equipment_audit.json';assert not out.exists()
def row(o):
    points=[o.matrix_world@Vector(v) for v in o.bound_box] if o.type=='MESH' else []
    return dict(name=o.name,type=o.type,collections=[c.name for c in o.users_collection],hide_render=o.hide_render,location=list(o.matrix_world.translation),dimensions=list(o.dimensions),bounds_min=[min(p[i] for p in points) for i in range(3)] if points else None,bounds_max=[max(p[i] for p in points) for i in range(3)] if points else None)
data=dict(blend=bpy.data.filepath,collections=[dict(name=c.name,objects=len(c.objects),all_objects=len(c.all_objects),hide_render=c.hide_render,children=[v.name for v in c.children]) for c in bpy.data.collections],objects=[row(o) for o in bpy.context.scene.objects if o.type in ['MESH','CAMERA'] and o.name!='MS_Mouse'],cameras={c:dict(matrix=[list(r) for r in bpy.data.objects['Replica_FR2_cam'+c].matrix_world]) for c in 'ABCDEFG'})
out.write_text(json.dumps(data,indent=2));print('EQUIPMENT_AUDIT_COMPLETE',len(data['objects']),flush=True)
