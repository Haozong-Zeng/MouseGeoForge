"""Read-only source-scene extraction: head-local detached mesh assets, no source animation."""
import bpy,json,sys,hashlib
from pathlib import Path
from mathutils import Matrix
root=Path(sys.argv[sys.argv.index('--')+1]);out=root/'output/pymousestudio_realism_cycle_20261002/head_asset';out.mkdir(exist_ok=False)
source=Path(bpy.data.filepath);source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
s=bpy.context.scene;s.frame_set(31);bpy.context.view_layer.update();dg=bpy.context.evaluated_depsgraph_get()
arm=bpy.data.objects['DeformationSystem.001'];head=arm.matrix_world@arm.pose.bones['Head1_M'].matrix;inverse=head.inverted();objects=[];rows=[]
for name in ['MouseFiberWindow','立方体']:
    ob=bpy.data.objects.get(name);assert ob is not None and ob.type=='MESH',name
    ev=ob.evaluated_get(dg);mesh=bpy.data.meshes.new_from_object(ev,preserve_all_data_layers=True,depsgraph=dg)
    mesh.transform(inverse@ev.matrix_world);new=bpy.data.objects.new('ReplicaAsset_'+name,mesh);new.matrix_world=Matrix.Identity(4)
    objects.append(new);world=[ev.matrix_world@v.co for v in ev.data.vertices];local=[v.co for v in mesh.vertices]
    rows.append(dict(source_name=name,asset_name=new.name,vertices=len(mesh.vertices),faces=len(mesh.polygons),source_parent=ob.parent.name if ob.parent else None,parent_type=ob.parent_type,parent_bone=ob.parent_bone,hide_render=ob.hide_render,
        source_world_dimensions=[max(p[k] for p in world)-min(p[k] for p in world) for k in range(3)],head_local_bounds=[[min(p[k] for p in local) for k in range(3)],[max(p[k] for p in local) for k in range(3)]],materials=[m.name for m in mesh.materials if m]))
target=out/'head_mount_local.blend';bpy.data.libraries.write(str(target),set(objects),fake_user=True,compress=True)
assert hashlib.sha256(source.read_bytes()).hexdigest()==source_hash
(out/'manifest.json').write_text(json.dumps(dict(source_blend=str(source),source_sha256=source_hash,source_frame=31,source_head_matrix=[list(r) for r in head],asset_blend=str(target),objects=rows,source_unchanged=True,exported_animation=False),ensure_ascii=False,indent=2))
print(json.dumps(rows,ensure_ascii=False,indent=2),flush=True)
