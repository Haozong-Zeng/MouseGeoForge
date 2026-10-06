"""Read-only Blender relocation and mesh/camera audit for the new dataset."""
import bpy,sys,json,hashlib
from pathlib import Path
import numpy as np
root=Path(sys.argv[-1]);out=root/'output/surface_curve_cycle_20261003';base=root/'BlenderPyMouseProjects/Cylinder/PyMouseStudio-RealCapture-Final'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
manifest=json.loads((base/'final_manifest.json').read_text());rows=[]
for record in manifest['records']:
    path=root/record['final'];assert sha(path)==record['sha256'];bpy.ops.wm.open_mainfile(filepath=str(path));s=bpy.context.scene
    images=[dict(name=i.name,source=i.source,packed=bool(i.packed_file or i.packed_files),path=i.filepath) for i in bpy.data.images if i.source=='FILE']
    libraries=[dict(name=l.name,path=l.filepath) for l in bpy.data.libraries]
    missing=[i for i in images if not i['packed'] and not Path(bpy.path.abspath(i['path'])).is_file()]
    assert not missing and not libraries,(missing,libraries)
    obj=bpy.data.objects['MS_Mouse'];arm=bpy.data.objects['DeformationSystem'];bpy.context.view_layer.update();ev=obj.evaluated_get(bpy.context.evaluated_depsgraph_get());me=ev.to_mesh()
    try:
        v=np.array([ev.matrix_world@p.co for p in me.vertices]);faces=[list(p.vertices) for p in me.polygons]
        np.savez_compressed(out/(record['pose']+'_mesh.npz'),vertices=v,rest_vertices=np.array([v.co for v in obj.data.vertices]),edges=np.array([e.vertices[:] for e in obj.data.edges]),matrix_world=np.array(obj.matrix_world))
        evaluated_count=len(me.vertices)
    finally:ev.to_mesh_clear()
    cams=[]
    for c in 'ABCDEFG':
        cam=bpy.data.objects.get('Replica_FR2_cam'+c)
        if cam:cams.append(dict(camera=c,type=cam.data.type,world=np.array(cam.matrix_world).tolist(),props={k:v for k,v in cam.items() if isinstance(v,(str,int,float))}))
    rows.append(dict(**record,frame_current=s.frame_current,image_wh=[s.render.resolution_x,s.render.resolution_y],packed_images=sum(i['packed'] for i in images),image_files=images,libraries=libraries,cameras=cams,rest_vertices=len(obj.data.vertices),evaluated_vertices=evaluated_count,topology_sha256=hashlib.sha256(json.dumps([list(p.vertices) for p in obj.data.polygons]).encode()).hexdigest(),modifiers=[dict(name=m.name,type=m.type,viewport=m.show_viewport,render=m.show_render) for m in obj.modifiers],objects=[dict(name=o.name,type=o.type,hide_render=o.hide_render) for o in bpy.data.objects],bones=[b.name for b in arm.pose.bones]))
    assert sha(path)==record['sha256']
    print('PREFLIGHT',record['pose'],len(cams),len(obj.data.vertices),evaluated_count,flush=True)
all_packed=all(i['packed'] for r in rows for i in r['image_files'])
(out/'blender_preflight.json').write_text(json.dumps(dict(complete=True,all_dependencies_packed=all_packed,poses=rows),indent=2))
assert all_packed,'Archive blocked by unpacked dependency'
print('PREFLIGHT_COMPLETE',flush=True)
