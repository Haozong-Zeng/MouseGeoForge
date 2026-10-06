"""Read-only existing Blender geometry to candidate curve sidecars; never rerenders/saves."""
import bpy,sys,json,hashlib,datetime as dt
from pathlib import Path
import numpy as np
ROOT=Path(sys.argv[-1]);OUT=ROOT/'output/surface_curve_cycle_20261003/diverse_curve_gt_v1'
PROTOCOL=ROOT/'output/surface_curve_cycle_20261003/training_v1/protocol.json'
ROLE_FILTER=None
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def topo(mesh):return hashlib.sha256(json.dumps([list(p.vertices) for p in mesh.polygons]).encode()).hexdigest()
def main():
    assert not (OUT/'manifest.json').exists();OUT.mkdir(exist_ok=True)
    cfg=json.loads(PROTOCOL.read_text())
    jobs=json.loads((ROOT/'output/surface_keypoint_migration_v3/inventory.json').read_text())['jobs'];jobmap={j['dataset']:j for j in jobs}
    plan=cfg['synthetic_train']+cfg['synthetic_development'];groups={}
    if ROLE_FILTER is not None:plan=[r for r in plan if r['role']==ROLE_FILTER]
    for row in plan:
        name=Path(row['scene']).parent.name;assert name in jobmap,name;groups.setdefault(name,[]).append(row)
    proposal=json.loads((ROOT/'output/surface_anchor_demo_v1/rest_surface_anchors.json').read_text())
    with np.load(ROOT/'dataset/BlenderPyMouse/PyMouseStudio-RealCapture-SurfaceCurve-Multiview/frame_000001/geometry.npz') as a:
        reference=a['rest_vertices'];tri_ids=a['curve_triangle_ids'];weights=a['curve_barycentric_weights']
    assert tri_ids.shape==(65,3) and weights.shape==(65,3)
    records=[];blends=[]
    for name,rows in sorted(groups.items()):
        assert dt.datetime.now(dt.timezone.utc)<dt.datetime(2026,10,4,8,5,16,tzinfo=dt.timezone.utc)
        job=jobmap[name];source=Path(job['source_blend']);before=sha(source);bpy.ops.wm.open_mainfile(filepath=str(source))
        for obj in bpy.data.objects:
            if obj.type=='CURVES':obj.hide_render=True;obj.hide_viewport=True
        bpy.context.view_layer.update();meshes=[bpy.data.objects[n] for n in job['mesh_names']];rest_errors=[]
        for obj in meshes:
            assert topo(obj.data)==proposal['topology_sha256']
            rest=np.array([v.co for v in obj.data.vertices]);X=np.c_[reference,np.ones(len(reference))]
            A=np.linalg.lstsq(X,rest,rcond=None)[0];error=np.linalg.norm(X@A-rest,axis=-1).max()/np.linalg.norm(np.ptp(rest,axis=0))
            assert error<1e-5 and np.linalg.det(A[:3])>0,(name,obj.name,error);rest_errors.append(float(error))
        for row in rows:
            scene=ROOT/row['scene'];path=scene/'keypoints.npz'
            with np.load(path) as a:kp=dict(a)
            frame=int(kp['frame']);mcount=len(meshes);old=kp['support_world_xyz'];ids=kp['support_vertex_ids'];joints=kp['joint_world_xyz']
            if mcount==1:old=old[None];ids=ids[None];joints=joints[None]
            curves=[];errors=[];anchor_errors=[]
            bpy.context.scene.frame_set(frame);bpy.context.view_layer.update()
            for m,obj in enumerate(meshes):
                bpy.context.scene.frame_set(frame);bpy.context.view_layer.update()
                ev=obj.evaluated_get(bpy.context.evaluated_depsgraph_get());em=ev.to_mesh()
                try:
                    assert len(em.vertices)==len(reference)
                    xyz=np.array([ev.matrix_world@v.co for v in em.vertices])
                finally:ev.to_mesh_clear()
                delta=np.linalg.norm(xyz[ids[m]]-old[m],axis=-1);errors.append(float(np.nanmax(delta)))
                current=np.array([np.array(a['barycentric_weights'])@xyz[a['triangle_vertex_ids']] for a in proposal['anchors']])
                actual=joints[m,[a['core23_id'] for a in proposal['anchors']]]
                anchor_errors.append(float(np.nanmax(np.linalg.norm(current-actual,axis=-1))))
                curves.append(np.einsum('si,sij->sj',weights,xyz[tri_ids]))
            world=np.array(curves);uv=[];cz=[]
            for E,K in zip(kp['extrinsics'].astype(float),kp['intrinsics'].astype(float)):
                p=world@E[:3,:3].T+E[:3,3];q=p@K.T;uv.append(q[...,:2]/q[...,2:3]);cz.append(p[...,2])
            uv=np.array(uv);cz=np.array(cz);valid=np.array([e<.002 and a<.002 for e,a in zip(errors,anchor_errors)])
            # Fail closed per SCENE for supervised use if any source instance mismatches.
            accepted=bool(valid.all() and np.isfinite(uv).all() and (cz>0).all())
            folder=OUT/name/scene.name;folder.mkdir(parents=True,exist_ok=False)
            np.savez_compressed(folder/'surface_curve.npz',world_xyz=world.astype(np.float32),uv=uv.astype(np.float32),camera_z=cz.astype(np.float32),
                position_valid=np.full(cz.shape,accepted,bool),geometry_valid=valid,view_ids=kp['view_ids'],instance_ids=kp.get('instance_ids',np.arange(mcount)),
                triangle_vertex_ids=tri_ids,barycentric_weights=weights,source_annotation_sha256=np.asarray(sha(path)),source_blend_sha256=np.asarray(before),
                visibility_valid=np.zeros(cz.shape,bool),definition=np.asarray('65fixed_rest_mesh_dorsal_surface_material_samples_v1'))
            record=dict(scene=row['scene'],role=row['role'],source_group=row['source_group'],dataset=name,source_frame=frame,
                sidecar=(folder/'surface_curve.npz').relative_to(ROOT).as_posix(),source_blend=str(source),source_blend_sha256=before,source_annotation_sha256=sha(path),
                max_support_error=errors,max_anchor_error=anchor_errors,geometry_valid=valid.tolist(),accepted_for_training=accepted,
                views=kp['view_ids'].tolist(),instances=mcount,units='same_world_units_as_source_keypoints_and_camera_translation',
                visibility='unknown_scene_visibility; positions are exact amodal synthetic surface geometry, never labeled directly visible')
            records.append(record);save(OUT/'partial_manifest.json',dict(records=records));print('CURVE_EXPORT',name,scene.name,accepted,errors,flush=True)
        assert sha(source)==before,'Source blend was modified';blends.append(dict(path=str(source),sha256=before,rest_affine_relative_errors=rest_errors))
    save(OUT/'manifest.json',dict(status='candidate_needs_independent_projection_and_visual_QA',records=records,blends=blends,
        source_protocol_sha256=sha(PROTOCOL),
        topology_sha256=proposal['topology_sha256'],definition='Reuse EXACT triangle IDs/barycentric coordinates from published3pose curve dataset; validate rest mesh affine and ALLCore23support geometry against each source render annotation',
        labels_are_not_internal_spine=True,temporal_supervision=False,production_dataset_modified=False,visibility_supervision=False))
    print('DIVERSE_CURVE_EXPORT_COMPLETE',len(records),sum(r['accepted_for_training'] for r in records),flush=True)
if __name__=='__main__':main()
