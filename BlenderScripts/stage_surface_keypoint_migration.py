"""Stage topology-verified 06/07/23 geometry; never modifies datasets/blends."""
import hashlib
import json
import sys
from pathlib import Path
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

KIDS=[5,6,22]


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def topo(mesh):return hashlib.sha256(json.dumps([list(p.vertices) for p in mesh.polygons]).encode()).hexdigest()


def main():
    args=sys.argv[sys.argv.index('--')+1:];root=Path(args[0]);name=args[1]
    base=root/'output/surface_keypoint_migration_v3';out=base/'staged'/name;out.mkdir(parents=True,exist_ok=True)
    job=next(j for j in json.loads((base/'inventory.json').read_text())['jobs'] if j['dataset']==name)
    ds=root/'dataset/BlenderPyMouse'/name
    proposal=json.loads((root/'output/surface_anchor_demo_v1/rest_surface_anchors.json').read_text());anchors=proposal['anchors']
    meshes=[bpy.data.objects[n] for n in job['mesh_names']]
    for obj in bpy.data.objects:
        if obj.type=='CURVES':obj.hide_viewport=True;obj.hide_render=True
    bpy.context.view_layer.update()
    refpath=base/'reference_rest_vertices.npz'
    if name=='PyMouseStudio-BrownFur-WalkAround-Core23':
        ref=np.array([v.co for v in meshes[0].data.vertices],dtype=np.float64)
        assert topo(meshes[0].data)==proposal['topology_sha256']
        np.savez_compressed(refpath,vertices=ref)
    else:
        with np.load(refpath) as z:ref=z['vertices']
    geometry=[];support_ids=[]
    for obj in meshes:
        assert topo(obj.data)==proposal['topology_sha256'],f'Topology mismatch {obj.name}'
        xyz=np.array([v.co for v in obj.data.vertices],dtype=np.float64)
        # Applied object transforms may have changed stored rest units/orientation.
        A=np.linalg.lstsq(np.c_[ref,np.ones(len(ref))],xyz,rcond=None)[0]
        error=np.linalg.norm(np.c_[ref,np.ones(len(ref))]@A-xyz,axis=-1).max()
        relative=error/np.linalg.norm(np.ptp(xyz,axis=0))
        assert relative<1e-5 and np.linalg.det(A[:3])>0,(obj.name,relative)
        ids=[]
        for a in anchors:
            center=np.array(a['barycentric_weights'])@xyz[a['triangle_vertex_ids']]
            chosen=list(a['triangle_vertex_ids'])
            chosen+=sorted(set(a['region_vertex_ids'])-set(chosen),key=lambda i:np.linalg.norm(xyz[i]-center))[:13]
            assert len(chosen)==16;ids.append(chosen)
        support_ids.append(ids)
        geometry.append(dict(mesh=obj.name,topology_verified=True,rest_affine_relative_error=float(relative),rest_affine=A.tolist()))
    source=Path(bpy.data.filepath);source_hash=sha(source);reports=[]
    for file in sorted(ds.glob('frame_*/keypoints.npz')):
        with np.load(file,allow_pickle=False) as z:a=dict(z)
        if str(a.get('definition_version',''))=='pymousestudio_manual_surface_core23_v3':raise ValueError('Already migrated source')
        mcount=len(meshes);world=[];support=[];trees=[];local_errors=[];full_errors=[];bones_errors=[]
        frame=int(a['frame']);old_support=a['support_world_xyz'];old_joint=a['joint_world_xyz']
        if mcount==1:old_support=old_support[None];old_joint=old_joint[None]
        # The original exporter evaluates once outside, then once per instance.
        bpy.context.scene.frame_set(frame);bpy.context.view_layer.update()
        for m,(obj,armname) in enumerate(zip(meshes,job['armature_names'])):
            bpy.context.scene.frame_set(frame);bpy.context.view_layer.update()
            dg=bpy.context.evaluated_depsgraph_get();ev=obj.evaluated_get(dg);em=ev.to_mesh()
            try:
                assert len(em.vertices)==len(ref)
                xyz=np.array([ev.matrix_world@v.co for v in em.vertices],dtype=np.float64)
                faces=[list(p.vertices) for p in em.polygons]
            finally:ev.to_mesh_clear()
            ids=a['support_vertex_ids'] if mcount==1 else a['support_vertex_ids'][m]
            diff=np.linalg.norm(xyz[ids]-old_support[m],axis=-1)
            local_errors.append(float(diff[KIDS].max()));full_errors.append(float(diff.max()))
            arm=bpy.data.objects[armname]
            b=np.array([arm.matrix_world@arm.pose.bones[str(a['bones'][k])].head for k in KIDS])
            bones_errors.append(float(np.linalg.norm(b-old_joint[m,KIDS],axis=-1).max()))
            # Reject wrong frame/rig rather than updating based on a nearby pose.
            # Fail closed per instance, retaining candidates only for inspection.
            # The writer must disable all affected supervision on failed instances.
            world.append([np.array(t['barycentric_weights'])@xyz[t['triangle_vertex_ids']] for t in anchors])
            support.append(xyz[support_ids[m]])
            trees.append(BVHTree.FromPolygons([Vector(v) for v in xyz],faces))
        world=np.array(world);support=np.array(support)
        uv=[];camera_xyz=[];body_visible=[];cause=[];support_counts=[]
        for E,K in zip(a['extrinsics'].astype(float),a['intrinsics'].astype(float)):
            camera=(world@E[:3,:3].T)+E[:3,3];project=camera@K.T
            xy=project[...,:2]/project[...,2:3]
            origin=Vector((-E[:3,:3].T@E[:3,3]).tolist())
            vvis=[];vcause=[];vcounts=[]
            for m in range(mcount):
                vis=[];causes=[];counts=[]
                for k in range(3):
                    target=Vector(world[m,k]);ray=target-origin;distance=ray.length
                    hits=[]
                    eps=max(.0002,distance*2e-6)
                    for j,tree in enumerate(trees):
                        hit=tree.ray_cast(origin,ray.normalized(),max(0,distance-eps))
                        if hit[0] is not None:hits.append((hit[3],j))
                    blocker=min(hits)[1] if hits else -1
                    vis.append(blocker<0);causes.append(0 if blocker<0 else 1 if blocker==m else 2)
                    count=0
                    for p in support[m,k]:
                        ray=Vector(p)-origin;distance=ray.length
                        count+=not any(tree.ray_cast(origin,ray.normalized(),max(0,distance-max(.0002,distance*2e-6)))[0] is not None for tree in trees)
                    counts.append(count)
                vvis.append(vis);vcause.append(causes);vcounts.append(counts)
            uv.append(xy);camera_xyz.append(camera);body_visible.append(vvis);cause.append(vcause);support_counts.append(vcounts)
        stage=out/(file.parent.name+'.npz')
        with stage.open('wb') as f:np.savez_compressed(f,world=np.array(world,dtype=np.float32),support=np.array(support,dtype=np.float32),
            support_ids=np.array(support_ids,dtype=np.int32),uv=np.array(uv,dtype=np.float32),camera_xyz=np.array(camera_xyz,dtype=np.float32),
            body_visible=np.array(body_visible,dtype=bool),occlusion_cause=np.array(cause,dtype=np.int8),support_visible_count=np.array(support_counts,dtype=np.int16),
            full_geometry_verified=np.asarray(max(full_errors)<.002),source_annotation_sha256=np.asarray(sha(file)),
            geometry_valid=np.asarray([(x<.002 and b<.002) for x,b in zip(local_errors,bones_errors)],dtype=bool),
            frame=np.asarray(frame),source_blend_sha256=np.asarray(source_hash))
        reports.append(dict(folder=file.parent.name,frame=frame,local_support_error=max(local_errors),bone_error=max(bones_errors),full_support_error=max(full_errors),invalid_instances=[i for i,(x,b) in enumerate(zip(local_errors,bones_errors)) if x>=.002 or b>=.002]))
        if len(reports)%25==0:print('STAGED',name,len(reports),flush=True)
    report=dict(dataset=name,source_blend=str(source),source_blend_sha256=source_hash,geometry=geometry,anchors=anchors,
        frames=reports,status='staged',max_local_support_error=max(r['local_support_error'] for r in reports),
        uncertain_visibility_frames=[r['folder'] for r in reports if r['full_support_error']>=.002])
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print('STAGING_COMPLETE',name,len(reports),len(report['uncertain_visibility_frames']),flush=True)


if __name__=='__main__':main()
