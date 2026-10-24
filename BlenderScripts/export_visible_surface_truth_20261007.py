"""Deterministic, CPU, opaque skin-surface truth with one shared primary ray hit.

This diagnostic exporter never saves a blend or overwrites dataset labels.
Material transparency/refraction and hair are deliberately excluded: this is
an explicit geometric skin surface, not a claim about the RGB hair silhouette.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_visible_surface_geometry_20261007 import camera_parameters, mesh_arrays, write


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def fixture():
    # A tilted background z = 5 + .2*x and a front square z=3. Analytic
    # intersections, silhouette/occlusion and camera-Z versus ray length.
    vertices = [(-10,-10,3), (10,-10,7), (10,10,7), (-10,10,3),
                (-.4,-.4,3), (.4,-.4,3), (.4,.4,3), (-.4,.4,3)]
    tree = BVHTree.FromPolygons(vertices, [(0,1,2),(0,2,3),(4,5,6),(4,6,7)], all_triangles=True)
    errors=[]
    for x,y in [(0,0),(.3,0),(-.4,.2),(.05,-.03)]:
        ray=Vector((x,y,1));ray.normalize()
        hit,_,face,distance=tree.ray_cast(Vector((0,0,0)),ray)
        expected=3 if abs(x*3)<.4 and abs(y*3)<.4 else 5/(1-.2*x)
        errors.append(abs(hit.z-expected))
        assert (face>=2)==(expected==3)
        assert abs(distance*np.asarray(ray)[2]-expected)<2e-6
    assert max(errors)<2e-6
    # The near clip is a camera-Z plane, not a spherical radius or a filter
    # after taking the closest hit. A clipped occluder must expose the next hit.
    hit,_,face,_=tree.ray_cast(Vector((0,0,4)),Vector((0,0,1)))
    assert face<2 and abs(hit.z-5)<2e-6
    return dict(passed=True, max_camera_z_error=max(errors), tests=['tilted analytic plane', 'front object occlusion and triangle ID', 'camera-Z differs from ray length', 'near-plane clipping skips foreground and exposes background'])


def visible_collection_objects(collection):
    if collection.hide_render:
        return set()
    names={o.name for o in collection.objects}
    for child in collection.children:
        names.update(visible_collection_objects(child))
    return names


def geometry(scene, rec, camera_name):
    allowed=visible_collection_objects(scene.collection)
    blocked=set()
    def excluded_layers(layer):
        if layer.exclude:
            blocked.update(o.name for o in layer.collection.all_objects)
        else:
            for child in layer.children: excluded_layers(child)
    excluded_layers(bpy.context.view_layer.layer_collection)
    homecage=camera_name.startswith('HomecageCam_')
    selected_proxy=None
    if homecage and camera_name!='HomecageCam_Top':
        selected_proxy='MGF_DepthShellProxy_'+camera_name.split('HomecageCam_')[1]
        assert selected_proxy in bpy.data.objects
    ignores=set(rec['source_depth_policy'].get('depth_excluded_objects') or [])
    # Explicit policy for old Cylinder sources lacking a machine-readable list.
    glass={'Part_GlassCylinder160mm','Part_AntiReflectiveGlass','Part_AntiReflectiveGlass_Circle',
           'Part_GlassCylinder160mm.001','Part_AntiReflectiveGlass.001','Part_AntiReflectiveGlass_Circle.001'}
    prefixes=('CameraItems.','Part_CameraIO.','Part_CameraLens.','Part_CameraMount.','Part_CameraPCB.','Part_CamLen.')
    candidates=[];excluded=[]
    for obj in sorted(scene.objects, key=lambda o:o.name):
        if obj.type not in ('MESH','CURVE','CURVES','SURFACE','META','FONT','VOLUME'):continue
        reason=None
        if obj.type!='MESH':reason='non-mesh excluded (hair/curves/volumes); skin policy'
        elif obj.name in glass or obj.name in ignores:reason='explicit depth-ignore surface'
        elif obj.get('mgf_depth_proxy') and obj.name!=selected_proxy:reason='other-camera depth proxy'
        elif selected_proxy and obj.name=='MGF_MARS_HomecageShell':reason='replaced by per-camera shell proxy'
        elif obj.name==selected_proxy:pass
        elif obj.name not in allowed or obj.name in blocked:reason='collection hidden/excluded'
        elif obj.hide_render or not getattr(obj,'visible_camera',True):reason='object hidden for camera/render'
        elif not homecage and obj.name.startswith(prefixes):reason='physical camera geometry excluded'
        if reason:
            excluded.append(dict(object=obj.name, reason=reason));continue
        assert obj.type=='MESH'
        obj.hide_viewport=False
        obj.hide_set(False)
        for mod in obj.modifiers:
            mod.show_viewport=mod.show_render
            if mod.type=='SUBSURF':mod.levels=mod.render_levels
        candidates.append(obj)
    bpy.context.view_layer.update()
    dg=bpy.context.evaluated_depsgraph_get()
    vertices=[];faces=[];face_objects=[];face_instances=[];objects=[];offset=0
    for obj in candidates:
        v,f=mesh_arrays(obj,dg)
        if not len(f):continue
        oid=len(objects)+1
        iid=rec['meshes'].index(obj.name)+1 if obj.name in rec['meshes'] else 0
        vertices.append(v);faces.append(f+offset);offset+=len(v)
        face_objects.append(np.full(len(f),oid,np.int32));face_instances.append(np.full(len(f),iid,np.int16))
        objects.append(dict(object_id=oid, instance_id=iid, name=obj.name, vertices=len(v), triangles=len(f)))
    assert set(rec['meshes']).issubset({o['name'] for o in objects})
    verts=np.concatenate(vertices);tris=np.concatenate(faces)
    tree=BVHTree.FromPolygons(verts.tolist(),tris.tolist(),all_triangles=True)
    return tree,verts,tris,np.concatenate(face_objects),np.concatenate(face_instances),objects,excluded


def trace(tree, origins, directions, max_distance):
    count=len(directions)
    xyz=np.zeros((count,3),np.float32);normals=np.zeros_like(xyz)
    face=np.full(count,-1,np.int32)
    common=np.asarray(origins).ndim==1
    origin=Vector(origins) if common else None
    for j,d in enumerate(directions):
        if not common:origin=Vector(origins[j])
        point,normal,index,_=tree.ray_cast(origin,Vector(d),max_distance)
        if index is not None:
            xyz[j]=point;normals[j]=normal;face[j]=index
    return xyz,normals,face


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True)
    args=p.parse_args(sys.argv[sys.argv.index('--')+1:]);protocol=json.loads(args.protocol.read_text())
    root=Path(protocol['windows_root']);dest=root/protocol['output']/protocol.get('truth_subdirectory','surface_truth')
    dest.mkdir(exist_ok=False);write(dest/'analytic_fixture.json',fixture())
    results=[];opened=None
    for rec in protocol['records']:
        probe=json.loads((root/protocol['output']/'probe'/f'{rec["index"]:02d}.json').read_text())
        assert max(m['support_max'] for m in probe['meshes'])<=protocol['gates']['source_support_max_world_units']
        assert all(c['k_max_abs']<.001 and c['e_max_abs']<.001 for c in probe['cameras'])
        source=root/rec['source_blend']
        # Reopen every scene: visibility/modifier settings from earlier cameras
        # must not become the initial state of a later frame.
        bpy.ops.wm.open_mainfile(filepath=str(source))
        for obj in bpy.data.objects:
            if obj.type=='CURVES':obj.hide_viewport=True
        scene=bpy.context.scene;scene.frame_set(rec['frame']);bpy.context.view_layer.update()
        ann=dict(np.load(root/rec['scene']/'keypoints.npz'))
        w,h=rec['source_size_wh'];yy,xx=np.mgrid[:h,:w];uv=np.stack([xx+.5,yy+.5,np.ones_like(xx)],-1).reshape(-1,3)
        for view in rec['views']:
            started=time.time();camera=bpy.data.objects[str(ann['camera_names'][view])]
            k,e=camera_parameters(scene,camera,w,h)
            tree,vertices,triangles,object_by_face,instance_by_face,objects,excluded=geometry(scene,rec,camera.name)
            rays_camera=uv@np.linalg.inv(k).T;rays=rays_camera@e[:3,:3]
            camera_near=rays_camera*(float(camera.data.clip_start)/rays_camera[:,2,None])
            starts=(camera_near-e[:3,3])@e[:3,:3]
            rays/=np.linalg.norm(rays,axis=1,keepdims=True);center=-e[:3,:3].T@e[:3,3]
            world,normals,face=trace(tree,starts,rays,float(camera.data.clip_end)*4)
            depth=(world@e[:3,:3].T+e[:3,3])[:,2]
            valid=(face>=0)&(depth>=camera.data.clip_start)&(depth<camera.data.clip_end)
            depth[~valid]=0;world[~valid]=0;normals[~valid]=0;face[~valid]=-1
            oid=np.zeros(len(face),np.int32);iid=np.zeros(len(face),np.int16)
            oid[valid]=object_by_face[face[valid]];iid[valid]=instance_by_face[face[valid]]
            # Independent algebra closure and explicit barycentric containment.
            selected=np.flatnonzero(valid & (iid>0))[::17]
            pixels=uv[selected];back=(pixels@np.linalg.inv(k).T)*depth[selected,None]
            reconstructed=(back-e[:3,3])@e[:3,:3]
            mesh_tri=vertices[triangles[face[selected]]]
            edge0=mesh_tri[:,1]-mesh_tri[:,0];edge1=mesh_tri[:,2]-mesh_tri[:,0]
            n=np.cross(edge0,edge1);nn=np.linalg.norm(n,axis=1)
            delta=reconstructed-mesh_tri[:,0];plane=np.abs((delta*n).sum(1))/nn
            aa=(edge0*edge0).sum(1);bb=(edge0*edge1).sum(1);cc=(edge1*edge1).sum(1)
            dd=(delta*edge0).sum(1);ee=(delta*edge1).sum(1);den=aa*cc-bb*bb
            b1=(cc*dd-bb*ee)/den;b2=(aa*ee-bb*dd)/den
            bary=np.stack([1-b1-b2,b1,b2],1)
            closure=dict(samples=len(selected), world_backproject_max=float(np.linalg.norm(reconstructed-world[selected],axis=1).max()),
                         triangle_plane_p99=float(np.quantile(plane,.99)),triangle_plane_max=float(plane.max()),
                         barycentric_min=float(bary.min()))
            ki=k.copy();ki[:2,2]-=.5
            path=dest/f'{rec["index"]:02d}_{view:02d}.npz'
            np.savez_compressed(path,depth_camera_z=depth.reshape(h,w).astype(np.float32),world_xyz=world.reshape(h,w,3),
                geometric_normal_world=normals.reshape(h,w,3),object_id=oid.reshape(h,w),instance_id=iid.reshape(h,w),
                triangle_id=face.reshape(h,w),valid=valid.reshape(h,w),intrinsic_edge_coordinates=k,
                intrinsic_array_indices=ki,extrinsic_world_to_camera=e)
            # Cross-view closure: continuously project these source hits, cast
            # the target ray against its own scene policy, classify occlusion.
            cross=[]
            for other in rec['views']:
                if other==view:continue
                target=bpy.data.objects[str(ann['camera_names'][other])];tk,te=camera_parameters(scene,target,w,h)
                target_tree,_,_,toids,tiids,_,_=geometry(scene,rec,target.name)
                cam=reconstructed@te[:3,:3].T+te[:3,3];hom=cam@tk.T
                puv=hom[:,:2]/hom[:,2,None];inside=(cam[:,2]>0)&(puv[:,0]>=0)&(puv[:,0]<w)&(puv[:,1]>=0)&(puv[:,1]<h)
                points=reconstructed[inside];tc=-te[:3,:3].T@te[:3,3];dirs=points-tc
                lengths=np.linalg.norm(dirs,axis=1);dirs/=lengths[:,None]
                dir_camera=dirs@te[:3,:3].T
                starts=tc+dirs*(float(target.data.clip_start)/dir_camera[:,2,None])
                hits,_,tf=trace(target_tree,starts,dirs,float(target.data.clip_end)*4)
                hz=(hits@te[:3,:3].T+te[:3,3])[:,2];expected=cam[inside,2]
                tol=np.maximum(1e-3,expected*1e-5);found=tf>=0
                same=found&(np.abs(hz-expected)<=tol);occluded=found&(hz<expected-tol)
                wrong=np.zeros(len(tf),bool);wrong[found]=tiids[tf[found]]!=iid[selected][inside][found]
                cross.append(dict(target_view=other,source_samples=len(selected),in_frame=int(inside.sum()),
                    visible_same_surface=int(same.sum()),occluded=int(occluded.sum()),
                    missing_or_behind=int((~same&~occluded).sum()),same_surface_wrong_instance=int((same&wrong).sum()),
                    visible_residual_p99=float(np.quantile(np.abs(hz[same]-expected[same]),.99)) if same.any() else None))
            record=dict(index=rec['index'],phase=rec['phase'],kind=rec['kind'],view=view,camera=camera.name,
                source_blend_sha256=digest(source),truth_path=str(path.relative_to(root)),truth_sha256=digest(path),
                resolution_wh=[w,h],seconds=time.time()-started,foreground_pixels=int((iid>0).sum()),
                clip_start=float(camera.data.clip_start),clip_end=float(camera.data.clip_end),
                instances={str(j):int((iid==j).sum()) for j in range(1,len(rec['meshes'])+1)},
                closure=closure,cross_view=cross,objects=objects,excluded_objects=excluded,
                policy='Primary opaque triangle hit in full render-visible mesh scene; skin not hair; explicit glass exclusions; Homecage side uses existing selective shell proxy, top retains shell. No refraction, transparency, antialiasing, temporal information or model predictions.',
                pixel_convention='Native source resolution; sample edge-coordinates (column+.5,row+.5); K_array principal point is K_edge minus .5. No resize or subpixel jitter.')
            write(dest/f'{rec["index"]:02d}_{view:02d}.json',record);results.append(record)
            write(dest/'results.json',results)
            print('SURFACE_TRUTH',rec['index'],view,'foreground',record['foreground_pixels'],'closure',closure,'seconds',round(record['seconds'],2),flush=True)
    write(dest/'completion.json',dict(views=len(results),scenes=len(protocol['records']),blender=bpy.app.version_string,
        script_sha256=digest(Path(__file__)),protocol_sha256=digest(args.protocol),results_sha256=digest(dest/'results.json'),cpu_only=True))


if __name__=='__main__':main()
