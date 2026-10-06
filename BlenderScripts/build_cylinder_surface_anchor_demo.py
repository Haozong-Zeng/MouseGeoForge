"""Versioned surface-landmark proposal, with fixed rest-topology anchors.

Run in each source Blender project. Never saves over the source or edits GT.
The point is a fixed barycentric combination of a rest triangle's vertices;
the larger connected vertex group is an anatomical illustration region only.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

SPECS = [(5, 'dorsal_thorax_surface', 'Spine3_M', 1, (1.0,.19,.035,1)),
         (6, 'dorsal_lumbar_surface', 'Spine1_M', 1, (.06,.7,.15,1)),
         (22, 'ventral_abdomen_surface', 'Spine3_M', -1, (.035,.48,1.0,1))]


def dump(path, data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')


def topology(mesh):
    return hashlib.sha256(json.dumps([list(p.vertices) for p in mesh.polygons]).encode()).hexdigest()


def material(name, color):
    m = bpy.data.materials.new(name)
    m.diffuse_color = color
    m.use_nodes = True
    bsdf = next(n for n in m.node_tree.nodes if n.type=='BSDF_PRINCIPLED')
    bsdf.inputs['Base Color'].default_value = color
    bsdf.inputs['Roughness'].default_value = .68
    return m


def anchors_from_rest(obj, arm):
    mesh = obj.data
    mesh.calc_loop_triangles()
    triangles = [list(t.vertices) for t in mesh.loop_triangles]
    vertices = [v.co.copy() for v in mesh.vertices]
    bvh = BVHTree.FromPolygons(vertices, triangles, all_triangles=True)
    adjacency = [set() for _ in vertices]
    for edge in mesh.edges:
        a,b = edge.vertices; adjacency[a].add(b); adjacency[b].add(a)
    # This verified mesh uses X=lateral, Y=dorsal, Z=longitudinal in rest space.
    ymax,ymin = max(v.y for v in vertices),min(v.y for v in vertices)
    result=[]
    for index,name,bone,sign,color in SPECS:
        bone_head = obj.matrix_world.inverted() @ arm.matrix_world @ arm.data.bones[bone].head_local
        origin = Vector((0,ymax+2 if sign>0 else ymin-2,bone_head.z))
        direction = Vector((0,-sign,0))
        hit,normal,triangle_id,distance = bvh.ray_cast(origin,direction)
        assert hit is not None
        ids = triangles[triangle_id]
        a,b,c = [np.array(vertices[i],float) for i in ids]
        uv = np.linalg.lstsq(np.stack([b-a,c-a],axis=1),np.array(hit)-a,rcond=None)[0]
        weights = np.array([1-uv.sum(),*uv]); weights[np.abs(weights)<1e-7]=0
        assert weights.min()>-1e-5
        weights=np.clip(weights,0,1);weights/=weights.sum()
        assert np.linalg.norm(weights@np.array([vertices[i] for i in ids])-np.array(hit))<1e-5
        region=set(ids)
        for _ in range(2):
            region.update(j for i in list(region) for j in adjacency[i])
        result.append(dict(core23_id=index,display_id=index+1,name=name,reference_bone=bone,
            direction='dorsal' if sign>0 else 'ventral',rest_xyz=list(hit),
            triangle_vertex_ids=ids,barycentric_weights=weights.tolist(),
            region_vertex_ids=sorted(region),color=list(color),
            coordinate_definition='fixed_rest_triangle_barycentric_surface_point',
            selection='rest local X=0; longitudinal Z at reference bone head; first ray hit from dorsal/ventral exterior'))
    return dict(schema='cylinder_surface_anchor_proposal_v1',topology_sha256=topology(mesh),
        rest_positions_sha256=hashlib.sha256(np.asarray(vertices,dtype='<f4').tobytes()).hexdigest(),
        vertex_count=len(vertices),anchors=result,training_allowed=False,
        legacy_names_are_not_semantically_equivalent=True)


def add_groups(obj, anchors):
    for a in anchors:
        group=obj.vertex_groups.new(name=f"KP{a['display_id']:02d}_{a['name']}_anchor")
        for i,w in zip(a['triangle_vertex_ids'],a['barycentric_weights']):
            if w>0: group.add([i],w,'REPLACE')
        region=obj.vertex_groups.new(name=f"KP{a['display_id']:02d}_{a['name']}_region")
        region.add(a['region_vertex_ids'],1,'REPLACE')


def points_from_vertices(vertices, anchors):
    return [np.asarray(a['barycentric_weights'])@vertices[a['triangle_vertex_ids']] for a in anchors]


def gpu(scene):
    scene.render.engine='CYCLES'
    pref=bpy.context.preferences.addons['cycles'].preferences
    pref.compute_device_type='OPTIX';pref.get_devices()
    devices=[]
    for dev in pref.devices:
        dev.use=dev.type=='OPTIX'
        if dev.use:devices.append(dev.name)
    if not devices:raise RuntimeError('No OptiX device; do not silently render on CPU')
    scene.cycles.device='GPU';scene.cycles.samples=24
    scene.cycles.use_denoising=True
    scene.render.resolution_x=1280;scene.render.resolution_y=720;scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG'
    scene.world=bpy.data.worlds.new(scene.name+'_World');scene.world.use_nodes=True
    background=next(n for n in scene.world.node_tree.nodes if n.type=='BACKGROUND')
    background.inputs[0].default_value=(.14,.17,.21,1)
    background.inputs[1].default_value=.55
    scene.view_settings.view_transform='AgX'
    return devices


def demo_scene(name, coordinates, faces, anchors, out, basis=None):
    scene=bpy.data.scenes.new(name)
    devices=gpu(scene)
    mesh=bpy.data.meshes.new(name+'_Skin')
    mesh.from_pydata(coordinates.tolist(),[],faces);mesh.update()
    obj=bpy.data.objects.new(name+'_Surface',mesh);scene.collection.objects.link(obj)
    add_groups(obj,anchors)
    obj['proposal_only']=True
    obj['surface_anchor_schema']='cylinder_surface_anchor_proposal_v1'
    obj.data.materials.append(material(name+'_gray',(.43,.46,.51,1)))
    for anchor in anchors:obj.data.materials.append(material(name+'_'+anchor['name'],tuple(anchor['color'])))
    for p in mesh.polygons:
        p.use_smooth=True
        for i,a in enumerate(anchors):
            if all(v in a['region_vertex_ids'] for v in p.vertices):p.material_index=i+1
    # Markers are true scene geometry: the skin can occlude them.
    points=points_from_vertices(coordinates,anchors)
    radius=float(np.ptp(coordinates[:,0]))*.011
    old_scene=bpy.context.window.scene;bpy.context.window.scene=scene
    for i,(a,point) in enumerate(zip(anchors,points)):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=16,ring_count=8,radius=radius,location=point)
        marker=bpy.context.object;marker.name=f"KP{a['display_id']:02d}_surface_anchor"
        marker.data.materials.append(obj.data.materials[i+1])
    scene['anchor_definition_json']=json.dumps(anchors)
    # Body crop excludes the long tail for readable anatomical definitions.
    # Coordinates are already world coordinates (Z up) in all scenes.
    center=(coordinates.min(0)+coordinates.max(0))/2
    span=float(np.linalg.norm(np.ptp(coordinates,axis=0)))
    for j,loc in enumerate([(span,-span,span),(-span,span,span),(0,0,-span)]):
        light=bpy.data.lights.new(name+f'_Area{j}','AREA');light.energy=span*span*18;light.shape='DISK';light.size=span*.9
        lo=bpy.data.objects.new(light.name,light);scene.collection.objects.link(lo);lo.location=Vector(center)+Vector(loc)
        lo.rotation_euler=(Vector(center)-lo.location).to_track_quat('-Z','Y').to_euler()
    cam_data=bpy.data.cameras.new(name+'_Camera');cam_data.type='ORTHO';cam_data.lens=50
    cam=bpy.data.objects.new(name+'_Camera',cam_data);scene.collection.objects.link(cam);scene.camera=cam
    records=[]
    from bpy_extras.object_utils import world_to_camera_view
    for role,direction in [('top',(0,0,1)),('bottom',(0,0,-1)),('side',(1,0,0))]:
        cam.location=Vector(center)+Vector(direction)*span*2
        cam.rotation_euler=(Vector(center)-cam.location).to_track_quat('-Z','Y').to_euler()
        # Blender's ortho_scale is the camera width at horizontal sensor fit.
        cam_data.sensor_fit='HORIZONTAL'
        bpy.context.view_layer.update()
        rotation=cam.rotation_euler.to_matrix().transposed()
        projected=np.array([rotation@(Vector(v)-Vector(center)) for v in coordinates])
        cam_data.ortho_scale=float(max(np.ptp(projected[:,0]),np.ptp(projected[:,1])*1280/720))*1.12
        scene.render.filepath=str(out/f'{name}_{role}_surface.png')
        bpy.ops.render.render(write_still=True,scene=scene.name)
        uv=[]
        for p in points:
            v=world_to_camera_view(scene,cam,Vector(p));uv.append([v.x*1280,(1-v.y)*720])
        records.append(dict(role=role,image=Path(scene.render.filepath).name,uv=uv,
                            camera_type='ORTHO',purpose='anatomical illustration only, not training camera'))
    bpy.context.window.scene=old_scene
    return scene,dict(scene=name,views=records,gpu=devices,world_points=[p.tolist() for p in points])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--action',required=True)
    parser.add_argument('--render-rest',action='store_true')
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    root=Path(args.root);out=root/'output/surface_anchor_demo_v1';out.mkdir(parents=True,exist_ok=True)
    action=args.action;source=bpy.data.filepath
    mesh=bpy.data.objects['MS_Mouse'];arm=bpy.data.objects['DeformationSystem']
    for o in bpy.data.objects:
        if o.type=='CURVES':o.hide_viewport=True;o.hide_render=True
    bpy.context.view_layer.update()
    anchor_path=out/'rest_surface_anchors.json'
    if action=='WalkAround':
        proposal=anchors_from_rest(mesh,arm);dump(anchor_path,proposal)
    else:
        proposal=json.loads(anchor_path.read_text())
        assert topology(mesh.data)==proposal['topology_sha256']
        positions=np.array([v.co for v in mesh.data.vertices],dtype='<f4')
        assert hashlib.sha256(positions.tobytes()).hexdigest()==proposal['rest_positions_sha256']
    anchors=proposal['anchors'];add_groups(mesh,anchors)
    faces=[list(p.vertices) for p in mesh.data.polygons]
    rest_world=np.array([mesh.matrix_world@v.co for v in mesh.data.vertices])
    src_gt=json.loads((root/f'dataset/BlenderPyMouse/PyMouseStudio-BrownFur-{action}-Core23/frame_000031/keypoints.json').read_text())
    frames=[];pose31=None;max_legacy_error=0
    for frame in range(1,92):
        bpy.context.scene.frame_set(frame);bpy.context.view_layer.update()
        dep=bpy.context.evaluated_depsgraph_get();evaluated=mesh.evaluated_get(dep);em=evaluated.to_mesh()
        assert len(em.vertices)==proposal['vertex_count']
        coords=np.array([evaluated.matrix_world@v.co for v in em.vertices]);evaluated.to_mesh_clear()
        points=points_from_vertices(coords,anchors)
        frames.append(dict(frame=frame,points=[p.tolist() for p in points],
                            dorsal_ventral_distance=float(np.linalg.norm(points[0]-points[2]))))
        if frame==31:
            pose31=coords.copy()
            for i,bone in enumerate(src_gt['bones']):
                p=np.array(arm.matrix_world@arm.pose.bones[bone].head)
                max_legacy_error=max(max_legacy_error,float(np.linalg.norm(p-np.asarray(src_gt['joint_world_xyz'][i]))))
    assert max_legacy_error<1e-3,(action,max_legacy_error)
    # True body-surface line-of-sight, independent of old joint-support visibility.
    bvh=BVHTree.FromPolygons([Vector(v) for v in pose31],faces,all_triangles=False)
    points=np.array(frames[30]['points']);views=[]
    for view in src_gt['views']:
        # Calibration is read from the scene metadata that generated the RGB.
        meta=json.loads((root/f'dataset/BlenderPyMouse/PyMouseStudio-BrownFur-{action}-Core23/frame_000031/scene_meta.json').read_text())
        cam=next(v for v in meta['views'] if v['view_id']==view['view_id'])
        E=np.array(cam['extrinsic']);K=np.array(cam['intrinsic'])
        camera=(E[:3,:3]@points.T).T+E[:3,3];uv=(K@camera.T).T;uv=uv[:,:2]/uv[:,2:3]
        origin=-E[:3,:3].T@E[:3,3];visibility=[]
        for point in points:
            ray=Vector(point-origin);distance=ray.length
            hit,normal,index,hit_distance=bvh.ray_cast(Vector(origin),ray.normalized(),distance+.01)
            visibility.append(hit is not None and abs(hit_distance-distance)<.08)
        views.append(dict(view_id=view['view_id'],uv=uv.tolist(),body_mesh_line_of_sight=visibility,
                          visibility_scope='bare mouse skin only; not fur or scene occlusion'))
    result=dict(action=action,source_blend=source,schema=proposal['schema'],anchors=anchors,
        max_legacy_world_error=max_legacy_error,frames=frames,frame31_views=views,
        training_allowed=False,pose_mesh_unchanged=True)
    dump(out/f'{action}_surface_tracks.json',result)
    # Lightweight, self-contained rest + evaluated pose review project, not the production rig.
    scenes=[];renders=[]
    if args.render_rest:
        scene,record=demo_scene('Rest',rest_world,faces,anchors,out);scenes.append(scene);renders.append(record)
    scene,record=demo_scene(action,pose31,faces,anchors,out);scenes.append(scene);renders.append(record)
    bpy.context.window.scene=scenes[0]
    bpy.data.libraries.write(str(out/f'{action}_SurfaceAnchorReview.blend'),set(scenes),fake_user=True,compress=True)
    dump(out/f'{action}_render_manifest.json',renders)
    print('SURFACE_ANCHOR_DEMO_COMPLETE',action,max_legacy_error,flush=True)


if __name__=='__main__':main()
