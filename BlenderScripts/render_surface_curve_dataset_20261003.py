"""Render accepted static poses unchanged; stage RGB, calibration and mesh surface GT.

Promotion to dataset/BlenderPyMouse requires depth/mask export and loader QA.
Never saves changes over accepted projects. Curves are mesh material midlines,
not internal skeletal spines and not per-view silhouettes.
"""
import bpy, sys, json, hashlib, time, datetime as dt
from pathlib import Path
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from bpy_extras.object_utils import world_to_camera_view

ROOT=Path(sys.argv[-1]);sys.path.insert(0,str(ROOT/'BlenderScripts'))
import convert_cylinder_renders_to_blenderpymouse_core23 as fmt
BASE=ROOT/'BlenderPyMouseProjects/Cylinder/PyMouseStudio-RealCapture-Final'
OUT=ROOT/'output/surface_curve_cycle_20261003/render_staging'
PLAN=json.loads((ROOT/'output/surface_curve_cycle_20261003/plan.json').read_text())
DEADLINE=dt.datetime.fromisoformat(PLAN['deadline_utc'].replace('Z','+00:00'))

def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def guard():assert dt.datetime.now(dt.timezone.utc)<DEADLINE-dt.timedelta(minutes=30),'Reserved closure window'

def geometry(obj,arm):
    proposal=json.loads((ROOT/'output/surface_anchor_demo_v1/rest_surface_anchors.json').read_text())
    topo=hashlib.sha256(json.dumps([list(p.vertices) for p in obj.data.polygons]).encode()).hexdigest()
    assert topo==proposal['topology_sha256']
    # Use original rest material coordinates, avoiding view-dependent silhouettes.
    rest=np.array([v.co for v in obj.data.vertices]);obj.data.calc_loop_triangles()
    triangles=np.array([list(t.vertices) for t in obj.data.loop_triangles],np.int32)
    tree=BVHTree.FromPolygons([Vector(v) for v in rest],triangles.tolist(),all_triangles=True)
    heads={b:np.array(obj.matrix_world.inverted()@arm.matrix_world@arm.data.bones[b].head_local) for b in ['Neck3_M','Tail0_M']}
    z0,z1=heads['Neck3_M'][2],heads['Tail0_M'][2]
    assert abs(z0-z1)>1e-3
    ids=[];weights=[];hits=[]
    for z in np.linspace(z0,z1,65):
        p,n,idx,d=tree.ray_cast(Vector((0,float(rest[:,1].max()+10),float(z))),Vector((0,-1,0)))
        assert p is not None,'Missing dorsal surface hit'
        tri=triangles[idx];a,b,c=rest[tri]
        uv=np.linalg.lstsq(np.stack([b-a,c-a],axis=1),np.array(p)-a,rcond=None)[0]
        w=np.array([1-uv.sum(),*uv]);assert w.min()>-1e-5
        w=np.clip(w,0,1);w/=w.sum();ids.append(tri);weights.append(w);hits.append(list(p))
    ev=obj.evaluated_get(bpy.context.evaluated_depsgraph_get());me=ev.to_mesh()
    try:
        assert len(me.vertices)==len(rest)
        xyz=np.array([ev.matrix_world@v.co for v in me.vertices]);me.calc_loop_triangles();faces=np.array([list(t.vertices) for t in me.loop_triangles],np.int32)
    finally:ev.to_mesh_clear()
    ids=np.array(ids);weights=np.array(weights);curve=np.einsum('si,sij->sj',weights,xyz[ids])
    names=[s['name'] for s in fmt.CORE23_SPECS]
    joints=np.array([arm.matrix_world@arm.pose.bones[s['bone']].head for s in fmt.CORE23_SPECS])
    for a in proposal['anchors']:
        joints[a['core23_id']]=np.array(a['barycentric_weights'])@xyz[a['triangle_vertex_ids']]
        names[a['core23_id']]=a['name']
    return dict(vertices=xyz,triangles=faces,rest_vertices=rest,curve_world_xyz=curve,curve_triangle_ids=ids,
                curve_barycentric_weights=weights,curve_rest_xyz=np.array(hits),joint_world_xyz=joints,names=np.array(names),bones=np.array([s['bone'] for s in fmt.CORE23_SPECS]))

def main():
    guard();OUT.mkdir(exist_ok=False)
    records=json.loads((BASE/'final_manifest.json').read_text())['records'];rows=[]
    result=dict(status='rendering_rgb_geometry_not_dataset_ready',resolution_wh=[1080,1200],poses=[],
                temporal_reliable=False,curve_definition='65 fixed rest-X=0 dorsal mesh material samples from neck level to tail-root level; barycentric deformation; not internal spine',
                depth_mask_status='pending_export_and_validation',production_dataset_published=False)
    save(OUT/'manifest.json',result)
    for fi,r in enumerate(records,1):
        guard();path=ROOT/r['final'];assert sha(path)==r['sha256'];bpy.ops.wm.open_mainfile(filepath=str(path));s=bpy.context.scene
        assert s.frame_current==r['frame'];bpy.context.view_layer.update()
        prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='OPTIX';prefs.get_devices()
        devices=[]
        for d in prefs.devices:
            d.use=d.type=='OPTIX'
            if d.use:devices.append(d.name)
        assert devices;s.render.engine='CYCLES';s.cycles.device='GPU';s.cycles.samples=128
        s.render.resolution_x=1080;s.render.resolution_y=1200;s.render.resolution_percentage=100
        s.render.use_persistent_data=True;s.render.image_settings.file_format='PNG';s.render.image_settings.color_mode='RGB';s.render.image_settings.color_depth='8'
        folder=OUT/f'frame_{fi:06d}';folder.mkdir();geo=geometry(bpy.data.objects['MS_Mouse'],bpy.data.objects['DeformationSystem'])
        np.savez_compressed(folder/'geometry.npz',**geo)
        pose=dict(pose=r['pose'],source_frame=r['frame'],frame_index=fi,blend=r['final'],blend_sha256=r['sha256'],devices=devices,views=[])
        result['poses'].append(pose)
        for vi,c in enumerate('ABCDEFG'):
            guard();cam=bpy.data.objects['Replica_FR2_cam'+c];assert cam.data.type=='PERSP';s.camera=cam
            aspect=float(cam['replica_pixel_aspect_y_over_x']);s.render.pixel_aspect_x=max(1.,1/aspect);s.render.pixel_aspect_y=max(1.,aspect);bpy.context.view_layer.update()
            K=fmt.camera_intrinsic(cam,1080,1200);E=fmt.camera_extrinsic(cam)
            cp,uv=fmt.project_points(geo['joint_world_xyz'],E,K);cc,cu=fmt.project_points(geo['curve_world_xyz'],E,K)
            builtin=np.array([[p.x*1080,(1-p.y)*1200] for p in [world_to_camera_view(s,cam,Vector(x)) for x in geo['curve_world_xyz']]])
            error=float(np.abs(builtin-cu).max());assert error<.01,(c,error)
            out=folder/'rgb'/f'{vi:08d}.png';out.parent.mkdir(exist_ok=True);s.render.filepath=str(out);t=time.monotonic();bpy.ops.render.render(write_still=True)
            pose['views'].append(dict(camera=c,view_id=vi,rgb=out.relative_to(OUT).as_posix(),intrinsic=K.tolist(),extrinsic=E.tolist(),pixel_aspect_y_over_x=aspect,projection_max_error_px=error,keypoint_uv=uv.tolist(),keypoint_camera_xyz=cp.tolist(),curve_uv=cu.tolist(),curve_camera_xyz=cc.tolist(),render_seconds=time.monotonic()-t))
            save(folder/'render_meta.json',pose);save(OUT/'manifest.json',result)
            print('RENDERED',r['pose'],c,flush=True)
        assert sha(path)==r['sha256']
    result['status']='rgb_geometry_complete_pending_depth_masks_loader_and_visual_QA';save(OUT/'manifest.json',result)
    print('RGB_GEOMETRY_COMPLETE',flush=True)

if __name__=='__main__':main()
