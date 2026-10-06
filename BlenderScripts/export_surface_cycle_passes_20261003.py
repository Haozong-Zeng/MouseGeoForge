"""Geometry-only scene depth/object-index passes, preserving far cylinder wall.

RGB and accepted .blend files are immutable. Depth excludes the near half of
the cylinder wall for side views; the glass floor is skipped only from below.
Mouse masks describe the biological skin mesh, not fur/implants/reflections.
"""
import bpy,sys,json,hashlib,datetime as dt,time
from pathlib import Path
import numpy as np
from mathutils import Vector
ROOT=Path(sys.argv[-1]);sys.path.insert(0,str(ROOT/'BlenderScripts'))
import convert_cylinder_renders_to_blenderpymouse_core23 as fmt
OUT=ROOT/'output/surface_curve_cycle_20261003/render_staging'
DEADLINE=dt.datetime.fromisoformat(json.loads((OUT.parent/'plan.json').read_text())['deadline_utc'].replace('Z','+00:00'))
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    manifest=json.loads((OUT/'manifest.json').read_text());assert manifest['status'].startswith('rgb_geometry_complete')
    report=dict(complete=False,policy='Side: remove only nearest camera-facing cylinder hemisphere, retain far hemisphere. Below floor: skip transparent glass floor. Above floor: retain floor. Opaque scene geometry retained. No reflected/refracted paths in geometry truth.',depth_type_raw='Native Blender5.1 camera-axis Z in numerical mm; verify independently before publication, do not assume radial ray distance',mask_definition='MS_Mouse biological skin only; excludes fur, implant, glass, all reflections',poses=[])
    for rec in manifest['poses']:
        source=ROOT/rec['blend'];assert sha(source)==rec['blend_sha256'];bpy.ops.wm.open_mainfile(filepath=str(source));s=bpy.context.scene
        folder=OUT/f"frame_{rec['frame_index']:06d}";passes=folder/'geometry_passes';passes.mkdir(exist_ok=True)
        assert not list(passes.iterdir()),'Refuse overwrite of existing geometry passes'
        pref=bpy.context.preferences.addons['cycles'].preferences;pref.compute_device_type='OPTIX';pref.get_devices()
        for d in pref.devices:d.use=d.type=='OPTIX'
        assert any(d.use for d in pref.devices);s.cycles.device='GPU';s.cycles.samples=1;s.cycles.use_denoising=False;s.cycles.use_adaptive_sampling=False
        s.render.resolution_x=1080;s.render.resolution_y=1200;s.render.resolution_percentage=100;s.render.use_compositing=False;s.render.use_sequencer=False
        s.render.image_settings.media_type='MULTI_LAYER_IMAGE';s.render.image_settings.file_format='OPEN_EXR_MULTILAYER';s.render.image_settings.color_depth='32';s.render.image_settings.exr_codec='ZIP';s.render.image_settings.use_exr_interleave=True;s.render.film_transparent=True
        mat=bpy.data.materials.new('SurfaceCycle_GeometryOnly');mat.use_nodes=True;nodes=mat.node_tree.nodes;nodes.clear();em=nodes.new('ShaderNodeEmission');em.inputs['Color'].default_value=(.1,.1,.1,1);op=nodes.new('ShaderNodeOutputMaterial');mat.node_tree.links.new(em.outputs[0],op.inputs['Surface'])
        vl=bpy.context.view_layer;vl.material_override=mat;vl.use_pass_z=True;vl.use_pass_object_index=True
        for ob in bpy.data.objects:
            ob.pass_index=1 if ob.name=='MS_Mouse' else 0
            if ob.type=='CURVES':ob.hide_render=True
        wall=bpy.data.objects['Part_GlassCylinder160mm'];floor=bpy.data.objects['Part_AntiReflectiveGlass']
        bpy.context.view_layer.update();ev=wall.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=ev.to_mesh()
        try:vertices=[ev.matrix_world@v.co for v in mesh.vertices];faces=[list(p.vertices) for p in mesh.polygons]
        finally:ev.to_mesh_clear()
        wall.hide_render=True;center=np.mean(np.array(vertices),axis=0)
        floor_z=np.mean([(floor.matrix_world@Vector(c)).z for c in floor.bound_box]);pr=dict(pose=rec['pose'],views=[]);report['poses'].append(pr)
        old_support=ROOT/'dataset/BlenderPyMouse/PyMouseStudio-BrownFur-ClimbWall-Core23/frame_000001/keypoints.npz'
        with np.load(old_support) as a:support_ids=a['support_vertex_ids'].copy()
        with np.load(folder/'geometry.npz') as a:xyz=a['vertices'].copy()
        assert support_ids.min()>=0 and support_ids.max()<len(xyz)
        np.savez_compressed(folder/'visibility_support.npz',support_vertex_ids=support_ids,support_world_xyz=xyz[support_ids])
        for view in rec['views']:
            assert dt.datetime.now(dt.timezone.utc)<DEADLINE-dt.timedelta(minutes=30)
            cam=bpy.data.objects['Replica_FR2_cam'+view['camera']];s.camera=cam;ratio=float(cam['replica_pixel_aspect_y_over_x']);s.render.pixel_aspect_x=max(1.,1/ratio);s.render.pixel_aspect_y=max(1.,ratio)
            camera=np.array(cam.matrix_world.translation);radial=camera[:2]-center[:2];side=np.linalg.norm(radial)>80
            keep=[]
            for face in faces:
                xy=np.mean(np.array([vertices[i] for i in face]),axis=0)[:2]
                if not side or np.dot(xy-center[:2],radial)<=0:keep.append(face)
            # Render-only proxy; no source file edits. Both surfaces of near glass removed.
            me=bpy.data.meshes.new('SurfaceCycle_RetainedWall');me.from_pydata(vertices,[],keep);me.update();proxy=bpy.data.objects.new('SurfaceCycle_RetainedWall',me);s.collection.objects.link(proxy)
            floor.hide_render=bool(camera[2]<floor_z)
            bpy.context.view_layer.update();K=fmt.camera_intrinsic(cam,1080,1200);E=fmt.camera_extrinsic(cam)
            assert np.max(np.abs(K-np.array(view['intrinsic'])))<1e-4 and np.max(np.abs(E-np.array(view['extrinsic'])))<1e-4
            p=passes/f"{view['view_id']:08d}.exr";s.render.filepath=str(p);t=time.monotonic();bpy.ops.render.render(write_still=True)
            pr['views'].append(dict(camera=view['camera'],file=p.relative_to(OUT).as_posix(),view_layer=vl.name,seconds=time.monotonic()-t,wall_faces_retained=len(keep),wall_faces_removed=len(faces)-len(keep),floor_ignored=floor.hide_render,camera_xyz=camera.tolist(),wall_center=center.tolist()))
            save(OUT/'geometry_pass_manifest.json',report);bpy.data.objects.remove(proxy,do_unlink=True);bpy.data.meshes.remove(me)
            print('GEOMETRY_PASS',rec['pose'],view['camera'],flush=True)
        assert sha(source)==rec['blend_sha256']
    report['complete']=True;save(OUT/'geometry_pass_manifest.json',report);print('GEOMETRY_PASSES_COMPLETE',flush=True)
if __name__=='__main__':main()
