"""Read-only comparison of the source rig/cameras with already rendered samples."""
import json
from pathlib import Path
import sys
import bpy
import numpy as np
ROOT=Path(r'\\wsl.localhost\Ubuntu\home\haozong\projects\MouseGeoForge')
sys.path.insert(0,str(ROOT/'BlenderScripts'))
import blender_water_reaching_rendering as renderer
from annotate_existing_blenderpymouse_keypoints import load_landmark_configuration,evaluated_landmarks
name=Path(bpy.data.filepath).stem
if name.endswith('_AniposeSyntheticReady'):
    name=name.removesuffix('_AniposeSyntheticReady')+'_PseudoSynthetic'
root=ROOT/'dataset/BlenderPyMouse'/name
meta=json.loads((root/'metadata.json').read_text())
s=bpy.context.scene
report={'blend':bpy.data.filepath,'scene_frames':[s.frame_start,s.frame_end],
    'scene_frame_step':s.frame_step,'fps':s.render.fps,'fps_base':s.render.fps_base,
    'resolution':[s.render.resolution_x,s.render.resolution_y],
    'engine':s.render.engine,'samples':s.cycles.samples,
    'unpacked_image_paths':[bpy.path.abspath(im.filepath) for im in bpy.data.images if im.source=='FILE' and not im.packed_file],
    'armatures':[o.name for o in bpy.data.objects if o.type=='ARMATURE'],
    'curves_count':sum(o.type=='CURVES' for o in bpy.data.objects),
    'mouse_collections':[c.name for c in bpy.data.objects['MS_Mouse'].users_collection],
    'action_frame_ranges':{a.name:list(a.frame_range) for a in bpy.data.actions},
    'keypoint_config':meta.get('keypoint_annotations',{}),'checks':[]}
s.render.resolution_x,s.render.resolution_y=meta['render_resolution']
s.render.resolution_percentage=100
folders=meta['scene_folders']
for folder in [folders[0],folders[len(folders)//2],folders[-1]]:
    fid=int(folder[6:]); s.frame_set(fid)
    npz=np.load(root/folder/'keypoints.npz')
    arm=bpy.data.objects['DeformationSystem']
    xyz=[]
    for bone in npz['names']:
        pb=arm.pose.bones[str(bone)]
        frac=.27 if str(bone)=='HeadEnd_M' else 0.
        xyz.append(list(arm.matrix_world@pb.head.lerp(pb.tail,frac)))
    expected=npz['joint_world_xyz'].reshape(-1,len(xyz),3)[0]
    errors=[]
    for v,camname in enumerate(meta['cameras']):
        ann=json.loads((root/folder/'annotations'/f'{v:08d}.json').read_text())
        K=renderer.camera_intrinsic_matrix(s,bpy.data.objects[camname])
        E=renderer.camera_extrinsic_matrix(bpy.data.objects[camname])
        errors.append({'view':v,'K_error':float(np.max(np.abs(K-ann['intrinsic']))),
            'E_error':float(np.max(np.abs(E-ann['extrinsic'])))})
    report['checks'].append({'frame':fid,'world_max_error':float(np.max(np.linalg.norm(np.array(xyz)-expected,axis=-1))),
        'cameras':errors})
out=ROOT/'output/temporal_completion'/f'{name}_source_inspection.json'
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(report,indent=2))
print('TEMPORAL_SOURCE_INSPECTION '+json.dumps(report),flush=True)
