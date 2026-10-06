"""Read-only source animation breadth audit; no imported or edited animation."""
import bpy,json,sys,hashlib
import numpy as np
from pathlib import Path
root=Path(sys.argv[sys.argv.index('--')+1]);out=root/'output/pymousestudio_realism_cycle_20261002/source_pose_range.json';assert not out.exists();source=Path(bpy.data.filepath);before=hashlib.sha256(source.read_bytes()).hexdigest();s=bpy.context.scene;arm=bpy.data.objects['DeformationSystem'];rows=[]
names=['Root_M','Spine1_M','Spine2_M','Spine3_M','Spine4_M','Chest_M','Neck_M','Head1_M','HeadEnd_M','Tail0_M','Wrist_L','Wrist_R','Ankle_L','Ankle_R','Fingers1_L','Fingers1_R','Toes1_L','Toes1_R'];names=[n for n in names if n in arm.pose.bones]
def p(n):return np.array(arm.matrix_world@arm.pose.bones[n].head)
def paw(side,hind=False):return np.mean([p(f'{d}{"Toe" if hind else "Finger"}1_{side}') for d in ['Index','Middle','Ring','Pinky']],axis=0)
for frame in sorted(set(list(range(1,251,5))+[31,250])):
    s.frame_set(frame);bpy.context.view_layer.update();fore=np.mean([paw('L'),paw('R')],0);hind=np.mean([paw('L',True),paw('R',True)],0);nose=p('HeadEnd_M');axis=nose-hind
    rows.append(dict(frame=frame,bones={n:p(n).tolist() for n in names},paws={n:paw(side,h).tolist() for n,side,h in [('foreL','L',False),('foreR','R',False),('hindL','L',True),('hindR','R',True)]},fore_mean=fore.tolist(),hind_mean=hind.tolist(),nose_minus_hind_mm=axis.tolist(),nose_axis_elevation_degrees=float(np.rad2deg(np.arctan2(axis[2],np.linalg.norm(axis[:2])))),fore_minus_hind_height_mm=float(fore[2]-hind[2])))
s.frame_set(1);bpy.context.view_layer.update();rest={b.name:dict(parent=b.parent.name if b.parent else None,head=(arm.matrix_world@b.head_local).to_tuple(),tail=(arm.matrix_world@b.tail_local).to_tuple(),use_connect=b.use_connect) for b in arm.data.bones}
result=dict(source_blend=str(source),source_sha256=before,source_unchanged=hashlib.sha256(source.read_bytes()).hexdigest()==before,action=arm.animation_data.action.name if arm.animation_data and arm.animation_data.action else None,frame_range=[s.frame_start,s.frame_end],sampled_frames=rows,rest_bones=rest,limitations=['Source motion is only a geometric initialization candidate, not a claim to reconstruct the real video.','Future fit must maintain floor/wall contacts and validate allviews. Diagnostic real3D not groundtruth.','No source scene saved and no other project animation copied.'])
out.write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(dict(frames=len(rows),elevation_range=[min(r['nose_axis_elevation_degrees'] for r in rows),max(r['nose_axis_elevation_degrees'] for r in rows)],source_unchanged=result['source_unchanged']),indent=2))
