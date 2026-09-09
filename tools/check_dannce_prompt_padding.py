"""Visual prompt-box ablation on the sole Mouse 2 seed, no held-out labels."""
from pathlib import Path
import json
import sys
import cv2
import numpy as np
import torch
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'BlenderScripts'))
from convert_dannce_mouse_to_blenderpymouse import load_session
from add_dannce_real_multiview_pseudo_depth import project,prompts_for_view
from segment_dannce_4dgs_sam2 import overlay,clean_mask
from sam2.build_sam import build_sam2_video_predictor

cv2.setNumThreads(1)
torch.set_num_threads(4)
s=BASE/'dataset/DANNCE-MarkerlessMouse2-Core23-RealMultiview/sessions/markerless_mouse_2'
out=BASE/'output/dannce_temporal_depth_pilot_v1/mouse2/prompt_ablation'
out.mkdir(exist_ok=True,parents=True)
raw=load_session(s/'label3d_dannce.mat')
xyz=raw['source_xyz'][list(raw['source_frames']).index(833)]
meta=json.loads((s/'metadata.json').read_text())
p=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_s.yaml',str(BASE/'third_party/checkpoints/sam2.1_hiera_small.pt'),device='cuda')
rows=[]
for v in (0,3,4):
    c=meta['cameras'][v]
    K,E=np.asarray(c['intrinsic']),np.asarray(c['extrinsic'])
    uv,z=project(xyz,K,E)
    valid=np.isfinite(xyz).all(-1)&(np.linalg.norm(xyz,axis=-1)>1e-5)&(z>0)&(uv[:,0]>=0)&(uv[:,0]<1152)&(uv[:,1]>=0)&(uv[:,1]<1024)
    im=cv2.imread(str(s/'rgb'/f'{v:08d}/000833.jpg'))
    numeric=out/f'view{v}'
    numeric.mkdir(exist_ok=True)
    cv2.imwrite(str(numeric/'000000.jpg'),im)
    tiles=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        state=p.init_state(video_path=str(numeric),offload_video_to_cpu=True)
        for padding in (18.,64.,96.,128.):
            p.reset_state(state)
            box,pts=prompts_for_view(uv,valid,1152,1024,padding)
            _,ids,logits=p.add_new_points_or_box(inference_state=state,frame_idx=0,obj_id=1,
                points=pts,labels=np.ones(len(pts),np.int32),box=box)
            mask=clean_mask((logits[ids.index(1),0]>0).cpu().numpy())
            result=overlay(im,mask,box)
            # Fixed crop per view based on the widest prompt plus margin.
            center=(uv[valid].min(0)+uv[valid].max(0))/2
            x=int(np.clip(center[0]-240,0,1152-480))
            y=int(np.clip(center[1]-200,0,1024-400))
            tile=cv2.resize(result[y:y+400,x:x+480],(360,300))
            cv2.putText(tile,f'view {v} padding {int(padding)}',(8,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            tiles.append(tile)
        p.reset_state(state)
    rows.append(np.concatenate(tiles,axis=1))
cv2.imwrite(str(out/'comparison.jpg'),np.concatenate(rows,axis=0))
print(out/'comparison.jpg')
