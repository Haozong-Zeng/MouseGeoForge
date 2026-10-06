"""Show mask/box/pose disagreements and all objectness-positive candidates."""
from pathlib import Path
import cv2,numpy as np,torch,json
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'output/real_cylinder_bottom_frozen_v1'
protocol=json.loads((out/'protocol.json').read_text());tiles=[]
for session,frames in protocol['sources'].items():
    for f in frames:
        data=torch.load(out/f'predictions/{session}_{f:06d}_bottom.pt',weights_only=True,map_location='cpu')['outputs']
        rgb=cv2.imread(str(out/f'review/{session}_{f:06d}_A_input.png'));h,w=rgb.shape[:2];panels=[]
        for name,p in data.items():
            scores=p['instance_objectness_logits'][0,0].sigmoid().numpy().reshape(-1);q=int(scores.argmax())
            mask=cv2.resize(p['instance_mask_logits'][0,0,q].numpy(),(w,h))>0
            im=rgb.copy();im[mask]=(.65*im[mask]+.35*np.array([0,255,0])).astype('uint8')
            for qi in np.flatnonzero(scores>=.5):
                cx,cy,bw,bh=p['instance_boxes_cxcywh'][0,0,qi].numpy()*[w,h,w,h]
                color=(255,0,255) if qi==q else (255,255,0)
                cv2.rectangle(im,(round(cx-bw/2),round(cy-bh/2)),(round(cx+bw/2),round(cy+bh/2)),color,2)
                cv2.putText(im,f'q{qi} {scores[qi]:.3f}',(round(cx-bw/2),round(cy-bh/2)-4),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1)
            for k in list(range(7))+list(range(14,23)):
                x,y=np.rint(p['instance_keypoint_xy'][0,0,q,k].numpy()*[w,h]).astype(int);cv2.circle(im,(x,y),3,(0,255,255),-1)
            cv2.rectangle(im,(0,0),(w,40),(30,30,30),-1);cv2.putText(im,f'{session} {f} {name}',(8,28),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
            panels.append(cv2.resize(im,(352,384)))
        tile=np.concatenate(panels,1);tiles.append(tile)
        cv2.imwrite(str(out/f'review/{session}_{f:06d}_branch_audit.png'),tile)
for i in range(3):
    cv2.imwrite(str(out/f'review/bottom_branches_session_{i+1}.png'),np.concatenate(tiles[i*3:i*3+3],1))
