"""Contact sheets for visual inspection, no production mask modifications."""
from pathlib import Path
import cv2,numpy as np

root=Path(__file__).resolve().parents[1]
review=root/'dataset/PyMouseStudio/sessions/AZL3/reconstruction_unified_guarded_v1/review'
tiles=[]
for frame in range(780,812):
    im=cv2.imread(str(review/f'camF_{frame:06d}.png'))
    tile=np.concatenate([im[:,370:740],im[:,1110:1480]],axis=1)
    tile=cv2.resize(tile,(296,180));cv2.putText(tile,str(frame),(8,175),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,255),2)
    tiles.append(tile)
for page in range(2):
    group=tiles[page*16:page*16+16]
    cv2.imwrite(str(review/f'camF_contact_{page+1}.png'),np.concatenate([np.concatenate(group[i:i+4],1) for i in range(0,16,4)],0))
