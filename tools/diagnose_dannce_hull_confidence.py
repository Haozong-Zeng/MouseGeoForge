"""Compare silhouette vote thresholds on existing continuous pilot masks."""
import json
from pathlib import Path
import cv2
import numpy as np
from add_dannce_real_multiview_pseudo_depth import visual_hull,rasterize_depth
from validate_dannce_temporal_pseudo_depth import silhouette_confidence

BASE=Path(__file__).resolve().parents[1]
cv2.setNumThreads(1)
for mouse,fid,clip in [(1,72,'f000000-000199'),(2,833,'f000800-000999')]:
    s=BASE/f'dataset/DANNCE-MarkerlessMouse{mouse}-Core23-RealMultiview/sessions/markerless_mouse_{mouse}'
    meta=json.loads((s/'metadata.json').read_text())
    Ks=[np.asarray(c['intrinsic']) for c in meta['cameras']]
    Es=[np.asarray(c['extrinsic']) for c in meta['cameras']]
    masks=[cv2.imread(str(s/'pseudo_depth_pilot'/clip/'masks'/f'{v:08d}/{fid:06d}.png'),0) for v in range(6)]
    com=np.load(s/'sync_com.npz')['com_world_xyz_mm'][fid]
    bounds=np.stack([com-120,com+120])
    for n in (4,5,6):
        points,stats=visual_hull(bounds,masks,Ks,Es,2.,0.,n)
        cover=[]
        conf=[]
        med=[]
        for v in range(6):
            depth=rasterize_depth(points,masks[v],Ks[v],Es[v],2.)
            confidence=silhouette_confidence(depth,masks,Ks,Es,v)
            cover.append(round(float(np.count_nonzero(depth)/np.count_nonzero(masks[v])),4))
            conf.append(round(float(np.count_nonzero(confidence>=212)/np.count_nonzero(depth)),4))
            med.append(float(np.median(depth[depth>0])))
        print(json.dumps({'mouse':mouse,'frame':fid,'hull_votes':n,'voxels':len(points),'coverage':cover,'confidence5':conf,'median_depth':med}),flush=True)
