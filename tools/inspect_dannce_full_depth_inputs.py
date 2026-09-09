"""Read-only audit before full-length DANNCE pseudo-label generation."""
from pathlib import Path
import json
import sys
import shutil
import numpy as np
from scipy.io import loadmat
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'BlenderScripts'))
from convert_dannce_mouse_to_blenderpymouse import load_session
from core23_temporal_format import write_json_atomic
reports=[]
for mouse in (1,2):
    session=BASE/f'dataset/DANNCE-MarkerlessMouse{mouse}-Core23-RealMultiview/sessions/markerless_mouse_{mouse}'
    meta=json.loads((session/'metadata.json').read_text())
    raw=load_session(session/'label3d_dannce.mat')
    stored_com=np.load(session/'sync_com.npz')['com_world_xyz_mm']
    com=np.asarray(loadmat(session/'label3d_dannce.mat',squeeze_me=True,struct_as_record=False)['com'].com3d)
    labels=np.asarray(raw['source_frames'],int)
    xyz=np.asarray(raw['source_xyz'])
    valid=np.isfinite(xyz).all(-1)&(np.linalg.norm(xyz,axis=-1)>1e-5)
    relative=xyz-com[labels,None,:]
    reports.append(dict(mouse=mouse,frames=meta['frame_count'],views=meta['views'],resolution=meta['resolution'],
        rgb_counts=[len(list((session/'rgb'/f'{v:08d}').glob('*.jpg'))) for v in range(6)],
        manual_frames=labels.tolist(),maximum_manual_gap=int(np.diff(labels).max()),
        com_shape=list(stored_com.shape),com_finite=bool(np.isfinite(com).all()),
        com_min=com.min(0).tolist(),com_max=com.max(0).tolist(),
        relative_landmark_bounds=[relative[valid].min(0).tolist(),relative[valid].max(0).tolist()]))
report=dict(sessions=reports,free_bytes=shutil.disk_usage(BASE).free,
    expected_depth_bytes=sum(r['frames']*r['views']*(1152*1024*4+24) for r in reports))
write_json_atomic(BASE/'output/dannce_full_pseudo_depth/input_audit.json',report)
print(json.dumps(report,indent=2))
