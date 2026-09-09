"""Real-pilot parity checks for the accelerated hull and temporal loader."""
import json
import sys
from pathlib import Path
import cv2
import numpy as np
from generate_dannce_full_pseudo_depth import CachedHull,BASE,session_path
from add_dannce_real_multiview_pseudo_depth import visual_hull
from core23_temporal_format import write_json_atomic

cv2.setNumThreads(1)
session=session_path(1)
meta=json.loads((session/'metadata.json').read_text())
stats=json.loads((BASE/'output/dannce_temporal_depth_pilot_v1/mouse1/per_frame_metrics.json').read_text())
bounds=np.array(stats[0]['bounds_mm'])
Ks=[np.asarray(c['intrinsic']) for c in meta['cameras']]
Es=[np.asarray(c['extrinsic']) for c in meta['cameras']]
masks=[cv2.imread(str(session/f'pseudo_depth_pilot/f000000-000199/masks/{v:08d}/000028.png'),0) for v in range(6)]
expected,_=visual_hull(bounds,masks,Ks,Es,2.,0.,6)
cache=CachedHull(bounds,Ks,Es)
actual=cache(masks)
assert np.array_equal(actual,expected),'CUDA hull voting differs from validated NumPy reference'
sys.path.insert(0,str(BASE/'vggt-omega'))
from training.dannce_temporal_dataset import DANNCECompleteTemporalDataset
ds=DANNCECompleteTemporalDataset(BASE/'dataset',split='train',window_size=2,
    image_size=(288,256),view_ids=(0,2),load_pseudo_depth=True)
sample=ds[0]
assert sample['images'].shape==(2,2,3,256,288)
assert sample['depth_available'].all() and sample['mask_available'].all()
assert sample['body_masks'].shape==(2,2,256,288)
assert sample['body_masks'].any() and sample['depth_masks'].any()
assert not sample['pose_supervision_valid'].any(),'Unannotated frames must not acquire fake pose GT'
disabled=DANNCECompleteTemporalDataset(BASE/'dataset',split='train',window_size=2,
    image_size=(288,256),view_ids=(0,),load_pseudo_depth=False)[0]
assert not disabled['depth_available'].any() and not disabled['mask_available'].any()
report={'passed':True,'hull_exact_numpy_parity':True,'occupied_voxels':len(actual),
    'loader_images_shape':list(sample['images'].shape),'loader_masks_shape':list(sample['body_masks'].shape),
    'unannotated_pose_stays_invalid':True,'pseudo_labels_opt_in':True}
write_json_atomic(BASE/'output/dannce_full_pseudo_depth/parity_validation.json',report)
print(json.dumps(report),flush=True)
