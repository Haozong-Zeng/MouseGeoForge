"""Verify every continuous pilot PFM/mask and attach explicit sparse depth indices."""
from pathlib import Path
import json
import shutil
import sys
import cv2
import numpy as np
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'vggt-omega'))
from training.dannce_temporal_dataset import read_depth_pfm
from export_dannce_complete_temporal import pilot_annotations
from core23_temporal_format import write_json_atomic,write_jsonl_atomic

cv2.setNumThreads(1)
results=[]
for mouse in (1,2):
    session=BASE/f'dataset/DANNCE-MarkerlessMouse{mouse}-Core23-RealMultiview/sessions/markerless_mouse_{mouse}'
    annotations,meta_update=pilot_annotations(session)
    checked=0
    near,far=float('inf'),0.
    for fid,row in annotations.items():
        for v in range(6):
            depth=read_depth_pfm(session/row['depth'][v],row['depth_pfm_flip_y'])
            mask=cv2.imread(str(session/row['mask'][v]),0)
            confidence=cv2.imread(str(session/row['confidence'][v]),0)
            assert depth.dtype==np.float32 and depth.shape==(1024,1152)
            assert mask.shape==confidence.shape==depth.shape
            assert np.isfinite(depth).all() and (depth>=0).all()
            assert not (depth[mask==0]>0).any()
            assert not (confidence[depth==0]>0).any()
            assert (confidence>=212).any()
            near=min(near,float(depth[depth>0].min()))
            far=max(far,float(depth.max()))
            checked+=1
    rows=[json.loads(line) for line in (session/'frames.jsonl').read_text().splitlines()]
    for row in rows:
        row.update(annotations.get(row['source_frame_id'],{}))
    write_jsonl_atomic(session/'frames.jsonl',rows)
    meta=json.loads((session/'metadata.json').read_text())
    meta.update(meta_update)
    write_json_atomic(session/'metadata.json',meta)
    # Remove only this toolchain's disposable numeric-JPEG tracking copies.
    removed=0
    for pilot in meta_update['pseudo_depth_pilots']:
        cache=session/pilot['path']/'tracking_rgb'
        if not cache.exists():
            continue
        expected=(session/pilot['path']).resolve()
        if cache.resolve().parent!=expected or cache.name!='tracking_rgb':
            raise ValueError('Unexpected temporary cache location')
        files=[p for p in cache.rglob('*') if p.is_file()]
        if any(p.suffix!='.jpg' or not p.stem.isdigit() or p.is_symlink() for p in files):
            raise ValueError('Unexpected file in generated tracking cache; refusing cleanup')
        # Verify all canonical RGB copies exist before dropping temporary copies.
        assert all((session/p).is_file() for row in rows if row['source_frame_id'] in annotations for p in row['rgb'])
        removed+=len(files)
        shutil.rmtree(cache)
    result={'mouse':mouse,'pilot_times':len(annotations),'pfm_mask_confidence_triplets_verified':checked,
        'resolution':[1152,1024],'depth_unit':'mm','camera_z_range_mm':[near,far],
        'pfm_flip_y':True,'temporary_tracking_jpegs_removed':removed,'canonical_rgb_untouched':True}
    results.append(result)
    print(json.dumps(result),flush=True)
write_json_atomic(BASE/'output/temporal_dataset_audit/pilot_file_validation.json',results)
