"""Verify split datasets, temporal boundaries, full DANNCE storage and real loader reads."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from PIL import Image

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'vggt-omega'))
from training.temporal_window_index import TemporalWindowIndex
from training.dannce_temporal_dataset import DANNCECompleteTemporalDataset
from core23_temporal_format import write_json_atomic,sha256


def verify_jpeg(path):
    stat=path.stat()
    if path.is_symlink() or stat.st_nlink!=1:
        raise ValueError(f'Non-independent file: {path}')
    with Image.open(path) as image:
        if image.size!=(1152,1024) or image.format!='JPEG':
            raise ValueError(f'Malformed JPEG header: {path}')
        image.verify()
    with path.open('rb') as f:
        f.seek(-2,2)
        if f.read()!=b'\xff\xd9':
            raise ValueError(f'Truncated JPEG: {path}')
    return stat.st_size


def main():
    cv2.setNumThreads(1)
    dataset=BASE/'dataset'
    index=TemporalWindowIndex(dataset,window_size=4,frame_step=2,window_stride=1)
    counts={}
    for si,_ in index.windows:
        key=index.sequences[si][0]['dataset_id']
        counts[key]=counts.get(key,0)+1
    report={'window_size':4,'frame_step':2,'window_count':len(index),
        'indexed_sequences':len(index.sequences),'windows_by_dataset':counts,'dannce':[],'blender':[]}
    for rec,path,ids,rows,positions in index.sequences:
        if not rec['path'].startswith('BlenderPyMouse/'):
            continue
        for row in rows:
            if not (path/row['scene']/'keypoints.npz').exists():
                raise FileNotFoundError(path/row['scene'])
        report['blender'].append({'dataset':rec['dataset_id'],'frames':len(ids),
            'source_frame_step':rec['source_frame_step'],'motion_source':rec['motion_source_id'],
            'split':rec['recommended_split'],'frame_range':[int(ids[0]),int(ids[-1])]})
    for mouse in (1,2):
        root=dataset/f'DANNCE-MarkerlessMouse{mouse}-Core23-RealMultiview'
        session=root/f'sessions/markerless_mouse_{mouse}'
        meta=json.loads((session/'metadata.json').read_text())
        assert meta['export_complete'] and meta['frame_count']==18000
        assert sha256(session/'label3d_dannce.mat')==sha256(dataset/f'DANNCE/raw/markerless_mouse_{mouse}/label3d_dannce.mat')
        rows=[json.loads(line) for line in (session/'frames.jsonl').read_text().splitlines()]
        assert [r['source_frame_id'] for r in rows]==list(range(18000))
        assert all(r['transition_valid'] for r in rows[1:]) and not rows[0]['transition_valid']
        images=[session/p for r in rows for p in r['rgb']]
        with ThreadPoolExecutor(max_workers=12) as pool:
            size=sum(pool.map(verify_jpeg,images))
        manual=0
        max_projection_error=0.
        pixel_difference=[]
        for row in rows:
            if row['manual_keypoints'] is None:
                assert not row['keypoint_label_valid']
                continue
            manual+=1
            npz=session/row['manual_keypoints']
            with np.load(npz,allow_pickle=False) as ann:
                assert int(ann['source_video_frame_zero_based'])==row['source_frame_id']
                assert int(ann['source_sample_id'])==row['source_sample_id']
                valid=ann['world_valid'][0]
                xyz=ann['joint_world_xyz'][0,valid].astype(float)
                for v,c in enumerate(meta['cameras']):
                    K,E=np.asarray(c['intrinsic']),np.asarray(c['extrinsic'])
                    camera=xyz@E[:3,:3].T+E[:3,3]
                    uvh=camera@K.T
                    error=np.linalg.norm(uvh[:,:2]/uvh[:,2:]-ann['uv'][v,0,valid],axis=-1)
                    max_projection_error=max(max_projection_error,float(error.max()))
            # Decode every manual time in all views; compare to the earlier sparse conversion.
            for v in range(6):
                full=cv2.imread(str(session/row['rgb'][v]))
                old=cv2.imread(str(npz.parent/'blended_images'/f'{v:08d}.jpg'))
                assert full is not None and old is not None and full.shape==old.shape
                pixel_difference.append(float(np.abs(full.astype(np.int16)-old.astype(np.int16)).mean()))
        # Decode both sides of every MP4 chunk boundary, including the final frame.
        assert max_projection_error<0.01,max_projection_error
        assert float(np.mean(pixel_difference))<0.2,pixel_difference
        for fid in [0,17999,*[x+d for x in range(3000,18000,3000) for d in (-1,0,1)]]:
            for v in range(6):
                assert cv2.imread(str(session/rows[fid]['rgb'][v])).shape==(1024,1152,3)
        split='train' if mouse==1 else 'test'
        loader=DANNCECompleteTemporalDataset(dataset,split=split,window_size=3,
            window_stride=1,image_size=(288,256),view_ids=(0,3),max_instances=2,
            load_pilot_pseudo_depth=True)
        first=26 if mouse==1 else 892
        sample=loader[first]
        assert sample['images'].shape==(3,2,3,256,288)
        assert sample['frame_ids'].tolist()==[first,first+1,first+2]
        assert sample['pose_supervision_valid'][:,0].tolist()==[False,True,False]
        with np.load(session/rows[first+1]['manual_keypoints'],allow_pickle=False) as expected:
            assert sample['instance_world_valid'][1,0].sum()==expected['world_valid'][0].sum()
        assert 0<int(sample['instance_world_valid'][1,0].sum())<=16
        assert not sample['instance_world_valid'][[0,2]].any()
        assert not sample['instance_valid'][:,:,1].any()
        assert sample['depth_available'].all() and sample['depth_masks'].any()
        assert (sample['depths'][sample['depth_masks']]>0).all()
        np.testing.assert_allclose(sample['time_seconds'].numpy(),np.arange(first,first+3)/100.)
        # Outside pilot: unavailable, not false background depth supervision.
        other=loader[15000]
        assert not other['depth_available'].any() and not other['depth_masks'].any()
        result={'mouse':mouse,'synchronized_times':18000,'view_count':6,
            'independent_jpeg_count':len(images),'independent_jpeg_bytes':size,
            'all_jpeg_headers_and_eof_verified':True,'manual_label_frames':manual,
            'max_manual_reprojection_error_px':max_projection_error,
            'sparse_vs_full_rgb_mean_absolute_difference':float(np.mean(pixel_difference)),
            'decoded_chunk_boundary_frames':True,'temporal_loader_shapes_and_validity_passed':True,
            'pilot_loader_depth_mm_range':[float(sample['depths'][sample['depth_masks']].min()),
                                           float(sample['depths'][sample['depth_masks']].max())]}
        report['dannce'].append(result)
        print(json.dumps(result),flush=True)
    write_json_atomic(BASE/'output/temporal_dataset_audit/delivery_validation.json',report)
    print('Temporal delivery audit passed',flush=True)


if __name__=='__main__':
    main()
