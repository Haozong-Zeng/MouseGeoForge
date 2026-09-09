"""Validate actual converted RGB-D, stock instance loading and temporal windows."""
import json
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'vggt-omega'))
from training.orbbec_openfield_dataset import build_orbbec_labeled_dataset, OrbbecOpenfieldTemporalDataset
from training.finetune_blendedmvs import read_pfm


def main():
    torch.set_num_threads(2)
    root=BASE/'dataset/Orbbec-Openfield'
    result={'splits':{},'temporal':{},'passed':False}
    valid_points=0
    visible_labels=0
    depth_frames=0
    for split in ('train','val','test'):
        ds=build_orbbec_labeled_dataset(root,split=split,image_size=(128,128))
        for idx in range(len(ds)):
            batch=ds[idx]
            scene=ds.scenes[idx]
            assert batch['images'].shape==(1,3,128,128)
            assert batch['instance_keypoint_xy'].shape==(1,2,23,2)
            assert batch['actual_instance_count'].item()==1
            assert batch['box_valid'][0,0] and not batch['mask_valid'].any()
            assert not batch['camera_available'].any() and not batch['instance_world_valid'].any()
            assert batch['depth_available'].all()
            with np.load(scene.root/'keypoints.npz',allow_pickle=False) as kp:
                assert np.allclose(batch['boxes_cxcywh'][0,0].numpy(),kp['boxes_cxcywh'][0,0])
                assert int(batch['instance_keypoint_valid'].sum())==int(kp['in_frame'].sum())
                assert not kp['in_frame'][0,0,14:].any()  # No paws/elbows/knees/abdomen.
            fid=int(scene.root.name.removeprefix('frame_'))
            q=cv2.imread(str(scene.root.parent/'depth_encoded'/f'depth_{fid:012d}.png'),cv2.IMREAD_UNCHANGED)
            assert q.dtype==np.uint8 and q.shape==(720,720,3)
            assert np.array_equal(q[:,:,0],q[:,:,1]) and np.array_equal(q[:,:,0],q[:,:,2])
            q=q[:,:,0]
            expected=q.astype(np.float32)*2.3529
            valid=(q>0)&(q<255)
            expected[~valid]=0
            depth=read_pfm(scene.root/'rendered_depth_maps/00000000.pfm',flip_y=False)
            assert np.array_equal(depth,expected)
            vm=cv2.imread(str(scene.root/'depth_valid/00000000.png'),0)>0
            assert np.array_equal(vm,valid)
            valid_points+=int(batch['instance_keypoint_valid'].sum())
            visible_labels+=int(batch['instance_keypoint_visibility_valid'].sum())
            depth_frames+=1
        sample=next(iter(DataLoader(ds,batch_size=2,num_workers=0)))
        result['splits'][split]={'labeled_frames':len(ds),'batch_images':list(sample['images'].shape),
            'batch_points':list(sample['instance_keypoint_xy'].shape)}
        print(f'{split}: {len(ds)} labeled scenes passed stock dataloader and exact depth checks',flush=True)
    temporal=OrbbecOpenfieldTemporalDataset(root,split=None,window_size=4,frame_step=2,window_stride=1,image_size=(128,128))
    # Verify every enumerated window excludes original frame-ID gaps.
    for si,start in temporal.windows:
        ids=temporal.sessions[si][2][start:start+7]
        assert (np.diff(ids)==1).all()
    for si,session in enumerate(temporal.sessions):
        first=next(i for i,pair in enumerate(temporal.windows) if pair[0]==si)
        sample=temporal[first]
        assert sample['images'].shape==(4,1,3,128,128)
        assert not sample['timestamp_valid'].any() and not sample['metric_velocity_valid'].any()
        assert not sample['camera_available'].any()
        assert (sample['frame_delta']==2).all()
        # Find a window whose first frame has a 2D annotation, compare to compact labels.
        annidx=session[3]
        candidates=[i for i,(s,p) in enumerate(temporal.windows) if s==si and annidx[p]>=0]
        sample=temporal[candidates[0]]
        assert sample['pose_supervision_valid'][0,0]
        assert sample['box_valid'][0,0,0]
    # Empty TXT frame must exist in the full pool but not in labeled scenes.
    cw=next(s for s in temporal.sessions if '-CW-' in s[0]['session_id'])
    p=int(np.flatnonzero(cw[2]==21294)[0])
    assert cw[3][p]==-1
    assert not (cw[1]/'frame_000000021294').exists()
    # Exercise batching with separate time and view axes.
    batched=next(iter(DataLoader(temporal,batch_size=2,num_workers=0)))
    result['temporal']={'windows_checked':len(temporal),'batch_images':list(batched['images'].shape),
        'no_gap_crossing':True,'empty_annotation_excluded_from_pose':True,'unknown_timestamps_masked':True}
    result.update(passed=True,labeled_depth_frames_exactly_verified=depth_frames,
                  valid_2d_points=valid_points,known_visibility_points=visible_labels)
    out=BASE/'output/orbbec_openfield_audit/conversion_validation.json'
    out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
