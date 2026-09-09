"""DANNCE complete rectified multiview RGB windows with sparse manual supervision.

Time [T] and views [V] stay separate. Optional pilot pseudo-depth is opt-in;
unlabeled RGB frames are never negative instances or fabricated zero poses.
"""
from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from training.temporal_window_index import TemporalWindowIndex


def read_depth_pfm(path,flip_y):
    with Path(path).open('rb') as f:
        if f.readline().strip()!=b'Pf':
            raise ValueError(f'Expected single-channel PFM: {path}')
        width,height=map(int,f.readline().split())
        scale=float(f.readline())
        depth=np.frombuffer(f.read(),dtype='<f4' if scale<0 else '>f4').reshape(height,width)
    return (np.flipud(depth) if flip_y else depth).copy()


class DANNCECompleteTemporalDataset(Dataset):
    def __init__(self,dataset_root,*,split='train',window_size=16,frame_step=1,
                 window_stride=16,image_size=(576,512),view_ids=(0,1,2,3,4,5),
                 max_instances=2,load_pilot_pseudo_depth=False,confidence_min_votes=5,
                 load_pseudo_depth=None):
        self.root=Path(dataset_root)
        if min(image_size)<=0 or any(x%16 for x in image_size) or max_instances<1:
            raise ValueError('Positive image dimensions divisible by 16 and max_instances >=1 required')
        if not view_ids or len(set(view_ids))!=len(view_ids) or any(v not in range(6) for v in view_ids):
            raise ValueError('Distinct DANNCE view IDs 0..5 required')
        if confidence_min_votes not in (4,5,6):
            raise ValueError('confidence_min_votes must be 4, 5 or 6')
        catalog=json.loads((self.root/'temporal_catalog.json').read_text())['datasets']
        ids=[r['dataset_id'] for r in catalog if r['dataset_id'].startswith('DANNCE-MarkerlessMouse')
             and '/sessions/' in r['dataset_id']]
        self.index=TemporalWindowIndex(self.root,split=split,window_size=window_size,
            frame_step=frame_step,window_stride=window_stride,dataset_ids=ids,require_rgb=True)
        if not len(self.index):
            raise ValueError('No complete DANNCE temporal windows for requested split')
        self.image_size=tuple(image_size)
        self.views=tuple(view_ids)
        self.max_instances=max_instances
        # New name covers complete sequences; preserve the old pilot opt-in API.
        self.pseudo=load_pilot_pseudo_depth if load_pseudo_depth is None else load_pseudo_depth
        self.confidence_threshold=int(np.floor(confidence_min_votes/6*255))
        self.metadata={}
        self.pilot={}
        for _,path,_,_,_ in self.index.sequences:
            self.metadata[path]=json.loads((path/'metadata.json').read_text())
            if self.pseudo:
                for mp in sorted((path/'pseudo_depth_pilot').glob('*/metadata.json')):
                    pm=json.loads(mp.read_text())
                    for line in (mp.parent/'frames.jsonl').read_text().splitlines():
                        row=json.loads(line)
                        key=(path,row['source_frame_id'])
                        if key in self.pilot:
                            raise ValueError(f'Overlapping pseudo depth pilots: {key}')
                        self.pilot[key]=(mp.parent,row,pm)

    def __len__(self):
        return len(self.index)

    def __getitem__(self,index):
        window=self.index[index]
        path=window['root']
        meta=self.metadata[path]
        rows=window['rows']
        T,V,M=len(rows),len(self.views),self.max_instances
        w,h=self.image_size
        images=np.zeros((T,V,3,h,w),np.float32)
        depths=np.zeros((T,V,h,w),np.float32)
        confidence=np.zeros_like(depths)
        depth_valid=np.zeros_like(depths,dtype=bool)
        depth_available=np.zeros((T,V),bool)
        body_masks=np.zeros((T,V,h,w),bool)
        mask_available=np.zeros((T,V),bool)
        mask_training_valid=np.zeros((T,V),bool)
        rectification_valid=np.zeros((V,h,w),bool)
        points=np.zeros((T,V,M,23,2),np.float32)
        point_valid=np.zeros(points.shape[:-1],bool)
        world=np.zeros((T,M,23,3),np.float32)
        world_valid=np.zeros(world.shape[:-1],bool)
        pose_valid=np.zeros((T,V),bool)
        K,E=[],[]
        for vi,view in enumerate(self.views):
            c=meta['cameras'][view]
            k=np.asarray(c['intrinsic'],np.float32).copy()
            k[0]*=w/1152
            k[1]*=h/1024
            K.append(k)
            E.append(np.asarray(c['extrinsic'],np.float32))
            valid=cv2.imread(str(path/'cams'/f'{view:08d}_rectification_valid.png'),0)
            rectification_valid[vi]=cv2.resize(valid,(w,h),interpolation=cv2.INTER_NEAREST)>0
        for t,row in enumerate(rows):
            for vi,view in enumerate(self.views):
                im=cv2.imread(str(path/row['rgb'][view]))
                if im is None or im.shape[:2]!=(1024,1152):
                    raise ValueError(f'Missing or malformed RGB {path/row["rgb"][view]}')
                im=cv2.cvtColor(cv2.resize(im,(w,h),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
                images[t,vi]=im.transpose(2,0,1).astype(np.float32)/255
            if row['manual_keypoints'] is not None:
                with np.load(path/row['manual_keypoints'],allow_pickle=False) as f:
                    assert int(f['source_video_frame_zero_based'])==row['source_frame_id']
                    points[t,:,0]=f['uv'][list(self.views),0]/np.array([1152,1024],np.float32)
                    point_valid[t,:,0]=f['in_frame'][list(self.views),0]
                    world[t,0]=f['joint_world_xyz'][0]
                    world_valid[t,0]=f['world_valid'][0]
                pose_valid[t]=True
            pilot=self.pilot.get((path,row['source_frame_id']))
            if self.pseudo and row.get('depth') and row.get('confidence') and row.get('mask'):
                # Completed full-depth chunks are indexed relative to the session.
                pilot=(path,row,{'pfm_flip_y':row.get('depth_pfm_flip_y',True)})
            if pilot is not None:
                pp,pr,pm=pilot
                for vi,view in enumerate(self.views):
                    d=read_depth_pfm(pp/pr['depth'][view],pm['pfm_flip_y'])
                    conf=cv2.imread(str(pp/pr['confidence'][view]),0)
                    mask=cv2.imread(str(pp/pr['mask'][view]),0)
                    if d.shape!=(1024,1152) or conf is None or mask is None or conf.shape!=d.shape or mask.shape!=d.shape:
                        raise ValueError('Pseudo depth/mask/confidence resolution mismatch')
                    d=cv2.resize(d,(w,h),interpolation=cv2.INTER_NEAREST)
                    conf=cv2.resize(conf,(w,h),interpolation=cv2.INTER_NEAREST)
                    body_masks[t,vi]=(cv2.resize(mask,(w,h),interpolation=cv2.INTER_NEAREST)>0)&rectification_valid[vi]
                    mask_available[t,vi]=True
                    mask_training_valid[t,vi]=pr.get('pseudo_mask_training_valid',True)
                    depths[t,vi]=d
                    confidence[t,vi]=conf.astype(np.float32)/255
                    depth_valid[t,vi]=(d>0)&np.isfinite(d)&(conf>=self.confidence_threshold)&rectification_valid[vi]&body_masks[t,vi]&pr.get('pseudo_depth_training_valid',True)
                    depth_available[t,vi]=True
        instance_valid=point_valid.any(-1)
        instance_ids=np.full((T,V,M),-1,np.int64)
        instance_ids[:,:,0]=np.where(instance_valid[:,:,0],0,-1)
        result={'session_id':meta['dataset_id'],'motion_source_id':window['motion_source_id'],
            'split':window['split'],'images':images,'depths':depths,'depth_masks':depth_valid,
            'depth_confidence':confidence,'depth_available':depth_available,
            'depth_is_pseudo_label':depth_available.copy(),'rectification_valid':rectification_valid,
            'body_masks':body_masks,'mask_available':mask_available,
            'mask_is_pseudo_label':mask_available.copy(),'mask_supervision_valid':mask_training_valid,
            'intrinsics':np.stack(K),'extrinsics':np.stack(E),'camera_available':np.ones(V,bool),
            'frame_ids':window['frame_ids'],'frame_delta':window['frame_delta'],
            'time_seconds':np.array([r['time_seconds'] for r in rows],np.float64),
            'timestamp_valid':np.ones(T,bool),'temporal_transition_valid':np.ones(max(0,T-1),bool),
            'pose_supervision_valid':pose_valid,'instance_valid':instance_valid,'instance_ids':instance_ids,
            'instance_keypoint_xy':points,'instance_keypoint_valid':point_valid,
            'instance_keypoint_world_xyz':world,'instance_world_valid':world_valid,
            'instance_keypoint_visible':np.zeros_like(point_valid),
            'instance_keypoint_visibility_valid':np.zeros_like(point_valid),
            'length_unit':'mm','fixed_camera':True}
        return {k:torch.from_numpy(v) if isinstance(v,np.ndarray) else v for k,v in result.items()}
