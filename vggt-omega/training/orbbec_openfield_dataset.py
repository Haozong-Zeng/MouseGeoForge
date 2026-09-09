"""Single-view labeled scenes and complete RGB-D temporal windows.

Time is a separate T axis. Unlabeled frames are NOT negative mouse examples.
No camera or skeletal 3D ground truth is asserted for reference calibration.
"""
from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def build_orbbec_labeled_dataset(root, *, split='train', image_size=(512,512), max_instances=2):
    """Use the stock BlenderPyMouse loader with a subject-disjoint scene list."""
    from training.finetune_blendedmvs import BlendedMVSDataset, load_scene
    if max_instances < 1 or min(image_size) <= 0 or any(v % 16 for v in image_size):
        raise ValueError('Positive instance count and image dimensions divisible by 16 required')
    root=Path(root)
    splits=json.loads((root/'splits.json').read_text())['labeled_scenes']
    paths=[p for group in splits.values() for p in group] if split is None else splits[split]
    dataset=BlendedMVSDataset(scenes=[load_scene(root/p) for p in paths],num_views=1,
        image_width=image_size[0],image_height=image_size[1],samples_per_scene=1,seed=5706,
        use_masked_images=False,pfm_flip_y=False,canonicalize_camera=False,scale_mode='raw',top_srcs=1,load_keypoints=True)
    dataset.max_instances=max_instances
    return dataset


class OrbbecOpenfieldTemporalDataset(Dataset):
    """Decode immutable encoded depths on demand without writing a PFM cache.

    RGB: [T,1,3,H,W]; depth: [T,1,H,W]; 2D points: [T,1,M,23,2].
    Consumers must gate all instance/objectness losses by pose_supervision_valid.
    Metric velocity losses require timestamps, which legacy exports do not supply.
    """
    def __init__(self,root,*,split='train',window_size=16,frame_step=1,window_stride=16,
                 image_size=(512,512),max_instances=2):
        self.root=Path(root)
        if min(window_size,frame_step,window_stride,max_instances)<=0:
            raise ValueError('Window parameters and max_instances must be positive')
        if min(image_size)<=0 or any(v%16 for v in image_size):
            raise ValueError('Image dimensions must be positive multiples of 16')
        self.window_size=window_size
        self.frame_step=frame_step
        self.image_size=tuple(image_size)
        self.max_instances=max_instances
        self.sessions=[]
        self.windows=[]
        for line in (self.root/'sequences.jsonl').read_text().splitlines():
            rec=json.loads(line)
            if split is not None and rec['split']!=split:
                continue
            path=self.root/rec['session_path']
            with np.load(path/'frame_index.npz',allow_pickle=False) as f:
                ids=f['frame_ids'].copy()
                annidx=f['annotation_index'].copy()
            with np.load(path/'keypoint_annotations.npz',allow_pickle=False) as f:
                annotations={k:f[k].copy() for k in f.files}
            calibration=json.loads((path/'calibration.json').read_text())
            session_index=len(self.sessions)
            self.sessions.append((rec,path,ids,annidx,annotations,calibration))
            span=(window_size-1)*frame_step
            if len(ids)<=span:
                continue
            bad_prefix=np.r_[0,np.cumsum(np.diff(ids)!=1)]
            starts=np.arange(0,len(ids)-span,window_stride)
            starts=starts[bad_prefix[starts+span]==bad_prefix[starts]]
            self.windows.extend((session_index,int(s)) for s in starts)
        if not self.windows:
            raise ValueError('No gap-free windows for requested split and window size')

    def __len__(self):
        return len(self.windows)

    def __getitem__(self,index):
        si,start=self.windows[index]
        rec,path,frame_ids,annidx,ann,calib=self.sessions[si]
        positions=start+np.arange(self.window_size)*self.frame_step
        ids=frame_ids[positions]
        w,h=self.image_size
        images=[]
        depths=[]
        masks=[]
        m=self.max_instances
        point_xy=np.zeros((self.window_size,1,m,23,2),np.float32)
        point_valid=np.zeros(point_xy.shape[:-1],bool)
        visible=np.zeros_like(point_valid)
        vis_valid=np.zeros_like(point_valid)
        boxes=np.zeros((self.window_size,1,m,4),np.float32)
        box_valid=np.zeros(boxes.shape[:-1],bool)
        pose_valid=np.zeros((self.window_size,1),bool)
        for ti,(pos,fid) in enumerate(zip(positions,ids)):
            with Image.open(path/'rgb'/f'rgb_{fid:012d}.jpg') as im:
                im=im.convert('RGB')
                original_size=im.size
                images.append(np.asarray(im.resize((w,h),Image.Resampling.BICUBIC),np.float32).transpose(2,0,1)/255.)
            q=cv2.imread(str(path/'depth_encoded'/f'depth_{fid:012d}.png'),cv2.IMREAD_UNCHANGED)
            if q is None or q.dtype!=np.uint8 or q.shape[:2]!=original_size[::-1]:
                raise ValueError(f'Invalid encoded depth in {path}, frame {fid}')
            if q.ndim==3:
                if q.shape[2]!=3 or not (np.array_equal(q[:,:,0],q[:,:,1]) and np.array_equal(q[:,:,0],q[:,:,2])):
                    raise ValueError('Encoded depth channels must agree')
                q=q[:,:,0]
            q=cv2.resize(q,(w,h),interpolation=cv2.INTER_NEAREST)
            valid=(q>0)&(q<255)
            z=q.astype(np.float32)*rec['depth_decode_scale_mm']
            z[~valid]=0
            depths.append(z)
            masks.append(valid)
            ai=annidx[pos]
            if ai>=0:
                point_xy[ti,0,0]=ann['uv'][ai,0,0]/np.asarray(original_size)
                point_valid[ti,0,0]=ann['in_frame'][ai,0,0]
                visible[ti,0,0]=ann['surface_visible'][ai,0,0]
                vis_valid[ti,0,0]=ann['visibility_valid'][ai,0,0]
                boxes[ti,0,0]=ann['boxes_cxcywh'][ai,0,0]
                box_valid[ti,0,0]=ann['box_valid'][ai,0,0]
                pose_valid[ti,0]=True
        instance_valid=box_valid|point_valid.any(-1)
        instance_ids=np.full(instance_valid.shape,-1,np.int64)
        instance_ids[:,:,0]=np.where(pose_valid,0,-1)
        K=np.array(calib['K_crop_prior'],np.float32)
        K[0]*=w/calib['image_size'][0]
        K[1]*=h/calib['image_size'][1]
        return {'session_id':rec['session_id'],'frame_ids':torch.from_numpy(ids.copy()),
            'images':torch.from_numpy(np.stack(images)[:,None]),'depths':torch.from_numpy(np.stack(depths)[:,None]),
            'depth_masks':torch.from_numpy(np.stack(masks)[:,None]),'depth_available':torch.ones((self.window_size,1),dtype=torch.bool),
            'time_seconds':torch.full((self.window_size,),float('nan'),dtype=torch.float64),
            'timestamp_valid':torch.zeros(self.window_size,dtype=torch.bool),
            'temporal_transition_valid':torch.ones(max(0,self.window_size-1),dtype=torch.bool),
            'metric_velocity_valid':torch.zeros(max(0,self.window_size-1),dtype=torch.bool),
            'frame_delta':torch.from_numpy(np.diff(ids)),
            'camera_available':torch.zeros(1,dtype=torch.bool),'camera_prior_intrinsics':torch.from_numpy(K[None]),
            'camera_prior_verified':torch.zeros(1,dtype=torch.bool),'extrinsics':torch.eye(4)[None],
            'pose_supervision_valid':torch.from_numpy(pose_valid),'instance_valid':torch.from_numpy(instance_valid),
            'instance_ids':torch.from_numpy(instance_ids),'boxes_cxcywh':torch.from_numpy(boxes),'box_valid':torch.from_numpy(box_valid),
            'instance_keypoint_xy':torch.from_numpy(point_xy),'instance_keypoint_valid':torch.from_numpy(point_valid),
            'instance_keypoint_visible':torch.from_numpy(visible),'instance_keypoint_visibility_valid':torch.from_numpy(vis_valid),
            'instance_world_valid':torch.zeros((self.window_size,m,23),dtype=torch.bool)}
