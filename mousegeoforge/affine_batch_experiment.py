"""Strict synthetic batch adapter for the isolated spatial-anchor pilot."""
import copy
import cv2
import numpy as np
import torch
from .affine_augmentation import centered_affine,transform_xy,warp_view


class UnsafeCrop(ValueError):
    pass


def augment_batch(batch, *, scale=1., dx=0., dy=0., minimum_mask_retained=.98):
    """Same affine on every view; no physical camera/world/depth scale change.

    Reject the entire variant if any valid instance loses >2% GT mask area,
    all supervised points, or its box. Never remove individual mice or use
    predicted outcomes for this acceptance decision. Identity is exact cloning.
    """
    if batch['images'].device.type!='cpu' or batch['images'].shape[0]!=1:
        raise ValueError('Only B=1 CPU pre-encoding batches supported')
    out={k:v.clone() if isinstance(v,torch.Tensor) else copy.deepcopy(v) for k,v in batch.items()}
    h,w=batch['images'].shape[-2:];a=centered_affine(h,w,scale=scale,dx=dx,dy=dy)
    if np.array_equal(a,np.eye(3)):
        return out,dict(affine=a.tolist(),identity=True,minimum_retained=1.)
    retained=[]
    for v in range(batch['images'].shape[1]):
        def src(k):return batch[k][0,v].numpy()
        def put(k,value):out[k][0,v]=torch.as_tensor(value,dtype=out[k].dtype)
        b=src('boxes_cxcywh')*np.array([w,h,w,h])
        boxes=np.concatenate([b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2],axis=-1)
        r=warp_view(rgb=src('images').transpose(1,2,0),depth=src('depths'),depth_valid=src('depth_masks'),
                    masks=src('masks'),xy=src('instance_keypoint_xy')*[w,h],point_valid=src('instance_keypoint_valid'),
                    boxes_xyxy=boxes,intrinsic=src('intrinsics'),extrinsic=src('extrinsics'),affine=a)
        for m in np.where(src('instance_valid'))[0]:
            if not src('mask_valid')[m] or not src('box_valid')[m]:raise ValueError('Pilot needs complete instance masks/boxes')
            ratio=float(r['masks'][m].sum()/max(1,src('masks')[m].sum()*scale**2));retained.append(ratio)
            if ratio<minimum_mask_retained or not r['box_in_frame'][m] or not r['point_valid'][m].any():
                raise UnsafeCrop(f'view{v} instance{m} retained={ratio:.6f}')
        put('images',r['rgb'].transpose(2,0,1));put('depths',r['depth']);put('depth_masks',r['depth_valid'])
        put('intrinsics',r['intrinsic']);put('masks',r['masks'])
        for k in ('body_masks','semantic_masks'):
            # Integer semantic IDs must survive exactly; float32 represents these small IDs.
            put(k,cv2.warpAffine(src(k).astype(np.float32),a[:2],(w,h),flags=cv2.INTER_NEAREST,borderValue=0))
        put('instance_keypoint_xy',r['xy']/[w,h]);put('instance_keypoint_valid',r['point_valid'])
        put('instance_keypoint_in_frame',r['point_in_frame'])
        put('instance_keypoint_visible',src('instance_keypoint_visible')&r['point_in_frame'])
        put('instance_keypoint_visibility_valid',src('instance_keypoint_visibility_valid')&r['point_in_frame'])
        clipped=r['boxes_xyxy'];newboxes=np.concatenate([(clipped[:,:2]+clipped[:,2:])/2,clipped[:,2:]-clipped[:,:2]],-1)/[w,h,w,h]
        # Preserve zero padded slots, including their flags and coordinate sentinels.
        newboxes[~src('instance_valid')]=0;put('boxes_cxcywh',newboxes)
        put('box_valid',src('box_valid')&r['box_in_frame'])
        legacy=transform_xy(src('keypoint_uv'),a)
        inside=np.isfinite(legacy).all(-1)&(legacy[...,0]>=0)&(legacy[...,0]<w)&(legacy[...,1]>=0)&(legacy[...,1]<h)
        put('keypoint_uv',legacy);put('keypoint_in_frame',inside)
        put('keypoint_surface_visible',src('keypoint_surface_visible')&inside)
        for k in ('instance_keypoint_xy','instance_keypoint_valid','instance_keypoint_in_frame',
                  'instance_keypoint_visible','instance_keypoint_visibility_valid'):
            out[k][0,v,~batch['instance_valid'][0,v]]=batch[k][0,v,~batch['instance_valid'][0,v]]
    # Label semantics, world validity, IDs, all world coordinates and E unchanged.
    return out,dict(affine=a.tolist(),identity=False,minimum_retained=min(retained))
