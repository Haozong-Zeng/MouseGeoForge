"""Isolated pixel-space augmentation primitives; not enabled in production.

Coordinates are OpenCV integer pixel centers, boxes are xyxy pixel coordinates.
An image-plane affine changes K, never physical E or depth units. RGB must be
re-encoded after this function; moving cached features is not equivalent.
"""
import cv2
import numpy as np


def centered_affine(height, width, *, scale=1., dx=0., dy=0.):
    if min(height,width)<=0 or not np.isfinite([scale,dx,dy]).all() or scale<=0:
        raise ValueError('Positive dimensions/scale and finite transform required')
    return np.array([[scale,0,(1-scale)*(width-1)/2+dx],
                     [0,scale,(1-scale)*(height-1)/2+dy],[0,0,1]],dtype=np.float64)


def transform_xy(xy, affine):
    return np.asarray(xy)@affine[:2,:2].T+affine[:2,2]


def warp_view(*, rgb, depth, depth_valid, masks, xy, point_valid,
              boxes_xyxy, intrinsic, extrinsic, affine):
    """Return independent arrays for one view, retaining all instance slots.

    Visibility/world validity and IDs remain caller metadata, not recomputed
    from silhouettes. Caller must update in-frame/supervision flags, legacy
    coordinates, semantic/body masks, normalized boxes and count policies when
    adapting a complete loader batch. This primitive is NOT that batch adapter.
    """
    h,w=rgb.shape[:2];a=np.asarray(affine,dtype=np.float64)
    if rgb.shape!=(h,w,3) or depth.shape!=(h,w) or depth_valid.shape!=(h,w):
        raise ValueError('RGB/depth shape mismatch')
    if a.shape!=(3,3) or not np.isfinite(a).all() or not np.allclose(a[2],[0,0,1]):
        raise ValueError('Finite image-plane affine required')
    if not np.allclose(a[:2,:2],np.eye(2)*a[0,0]) or a[0,0]<=0:
        raise ValueError('Only positive isotropic scale plus translation supported')
    if masks.ndim!=3 or masks.shape[1:]!=(h,w) or xy.shape[:-1]!=point_valid.shape or xy.shape[-1]!=2:
        raise ValueError('Instance mask/point shapes invalid')
    if len(masks)!=len(xy) or boxes_xyxy.shape!=(len(masks),4):
        raise ValueError('Instance slots mismatch')
    def warp(x, mode):return cv2.warpAffine(x,a[:2],(w,h),flags=mode,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    pixels_valid=warp(np.ones((h,w),np.float32),cv2.INTER_LINEAR)>=1-1e-6
    zvalid=depth_valid.astype(bool)&np.isfinite(depth)&(depth>0)
    newz=warp(np.where(zvalid,depth,0).astype(np.float32),cv2.INTER_NEAREST)
    newvalid=warp(zvalid.astype(np.uint8),cv2.INTER_NEAREST).astype(bool)&pixels_valid
    newz[~newvalid]=0
    newxy=transform_xy(xy,a)
    in_frame=np.isfinite(newxy).all(-1)&(newxy[...,0]>=0)&(newxy[...,0]<w)&(newxy[...,1]>=0)&(newxy[...,1]<h)
    boxes=boxes_xyxy.copy().astype(np.float64)
    boxes[...,:2]=transform_xy(boxes_xyxy[...,:2],a);boxes[...,2:]=transform_xy(boxes_xyxy[...,2:],a)
    boxes[:,[0,2]]=boxes[:,[0,2]].clip(0,w);boxes[:,[1,3]]=boxes[:,[1,3]].clip(0,h)
    box_in_frame=(boxes[:,2]>boxes[:,0])&(boxes[:,3]>boxes[:,1])&np.isfinite(boxes).all(-1)
    return dict(rgb=warp(rgb,cv2.INTER_LINEAR),pixels_valid=pixels_valid,
                depth=newz,depth_valid=newvalid,
                masks=np.stack([warp(m.astype(np.uint8),cv2.INTER_NEAREST)>0 for m in masks]),
                xy=newxy,point_valid=point_valid.astype(bool)&in_frame,point_in_frame=in_frame,
                boxes_xyxy=boxes,box_in_frame=box_in_frame,
                intrinsic=a@intrinsic,extrinsic=extrinsic.copy(),affine=a.copy())
