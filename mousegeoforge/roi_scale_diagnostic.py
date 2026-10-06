"""Oracle crop geometry for frozen diagnostics, never production detection."""
import cv2
import numpy as np
from .affine_augmentation import transform_xy

BODY=list(range(7))+list(range(14,23))


def body_roi_affine(xy,valid,image_hw,*,margin=1.5,max_scale=3.):
    h,w=image_hw
    xy=np.asarray(xy,dtype=np.float64);valid=np.asarray(valid,dtype=bool)
    if xy.shape!=(23,2) or valid.shape!=(23,) or min(h,w)<=0 or margin<1 or max_scale<=0:
        raise ValueError('Expected Core23 pixel coordinates and positive crop geometry')
    use=valid[BODY]&np.isfinite(xy[BODY]).all(-1);points=xy[BODY][use]
    if len(points)<2:raise ValueError('Fewer than two valid body points; diagnostic cannot define crop')
    low=points.min(0);high=points.max(0);extent=np.maximum(high-low,1.)
    scale=min(max_scale,(w-1)/(margin*extent[0]),(h-1)/(margin*extent[1]))
    center=(low+high)/2;shift=np.array([(w-1)/2,(h-1)/2])-scale*center
    affine=np.array([[scale,0,shift[0]],[0,scale,shift[1]],[0,0,1.]])
    projected=transform_xy(points,affine)
    if not ((projected>=0).all() and (projected<np.array([w,h])).all()):raise AssertionError('Body points cropped')
    return affine,dict(scale=float(scale),body_extent_px=extent.tolist(),body_points=len(points),margin=margin)


def warp_rgb_masks(rgb,masks,affine):
    h,w=rgb.shape[:2];a=np.asarray(affine,dtype=np.float64)
    if masks.ndim!=3 or masks.shape[1:]!=(h,w) or a.shape!=(3,3):raise ValueError('Raster geometry mismatch')
    out=cv2.warpAffine(rgb,a[:2],(w,h),flags=cv2.INTER_LINEAR,borderValue=0)
    all_masks=np.stack([cv2.warpAffine(m.astype(np.uint8),a[:2],(w,h),flags=cv2.INTER_NEAREST)>0 for m in masks])
    support=cv2.warpAffine(np.ones((h,w),np.uint8),a[:2],(w,h),flags=cv2.INTER_NEAREST)>0
    return out,all_masks,float(1-support.mean())


def inverse_points(xy,affine):
    return transform_xy(xy,np.linalg.inv(affine))


def mask_query_iou(predicted_masks,target_mask):
    if predicted_masks.ndim!=3 or predicted_masks.shape[1:]!=target_mask.shape:raise ValueError('Mask sizes differ')
    intersection=(predicted_masks&target_mask).sum((-2,-1))
    union=(predicted_masks|target_mask).sum((-2,-1))
    iou=intersection/np.maximum(union,1)
    return int(np.argmax(iou)),iou
