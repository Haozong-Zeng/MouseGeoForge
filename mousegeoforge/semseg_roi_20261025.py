"""Predicted-box ROI only; full-native probability restoration and GT denominator."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from mousegeoforge.semseg_pixel_center_20261025 import native_label,label_letterbox,native_probability as local_probability

def predicted_box(mask,padding=1.4,minimum=96):
 h,w=mask.shape;y,x=np.where(mask)
 if not len(x):return [0,0,w,h]
 side=min(max(h,w),max(minimum,int(np.ceil(max(x.max()-x.min()+1,y.max()-y.min()+1)*padding))))
 cx=(float(x.min())+float(x.max())+1)/2;cy=(float(y.min())+float(y.max())+1)/2
 x0=max(0,min(w-side,int(np.floor(cx-side/2)))) if w>=side else 0
 y0=max(0,min(h-side,int(np.floor(cy-side/2)))) if h>=side else 0
 return [x0,y0,min(w,x0+side),min(h,y0+side)]

def roi_sample(image,label,side,mode,box):
 h,w=image.shape[:2]
 if label.shape!=(h,w):raise ValueError('Image/label mismatch')
 if mode not in ['full','roi']:raise ValueError('Unknown crop mode')
 x0,y0,x1,y1=([0,0,w,h] if mode=='full' else [int(x) for x in box])
 if not (0<=x0<x1<=w and 0<=y0<y1<=h):raise ValueError('Invalid predicted crop')
 tensor,target,meta=label_letterbox(image[y0:y1,x0:x1],label[y0:y1,x0:x1],side,'center_exact')
 meta['full_native_hw']=[h,w];meta['crop_xyxy']=[x0,y0,x1,y1]
 affine=np.asarray(meta['pixel_center_affine']);affine[:,2]-=affine[:,:2]@np.array([x0,y0]);meta['pixel_center_affine']=affine.tolist()
 return tensor,target,meta

def read_roi_sample(row,root,side,mode):
 image=np.asarray(Image.open(Path(root)/row['image']).convert('RGB'))
 return roi_sample(image,native_label(row,root),side,mode,row['roi_box'])

def native_probability(probability,meta):
 value,_=local_probability(probability,meta)
 h,w=meta['full_native_hw'];x0,y0,x1,y1=meta['crop_xyxy'];full=np.zeros((h,w),np.float32)
 if value.shape!=(y1-y0,x1-x0):raise ValueError('Crop inverse mismatch')
 full[y0:y1,x0:x1]=value
 return full,(full>.5).astype(np.uint8)
