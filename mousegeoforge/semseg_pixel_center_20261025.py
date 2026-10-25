"""Versioned label-center helper for an independent frozen-encoder head pair."""
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image
from mousegeoforge.semseg_training import letterbox

def native_label(row,root):
    path=Path(root)/row['label']
    if path.suffix=='.npz':
        with np.load(path,allow_pickle=False) as z:label=z['body'].astype(np.uint8)
    else:
        raw=np.asarray(Image.open(path).convert('L'))
        label=raw if row.get('label_encoding')=='class_index_0_1_255' else (raw>0).astype(np.uint8)
    if not set(np.unique(label))<={0,1,255}:raise ValueError('Invalid pixel labels')
    return label

def label_letterbox(image,label,side,policy):
    if policy not in ['legacy','center_exact']:raise ValueError('Unknown label operator')
    image_tensor,target,meta=letterbox(image,label,side)
    if policy=='center_exact':
        y,x,h,w=meta['roi'];target=target.clone()
        target[y:y+h,x:x+w]=torch.from_numpy(cv2.resize(label.astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST_EXACT).astype(np.int64))
    return image_tensor,target,meta

def read_center_sample(row,root,side,policy):
    image=np.asarray(Image.open(Path(root)/row['image']).convert('RGB'))
    return label_letterbox(image,native_label(row,root),side,policy)

def native_probability(probability,meta):
    y,x,h,w=meta['roi']
    value=cv2.resize(probability[y:y+h,x:x+w].astype(np.float32),tuple(meta['native_hw'][::-1]),interpolation=cv2.INTER_LINEAR)
    if not np.isfinite(value).all():raise ValueError('Nonfinite probability')
    return value,(value>.5).astype(np.uint8)
