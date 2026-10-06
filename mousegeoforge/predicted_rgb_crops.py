"""Deterministic RGB crops from prediction boxes; no annotation inputs."""
import math
import torch
from torch.nn import functional as F
MEAN=(.485,.456,.406)
STD=(.229,.224,.225)


def predicted_rgb_crops(image,boxes,size=224,*,grayscale=False):
    if image.ndim!=3 or image.shape[0]!=3 or boxes.ndim!=2 or boxes.shape[-1]!=4 or size<16:raise ValueError('Expected RGB[3,H,W], predicted boxes[Q,4]')
    if not torch.isfinite(image).all():raise ValueError('Nonfinite RGB')
    h,w=image.shape[-2:];q=len(boxes);mean=image.new_tensor(MEAN)[:,None,None]
    crops=mean.expand(q,3,size,size).clone();valid=torch.zeros(q,dtype=torch.bool,device=image.device);bounds=torch.zeros(q,4,dtype=torch.long);content=torch.zeros(q,4,dtype=torch.long)
    for i,box in enumerate(boxes.detach().cpu().float()):
        if not torch.isfinite(box).all() or (box[2:]<=0).any():continue
        cx,cy,bw,bh=box.tolist();x0=max(0,min(w,math.floor((cx-bw/2)*w)));x1=max(0,min(w,math.ceil((cx+bw/2)*w)))
        y0=max(0,min(h,math.floor((cy-bh/2)*h)));y1=max(0,min(h,math.ceil((cy+bh/2)*h)))
        bounds[i]=torch.tensor([x0,y0,x1,y1])
        if x1<=x0 or y1<=y0:continue
        crop=image[:,y0:y1,x0:x1]
        if grayscale:crop=(crop*image.new_tensor([.299,.587,.114])[:,None,None]).sum(0,keepdim=True).expand(3,-1,-1)
        scale=size/max(y1-y0,x1-x0);nh=max(1,min(size,round((y1-y0)*scale)));nw=max(1,min(size,round((x1-x0)*scale)))
        small=F.interpolate(crop[None].float(),size=(nh,nw),mode='bilinear',align_corners=False,antialias=True)[0]
        top=(size-nh)//2;left=(size-nw)//2;crops[i,:,top:top+nh,left:left+nw]=small
        content[i]=torch.tensor([left,top,left+nw,top+nh]);valid[i]=True
    return dict(images=crops,valid=valid,bounds=bounds,content_bounds=content)
