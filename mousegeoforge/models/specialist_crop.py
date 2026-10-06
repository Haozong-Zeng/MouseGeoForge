"""Fixed predicted-box crop geometry for an experimental frozen specialist.

Coordinates use continuous image-edge coordinates, like the original loader.
No labels, instance counts, confidence or dataset identities are accepted.
"""
import math
import numpy as np
import torch
from PIL import Image
from .specialist_roi_readout import specialist_roi_readout, CORE_INDICES


def make_specialist_crop(image, box, *, expansion=1.25, canvas_hw=(768,704)):
    w,h=image.size;box=np.asarray(box,dtype=np.float64)
    if box.shape!=(4,) or not np.isfinite(box).all() or np.any(box[2:]<=0):return None
    if not math.isfinite(expansion) or expansion<=0:raise ValueError('Invalid expansion')
    cx,cy,bw,bh=box*np.array([w,h,w,h]);bw*=expansion;bh*=expansion
    left,top=math.floor(cx-bw/2),math.floor(cy-bh/2)
    right,bottom=math.ceil(cx+bw/2),math.ceil(cy+bh/2)
    if right<=0 or bottom<=0 or left>=w or top>=h:return None
    ch,cw=canvas_hw
    if min(ch,cw)<=0:raise ValueError('Invalid canvas')
    rw,rh=right-left,bottom-top
    if rw>4*w or rh>4*h:return None  # bounded allocation, explicit invalid fallback
    scale=min(cw/rw,ch/rh);nw=max(1,min(cw,round(rw*scale)));nh=max(1,min(ch,round(rh*scale)))
    px,py=(cw-nw)//2,(ch-nh)//2;sx,sy=nw/rw,nh/rh
    crop=image.crop((left,top,right,bottom)).resize((nw,nh),Image.Resampling.BICUBIC)
    canvas=Image.new('RGB',(cw,ch));canvas.paste(crop,(px,py))
    affine=np.array([[sx,0,px-left*sx],[0,sy,py-top*sy],[0,0,1]],dtype=np.float64)
    valid=np.array([max(left,0)*sx+affine[0,2],max(top,0)*sy+affine[1,2],min(right,w)*sx+affine[0,2],min(bottom,h)*sy+affine[1,2]])
    meta=dict(source_hw=[h,w],canvas_hw=[ch,cw],crop_ltrb=[left,top,right,bottom],resized_wh=[nw,nh],pad_xy=[px,py],affine=affine.tolist(),valid_ltrb=valid.tolist(),expansion=expansion)
    return canvas,meta


def decode_specialist_crop(heatmaps, baseline_xy, meta, *, enabled=False):
    if not enabled or meta is None:return baseline_xy
    ch,cw=meta['canvas_hw'];h,w=meta['source_hw'];l,t,r,b=meta['valid_ltrb']
    box=heatmaps.new_tensor([[(l+r)/(2*cw),(t+b)/(2*ch),(r-l)/cw,(b-t)/ch]])
    # Baseline here is only a fallback; coordinate conversion is applied only to supported points.
    out=specialist_roi_readout(heatmaps,box,baseline_xy[None],(ch,cw),enabled=True)
    A=heatmaps.new_tensor(meta['affine']);canvas_xy=out['xy'][0,CORE_INDICES]*heatmaps.new_tensor([cw,ch])
    xy=(canvas_xy-A[:2,2])/A.diag()[:2]/heatmaps.new_tensor([w,h])
    result=baseline_xy.clone();result[list(CORE_INDICES)]=torch.where(out['supported'][0,:,None],xy,baseline_xy[list(CORE_INDICES)])
    return result
