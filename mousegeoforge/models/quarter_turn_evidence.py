"""Pixel-exact in-plane rotations for the isolated local-evidence diagnostic.

No mirrors, interpolation, anatomical relabeling or camera/3D supervision.
Coordinates are pixel-index coordinates divided by width/height.
"""
import torch

def quarter_turn(image,target,eligible,image_hw,k):
    if k not in (0,1,2,3):raise ValueError('Quarter turns must be 0..3')
    h,w=map(int,image_hw)
    if image.ndim!=4 or image.shape[:2]!=(1,3) or image.shape[-2]<h or image.shape[-1]<w:raise ValueError('Expected normalized padded single RGB')
    if target.shape[-1]!=2 or eligible.shape!=target.shape[:-1]:raise ValueError('Target shape')
    if k==0:return image,target,(h,w)
    xy=target*target.new_tensor([w,h]);out=xy.clone();nh,nw=h,w
    for _ in range(k):
        x,y=out.unbind(-1);out=torch.stack((y,nw-1-x),-1);nh,nw=nw,nh
    out=out/target.new_tensor([nw,nh]);out=torch.where(eligible[...,None],out,target)
    if not torch.isfinite(out[eligible]).all() or ((out[eligible]<0)|(out[eligible]>1)).any():raise ValueError('Rotated known point outside image; check coordinate convention')
    rgb=torch.rot90(image[...,:h,:w],k,(-2,-1));black=-image.new_tensor([.485,.456,.406])/image.new_tensor([.229,.224,.225]);padded=black[None,:,None,None].expand(1,3,nh+(-nh)%32,nw+(-nw)%32).clone();padded[...,:nh,:nw]=rgb
    return padded,out,(nh,nw)
