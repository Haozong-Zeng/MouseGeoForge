"""Experimental frozen heatmap readout constrained only by predicted boxes.

No confidence threshold, instance count, GT, or dataset-kind routing is accepted.
Non-differentiable argmax diagnostic, not a trained replacement architecture.
"""
import torch

CORE_INDICES=(0,14,15,16,17)

def specialist_roi_readout(heatmaps,boxes,baseline_xy,image_hw,*,stride=4.,enabled=False):
    q=boxes.shape[0]
    if not enabled:
        return dict(xy=baseline_xy,supported=torch.zeros(q,5,dtype=torch.bool,device=baseline_xy.device),scores=baseline_xy.new_zeros(q,5))
    if boxes.shape!=(q,4) or baseline_xy.shape!=(q,23,2) or heatmaps.ndim!=3 or heatmaps.shape[0]!=5:
        raise ValueError('Expected predicted Q boxes, Qx23 fallback and five heatmaps')
    h,w=image_hw
    if h<=0 or w<=0 or stride<=0:raise ValueError('Positive sizes required')
    hh,ww=heatmaps.shape[-2:];device=heatmaps.device
    if boxes.device!=device or baseline_xy.device!=device:raise ValueError('Device mismatch')
    xx=(torch.arange(ww,device=device,dtype=heatmaps.dtype)+.5)*stride
    yy=(torch.arange(hh,device=device,dtype=heatmaps.dtype)+.5)*stride
    cy,cx=torch.meshgrid(yy,xx,indexing='ij');grid=torch.stack((cx/w,cy/h),-1)
    low=boxes[:,:2]-boxes[:,2:]/2;high=boxes[:,:2]+boxes[:,2:]/2
    valid_box=torch.isfinite(boxes).all(-1)&(boxes[:,2:]>0).all(-1)
    inside=(grid[None]>=low[:,None,None]).all(-1)&(grid[None]<=high[:,None,None]).all(-1)
    inside=inside&valid_box[:,None,None]&(cx<w)[None]&(cy<h)[None]
    finite=torch.isfinite(heatmaps)
    support=inside[:,None]&finite[None]
    supported=support.flatten(2).any(-1)
    masked=heatmaps[None].expand(q,-1,-1,-1).masked_fill(~support,-torch.inf).flatten(2)
    values,index=masked.max(-1)
    coords=torch.stack(((index%ww+.5)*stride/w,(index//ww+.5)*stride/h),-1).to(baseline_xy.dtype)
    output=baseline_xy.clone();output[:,CORE_INDICES]=torch.where(supported[...,None],coords,baseline_xy[:,CORE_INDICES])
    return dict(xy=output,supported=supported,scores=torch.where(supported,values,torch.zeros_like(values)))
