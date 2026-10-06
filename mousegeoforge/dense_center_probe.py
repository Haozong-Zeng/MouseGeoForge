"""CenterNet-inspired frozen-feature candidate probe; not the keypoint model."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from mousegeoforge.pilot_metrics import box_iou
from training.instance_pose_losses import _linear_sum_assignment


class DenseCenterProbe(nn.Module):
    def __init__(self, channels=256, hidden=64):
        super().__init__()
        self.shared=nn.Conv2d(channels,hidden,1)
        self.output=nn.Conv2d(hidden,5,1)
        with torch.no_grad():
            self.output.weight.mul_(.01);self.output.bias.zero_()
            self.output.bias[0]=-2.19
            self.output.bias[3:]=math.log(.2/.8)

    def forward(self, features):
        out=self.output(F.relu(self.shared(F.normalize(features,dim=1,eps=1e-6))))
        return dict(heat_logits=out[:,0],offset=out[:,1:3].sigmoid(),size=out[:,3:5].sigmoid())


def center_targets(boxes, valid, grid_hw):
    """One center per cell; explicit collision errors prevent silent label dropping."""
    height,width=grid_hw
    if boxes.ndim!=3 or boxes.shape[-1]!=4 or valid.shape!=boxes.shape[:2]:raise ValueError('Expected [V,I,4] boxes and [V,I] validity')
    heat=boxes.new_zeros(boxes.shape[0],height,width);offset=boxes.new_zeros(boxes.shape[0],2,height,width)
    size=torch.zeros_like(offset);mask=torch.zeros_like(heat,dtype=torch.bool)
    yy,xx=torch.meshgrid(torch.arange(height,device=boxes.device),torch.arange(width,device=boxes.device),indexing='ij')
    near_pairs=0
    for view in range(len(boxes)):
        cells=[]
        for instance in torch.where(valid[view])[0]:
            b=boxes[view,instance]
            if not torch.isfinite(b).all() or (b[:2]<0).any() or (b[:2]>1).any() or (b[2:]<=0).any():raise ValueError('Invalid normalized GT box')
            xy=(b[:2]*b.new_tensor([width,height])).clamp_min(0)
            xy=torch.minimum(xy,b.new_tensor([width-1e-4,height-1e-4]))
            ix,iy=xy.floor().long().tolist()
            if mask[view,iy,ix]:raise ValueError('Two GT centers collide in one feature cell')
            near_pairs+=sum(max(abs(ix-x),abs(iy-y))<=1 for x,y in cells);cells.append((ix,iy))
            sigma=(b[2:]*b.new_tensor([width,height])).min().div(6).clamp(.5,2.)
            gaussian=torch.exp(-((xx-ix).square()+(yy-iy).square())/(2*sigma.square()))
            heat[view]=torch.maximum(heat[view],gaussian)
            offset[view,:,iy,ix]=xy-xy.floor();size[view,:,iy,ix]=b[2:]
            mask[view,iy,ix]=True
    return dict(heat=heat,offset=offset,size=size,mask=mask,near_center_pairs=near_pairs)


def center_loss(pred,target):
    logits=pred['heat_logits'];p=logits.sigmoid();truth=target['heat'];positive=target['mask']
    pos=-F.logsigmoid(logits)*(1-p).square()*positive
    neg=-F.logsigmoid(-logits)*p.square()*(1-truth).pow(4)*(~positive)
    heat=(pos.sum()+neg.sum())/positive.sum().clamp_min(1)
    mask=positive[:,None].expand(-1,2,-1,-1)
    if bool(positive.any()):
        offset=F.l1_loss(pred['offset'][mask],target['offset'][mask])
        size=F.l1_loss(pred['size'][mask],target['size'][mask])
    else:
        offset=pred['offset'].sum()*0;size=pred['size'].sum()*0
    return heat+offset+5*size,dict(heat=heat,offset=offset,size=size)


def decode_centers(pred, topk=6):
    prob=pred['heat_logits'].sigmoid();n,height,width=prob.shape
    if topk>height*width:raise ValueError('Too many candidates')
    peaks=prob*prob.eq(F.max_pool2d(prob[:,None],3,stride=1,padding=1)[:,0])
    scores,indices=peaks.flatten(1).topk(topk,dim=1)
    offsets=pred['offset'].flatten(2).gather(2,indices[:,None].expand(-1,2,-1)).transpose(1,2)
    sizes=pred['size'].flatten(2).gather(2,indices[:,None].expand(-1,2,-1)).transpose(1,2)
    cells=torch.stack((indices%width,indices//width),dim=-1).to(offsets)
    xy=(cells+offsets)/offsets.new_tensor([width,height])
    return dict(boxes=torch.cat((xy,sizes),dim=-1),scores=scores,heatmap=prob)


def candidate_metrics(boxes,scores,targets):
    if len(targets)==0:return dict(tp=0,fp=int((scores>=.5).sum()),fn=0,all_topk_covered=0,gt=0,duplicate=False,matches=[])
    cost=5*torch.cdist(targets,boxes,p=1)-scores[None]
    pred,rows=_linear_sum_assignment(cost)
    ious=box_iou(targets[rows],boxes[pred]).diagonal()
    found=(scores[pred]>=.5)&(ious>=.5)
    active=scores>=.5;overlap=box_iou(targets,boxes[active])
    best=overlap.max(0) if bool(active.any()) else None
    mapped=best.indices[best.values>=.5].tolist() if best is not None else []
    return dict(tp=int(found.sum()),fp=int(active.sum()-found.sum()),fn=len(targets)-int(found.sum()),
                all_topk_covered=int((ious>=.5).sum()),gt=len(targets),duplicate=len(mapped)!=len(set(mapped)),
                matches=[dict(query=int(q),target=int(t),iou=float(iou),score=float(scores[q]),found=bool(f))
                         for q,t,iou,f in zip(pred,rows,ious,found)])
