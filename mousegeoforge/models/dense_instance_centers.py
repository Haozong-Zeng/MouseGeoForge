"""Experimental image-local projected-mask center detector.

Uses the modified focal objective described by CornerNet/CenterNet, implemented
here with numerically stable log-sigmoids. Reference implementation (not copied):
https://github.com/xingyizhou/CenterNet/blob/master/src/lib/models/losses.py
Centers denote visible projected mask centroids, NOT anatomical body centers.
No GT, bounding boxes, mouse counts, or externally provided proposals in forward.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .regional_peak_candidates import regional_top_peaks


class DenseInstanceCenters(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(32,64,3,padding=1),nn.GELU(),
                                    nn.Conv2d(64,64,3,padding=1),nn.GELU(),nn.Conv2d(64,1,1))
        nn.init.normal_(self.layers[-1].weight,std=.001)
        nn.init.constant_(self.layers[-1].bias,-2.1972245773362196)

    def forward(self,evidence,image_hw):
        features=evidence['features']
        if features.ndim!=3 or features.shape[0]!=32:
            raise ValueError('Expected 32-channel stride-four image features')
        h,w=map(int,image_hw)
        if min(h,w)<=0:
            raise ValueError('Positive image dimensions required')
        features=F.layer_norm(features.permute(1,2,0),(32,)).permute(2,0,1)
        logits=self.layers(features[None])[0]
        valid=evidence['spatial_valid']
        return dict(logits=logits.masked_fill(~valid,-torch.inf),grid=evidence['grid'],spatial_valid=valid)


def center_heatmap_targets(output,centers,valid,image_hw,sigma_px=16.):
    if centers.ndim!=2 or centers.shape[-1]!=2 or valid.shape!=centers.shape[:-1]:
        raise ValueError('Expected instance center xy and valid mask')
    if sigma_px<=0 or not torch.isfinite(centers[valid]).all():
        raise ValueError('Finite centers and positive sigma required')
    if ((centers[valid]<0)|(centers[valid]>1)).any():
        raise ValueError('Known centers must be in frame')
    grid=output['grid']; spatial=output['spatial_valid']
    if not spatial.any(): raise ValueError('No real image cells')
    heatmap=output['logits'].new_zeros(spatial.shape)
    positive=torch.zeros_like(spatial)
    scale=centers.new_tensor([image_hw[1],image_hw[0]])
    for xy in centers[valid]:
        distance=((grid-xy)*scale).square().sum(-1).masked_fill(~spatial,torch.inf)
        index=distance.flatten().argmin()
        # Snap the target peak to a real feature cell to give focal loss an
        # exact positive; quantization is <=sqrt(8) pixels for interior centers.
        snapped=grid.reshape(-1,2)[index]
        d=((grid-snapped)*scale).square().sum(-1)
        heatmap=torch.maximum(heatmap,(-.5*d/sigma_px**2).exp().masked_fill(~spatial,0.))
        positive.flatten()[index]=True
    heatmap=heatmap.masked_fill(positive,1.)
    return dict(heatmap=heatmap,positive=positive,
                collisions=int(valid.sum())-int(positive.sum()))


def dense_center_loss(output,centers,valid,image_hw,*,annotation_complete=True):
    target=center_heatmap_targets(output,centers,valid,image_hw)
    spatial=output['spatial_valid']; positive=target['positive']
    logits=output['logits'][0].masked_fill(~spatial,0.)
    prob=logits.sigmoid()
    positive_loss=(-(1-prob).square()*F.logsigmoid(logits))[positive].sum()
    if annotation_complete:
        negative=(~positive)&spatial
        weight=(1-target['heatmap']).pow(4)
        negative_loss=(-prob.square()*F.logsigmoid(-logits)*weight)[negative].sum()
    else:
        negative_loss=logits.sum()*0
    return (positive_loss+negative_loss)/positive.sum().clamp_min(1)


def dense_center_gaussian_loss(output,centers,valid,image_hw,*,annotation_complete=True):
    """Experimental Gaussian-field regression, not a paper reproduction.

    Inspired by HigherHRNet/MMPose heatmap squared-error supervision. Unlike
    that implementation, retain our sigmoid and normalize by target energy
    to avoid changing loss scale solely with image area. This normalization
    is part of the tested objective, not a calibrated detection probability.
    Incomplete annotation is deliberately unsupported: unlabeled pixels must
    not become negative training examples.
    """
    if not annotation_complete:
        raise ValueError('Gaussian-field regression requires complete instance annotation')
    target=center_heatmap_targets(output,centers,valid,image_hw)['heatmap']
    spatial=output['spatial_valid']
    prediction=output['logits'][0].masked_fill(~spatial,0.).sigmoid()
    return (prediction-target).square()[spatial].sum()/target.square()[spatial].sum().clamp_min(1.)


@torch.no_grad()
def decode_centers(output,image_hw,capacity=2,threshold=.5,radius_px=8.):
    """At most capacity distinct image peaks; fixed threshold, never GT count."""
    if not 0<threshold<1: raise ValueError('Threshold must be in (0,1)')
    candidates=regional_top_peaks(output,image_hw,count=capacity,radius_px=radius_px)
    xy=candidates['xy'][0].to(output['logits'].device)
    available=candidates['available'][0].to(xy.device)
    h,w=output['spatial_valid'].shape
    ix=(xy[:,0]*image_hw[1]/4-.5).round().long().clamp(0,w-1)
    iy=(xy[:,1]*image_hw[0]/4-.5).round().long().clamp(0,h-1)
    # Unlike point spatial-softmax scores, this is a sigmoid heatmap score;
    # it has NOT been calibrated as an instance-existence probability.
    logits=output['logits'][0,iy,ix].masked_fill(~available,-torch.inf)
    probability=logits.sigmoid()
    return dict(centers=xy,object_logits=logits,available=available,
                object_probability=probability,instance_active=available&(probability>=threshold))
