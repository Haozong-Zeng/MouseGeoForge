"""Opt-in feature-only association metric; no pose outputs or identity inputs."""
import torch
from torch import nn
from torch.nn import functional as F


class AssociationProjection(nn.Module):
    def __init__(self,input_dim=1024,output_dim=64):
        super().__init__();self.projection=nn.Linear(input_dim,output_dim,bias=False)
        nn.init.orthogonal_(self.projection.weight)

    def forward(self,features,*,enabled=False):
        if not enabled:return features
        if features.shape[-1]!=self.projection.in_features or not torch.isfinite(features).all():raise ValueError('Invalid descriptors')
        return F.normalize(self.projection(F.normalize(features.float(),dim=-1)),dim=-1)


def scene_contrastive_loss(embeddings,identities,physical_views,temperature=.1):
    """Labels enter loss only; unknown=-1 is neither positive nor negative.

    Positives are same identity across distinct cameras in this scene; negatives
    are different known identities. No across-scene identity assumption.
    """
    if embeddings.ndim!=3 or identities.shape!=embeddings.shape[:2] or physical_views.shape!=identities.shape or temperature<=0:raise ValueError('Invalid scene supervision')
    z=F.normalize(embeddings,dim=-1);logits=z@z.transpose(-1,-2)/temperature
    known=identities>=0;pair=known[:,:,None]&known[:,None,:]
    same=identities[:,:,None]==identities[:,None,:];different_view=physical_views[:,:,None]!=physical_views[:,None,:]
    positive=pair&same&different_view;negative=pair&~same;candidate=positive|negative
    active=positive.any(-1)&negative.any(-1)
    logdenom=logits.masked_fill(~candidate,-1e9).logsumexp(-1)
    per_anchor=-((logits-logdenom[:,:,None])*positive).sum(-1)/positive.sum(-1).clamp_min(1)
    scene_valid=active.any(-1);per_scene=(per_anchor*active).sum(-1)/active.sum(-1).clamp_min(1)
    loss=per_scene[scene_valid].mean() if scene_valid.any() else embeddings.sum()*0
    return loss,dict(active_anchors=active.sum(),active_scenes=scene_valid.sum())
