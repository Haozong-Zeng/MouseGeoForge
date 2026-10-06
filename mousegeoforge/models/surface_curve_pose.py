"""Experimental optional per-instance surface-curve branch; base model unmodified."""
import torch
from torch import nn
from .depth_pose import DepthPoseModel
from mousegeoforge.surface_curve_supervision import cubic_bspline

class SurfaceCurveHead(nn.Module):
    def __init__(self, hidden_dim, controls=6, samples=33):
        super().__init__();self.controls=controls;self.samples=samples
        self.net=nn.Sequential(nn.LayerNorm(hidden_dim),nn.Linear(hidden_dim,hidden_dim),nn.GELU(),nn.Linear(hidden_dim,controls*2))
        nn.init.zeros_(self.net[-1].weight)
        prior=torch.stack([torch.full((controls,),.5),torch.linspace(.15,.85,controls)],-1)
        with torch.no_grad():self.net[-1].bias.copy_(torch.logit(prior).flatten())

    def forward(self,tokens,boxes):
        relative=self.net(tokens).reshape(*tokens.shape[:-1],self.controls,2).sigmoid()-.5
        # Predicted box provides coordinates only; curve loss cannot resize the
        # detector box to reduce its own error. No GT locations enter the head.
        b=boxes.detach();controls=b[...,:2,None].transpose(-1,-2)+relative*b[...,2:,None].transpose(-1,-2)
        return controls,cubic_bspline(controls,self.samples)

class SurfaceCurveDepthPoseModel(DepthPoseModel):
    def __init__(self,*args,surface_controls=6,**kwargs):
        super().__init__(*args,**kwargs)
        self.surface_curve_head=SurfaceCurveHead(self.instance_pose_head.hidden_dim,surface_controls)
        self.surface_curve_enabled=False

    def _decode(self,images,cached,patch_start,*,coupling=None):
        if not self.surface_curve_enabled:
            return super()._decode(images,cached,patch_start,coupling=coupling)
        tokens=[]
        # Local transaction: observe image-derived instance tokens without
        # changing the upstream head API/checkpoints. Always remove on failure.
        handle=self.instance_pose_head.instance_decoder.register_forward_hook(lambda module,args,out:tokens.append(out))
        try:prediction=super()._decode(images,cached,patch_start,coupling=coupling)
        finally:handle.remove()
        if len(tokens)!=1:raise RuntimeError('Expected one instance decoder invocation')
        b,v=images.shape[:2];features=tokens[0].reshape(b,v,*tokens[0].shape[1:])
        control,curve=self.surface_curve_head(features,prediction['instance_boxes_cxcywh'])
        return {**prediction,'instance_surface_control_xy':control,'instance_surface_curve_xy':curve}
