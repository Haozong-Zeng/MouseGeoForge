"""Experimental prediction-only spatial refinement; no GT enters forward."""
import torch
from torch import nn
from torch.nn import functional as F
from mousegeoforge.surface_curve_supervision import cubic_bspline


class SurfaceCurveSpatialRefiner(nn.Module):
    """Shared residual updater with local-vs-global image feature control.

    Sample ON the current curve, not at latent spline control points (which may
    lie outside the body). Frozen predicted base controls are not GT anchors.
    Both modes have identical parameters and initialization.
    """
    def __init__(self, mode='local', iterations=3):
        super().__init__()
        if mode not in ('local','global'):raise ValueError('Expected local/global')
        self.mode=mode;self.iterations=iterations
        self.pose=nn.Sequential(nn.LayerNorm(1024),nn.Linear(1024,128),nn.GELU())
        self.spatial=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,128),nn.GELU())
        self.update=nn.Sequential(nn.Linear(259,128),nn.GELU(),nn.Linear(128,2))
        nn.init.zeros_(self.update[-1].weight);nn.init.zeros_(self.update[-1].bias)
        self.register_buffer('sample_ids',torch.tensor([0,4,11,21,28,32]))
        self.register_buffer('offsets',torch.tensor([[0,0],[.04,0],[-.04,0],[0,.04],[0,-.04]],dtype=torch.float32))

    def forward(self,pose_tokens,memory,base_controls):
        if pose_tokens.ndim!=2 or pose_tokens.shape[-1]!=1024:raise ValueError('Expected [N,1024]')
        if memory.ndim!=4 or memory.shape[:2]!=(len(pose_tokens),256):raise ValueError('Expected [N,256,H,W]')
        if base_controls.shape!=(len(pose_tokens),6,2):raise ValueError('Expected [N,6,2]')
        controls=base_controls.detach()
        context=self.pose(pose_tokens.detach())[:,None].expand(-1,6,-1)
        u=(self.sample_ids.to(controls.dtype)/32)[None,:,None].expand(len(controls),-1,-1)
        for _ in range(self.iterations):
            locations=cubic_bspline(controls,33)[:,self.sample_ids]
            if self.mode=='local':
                grid=locations[:,:,None,:]+self.offsets[None,None]
                sampled=F.grid_sample(memory.detach(),2*grid-1,align_corners=False,padding_mode='border')
                spatial=sampled.mean(-1).transpose(1,2)
            else:
                spatial=memory.detach().mean((-2,-1))[:,None].expand(-1,6,-1)
            inputs=torch.cat([context,self.spatial(spatial),locations,u],dim=-1)
            controls=controls+.05*self.update(inputs).tanh()
        return cubic_bspline(controls,33)
