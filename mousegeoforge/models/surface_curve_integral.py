"""Six on-curve spatial expectations, interpolated to a cubic material curve."""
import torch
from torch import nn
from mousegeoforge.surface_curve_supervision import cubic_bspline
from mousegeoforge.models.back_point_integral import BackPointIntegralProbe


class SurfaceCurveIntegralProbe(nn.Module):
    def __init__(self,mode='spatial'):
        super().__init__()
        self.points=BackPointIntegralProbe(mode,num_points=6)
        eye=torch.eye(6,dtype=torch.float64)[...,None].expand(-1,-1,2)
        b6=cubic_bspline(eye,6)[:,:,0].T
        b33=cubic_bspline(eye,33)[:,:,0].T
        assert torch.linalg.cond(b6)<20
        self.register_buffer('interpolation',torch.linalg.solve(b6.T,b33.T).T.float())

    def forward(self,pose,memory,base_controls,return_distribution=False):
        if base_controls.shape!=(len(pose),6,2):raise ValueError('Expected predicted spline controls')
        prior=cubic_bspline(base_controls.detach(),6)
        points,prob=self.points(pose,prior,memory,True)
        curve=torch.einsum('sk,nkd->nsd',self.interpolation,points)
        return (curve,points,prob) if return_distribution else curve
