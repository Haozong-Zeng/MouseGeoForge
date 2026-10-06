"""Diagnostic material-parameter cubic projection, not anatomical correction."""
import torch
from mousegeoforge.surface_curve_supervision import cubic_bspline


def project_curve_four_controls(points):
    """Least-squares cubic Bezier fit with BOTH input endpoints kept exact.

    Assumes input samples have uniform existing material parameter u. This is
    not 2D arclength resampling or a3D reconstruction. Cannot repair bad endpoints
    or guarantee no loops. No GT enters prediction projection.
    """
    if points.ndim<2 or points.shape[-1]!=2 or points.shape[-2]<4:raise ValueError('Expected [...,S>=4,2]')
    if not torch.isfinite(points).all():raise ValueError('Nonfinite curve')
    samples=points.shape[-2]
    eye=torch.eye(4,device=points.device,dtype=points.dtype)[...,None].expand(-1,-1,2)
    basis=cubic_bspline(eye,samples)[:,:,0].T
    endpoints=basis[:,0,None]*points[...,0:1,:]+basis[:,-1,None]*points[...,-1:,:]
    residual=points-endpoints
    interior=torch.einsum('ks,...sd->...kd',torch.linalg.pinv(basis[:,1:3]),residual)
    controls=torch.cat([points[...,0:1,:],interior,points[...,-1:,:]],-2)
    return cubic_bspline(controls,samples)
