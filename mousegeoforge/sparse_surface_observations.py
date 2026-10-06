"""Weak ordered point evidence, deliberately NOT full-curve ground truth."""
import torch
from torch.nn import functional as F
from .surface_curve_supervision import points_to_polyline


def sparse_back_observation_loss(curve, observed_xy, observed_valid, *, scale=768.,
                                 beta=.01, order_weight=.01):
    """Observed manual06/07 can slide freely along a predicted dorsal curve.

    No material fractions, endpoints, reverse Chamfer, or guessed visibility.
    One/two points cannot identify a complete curve: synthetic replay and
    independent diagnostics are required. This loss alone permits unobserved
    extensions and does not reject every possible loop or collapse.
    """
    if curve.ndim!=2 or curve.shape[1]!=2 or len(curve)<3:raise ValueError('Expected [N>=3,2]')
    if observed_xy.shape!=(2,2) or observed_valid.shape!=(2,):raise ValueError('Expected exactly06/07slots and explicit eligibility')
    if scale<=0 or beta<=0 or order_weight<0:raise ValueError('Invalid loss settings')
    if not torch.isfinite(curve).all():raise ValueError('Nonfinite predicted curve')
    good=observed_valid.bool()&torch.isfinite(observed_xy).all(-1)
    if not good.any():return curve.sum()*0
    distance2,parameter=points_to_polyline(observed_xy[good]/scale,curve/scale,torch.ones(len(curve),device=curve.device,dtype=torch.bool))
    # Stable radial Huber without sqrt(0) gradient at an exactly fitting point.
    robust=torch.where(distance2<beta*beta,.5*distance2/beta,torch.sqrt(distance2+1e-12)-.5*beta)
    loss=robust.mean()
    if good.all():loss=loss+order_weight*F.relu(parameter[0]-parameter[1]).square()
    return loss
