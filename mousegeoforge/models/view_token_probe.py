"""Reversible inference-only special-token intervention, not a production fix."""
from contextlib import contextmanager
import torch

@contextmanager
def shared_first_view_tokens(aggregator, *, enabled=False):
    if not enabled:
        yield
        return
    if aggregator.training:
        raise ValueError('Special-token probe is inference-only')
    parameters=[aggregator.camera_token,aggregator.register_token]
    for p in parameters:
        if p.ndim!=4 or p.shape[:2]!=(1,2) or p.requires_grad:
            raise ValueError('Expected frozen [1,2,K,D] special-token parameters')
    if parameters[0].shape[2]!=1 or parameters[0].shape[-1]!=parameters[1].shape[-1]:
        raise ValueError('Unexpected camera/register token geometry')
    saved=[p.detach().clone() for p in parameters]
    try:
        with torch.no_grad():
            for p in parameters:p[:,1:2].copy_(p[:,0:1])
        yield
    finally:
        with torch.no_grad():
            for p,original in zip(parameters,saved):p.copy_(original)
