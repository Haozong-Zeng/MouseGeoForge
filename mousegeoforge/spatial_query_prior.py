"""Experimental soft spatial cross-attention prior, default OFF.

SMCA-inspired Gaussian log-attention bias; not a full SMCA/DAB-DETR reproduction.
No GT input, hard region, inter-query repulsion, image-role assumption or identity.
"""
from contextlib import contextmanager
import math
import torch
from torch import nn
from torch.nn import functional as F


class ConditionalSpatialPrior(nn.Module):
    def __init__(self, layers, queries, dim):
        super().__init__()
        if min(layers, queries, dim) <= 0:
            raise ValueError('Positive geometry required')
        columns = math.ceil(math.sqrt(queries))
        rows = math.ceil(queries / columns)
        xy = torch.tensor([((q % columns + .5) / columns, (q // columns + .5) / rows)
                           for q in range(queries)])
        self.center_logits = nn.Parameter(torch.logit(xy).repeat(layers, 1, 1))
        sigma_fraction = (.5 - .08) / .92
        self.sigma_logits = nn.Parameter(torch.full((layers, queries, 2), math.log(sigma_fraction/(1-sigma_fraction))))
        self.strength_logits = nn.Parameter(torch.full((layers, queries), math.log(math.expm1(.1))))
        # Direct Parameters avoid random initialization and preserve RNG state.
        self.condition_weight = nn.Parameter(torch.zeros(layers, 4, dim))

    def geometry(self, query, layer):
        if query.ndim != 3 or query.shape[1] != self.center_logits.shape[1] or query.shape[2] != self.condition_weight.shape[2]:
            raise ValueError('Expected batch-first [frames, queries, dim]')
        delta = F.linear(F.normalize(query, dim=-1, eps=1e-6), self.condition_weight[layer].to(query))
        center = (self.center_logits[layer].to(query)[None] + delta[..., :2]).sigmoid()
        sigma = .08 + .92 * (self.sigma_logits[layer].to(query)[None] + delta[..., 2:]).sigmoid()
        strength = F.softplus(self.strength_logits[layer].to(query)).clamp(max=2.)
        return center, sigma, strength

    def forward(self, query, layer, grid_hw, heads):
        height, width = grid_hw
        if min(height, width, heads) <= 0:
            raise ValueError('Invalid spatial grid or head count')
        center, sigma, strength = self.geometry(query, layer)
        y = (torch.arange(height, device=query.device, dtype=query.dtype) + .5) / height
        x = (torch.arange(width, device=query.device, dtype=query.dtype) + .5) / width
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        grid = torch.stack((xx, yy), dim=-1).reshape(1, 1, height*width, 2)
        squared = ((grid-center[:, :, None]) / sigma[:, :, None]).square().sum(-1)
        bias = (-.5 * strength[None, :, None] * squared).clamp(min=-20.)
        # nn.MultiheadAttention expects [batch * heads, queries, memory].
        return bias[:, None].expand(-1, heads, -1, -1).reshape(query.shape[0]*heads, query.shape[1], height*width)


def install_spatial_query_prior(model):
    head = model.instance_pose_head if hasattr(model, 'instance_pose_head') else model
    if hasattr(head, 'spatial_query_prior'):
        raise ValueError('Already installed')
    reference = next(head.parameters())
    head.spatial_query_prior = ConditionalSpatialPrior(len(head.instance_decoder.layers), head.num_queries, head.hidden_dim).to(reference)
    return model


@contextmanager
def spatial_query_context(model=None, *, enabled=False):
    if not enabled:
        yield
        return
    head = model.instance_pose_head if hasattr(model, 'instance_pose_head') else model
    if not hasattr(head, 'spatial_query_prior'):
        raise ValueError('Install spatial prior first')
    if getattr(head, '_spatial_query_active', False):
        raise RuntimeError('Non-reentrant spatial context')
    head._spatial_query_active = True
    geometry = {}; handles = []
    try:
        def capture(module, args, kwargs):
            images = args[1] if len(args) > 1 else kwargs.get('images')
            if images is None or images.ndim != 5:
                raise ValueError('Need image tensor for exact patch geometry')
            geometry['hw'] = (images.shape[-2] // head.patch_size, images.shape[-1] // head.patch_size)
            geometry['frames'] = images.shape[0] * images.shape[1]
        handles.append(head.register_forward_pre_hook(capture, with_kwargs=True))
        for layer_index, layer in enumerate(head.instance_decoder.layers):
            def inject(module, args, kwargs, index=layer_index):
                if len(args) > 3:
                    raise ValueError('Only query/key/value positional arguments supported')
                query = args[0] if args else kwargs['query']
                key = args[1] if len(args) > 1 else kwargs['key']
                if not geometry or query.shape[0] != geometry['frames'] or key.shape[1] != math.prod(geometry['hw']):
                    raise ValueError('Memory/image geometry mismatch')
                if kwargs.get('attn_mask') is not None or kwargs.get('is_causal', False):
                    raise ValueError('Existing attention/causal mask is unsupported, never silently overwrite')
                bias = head.spatial_query_prior(query, index, geometry['hw'], module.num_heads)
                return args, dict(kwargs, attn_mask=bias)
            handles.append(layer.multihead_attn.register_forward_pre_hook(inject, with_kwargs=True))
        yield
    finally:
        for handle in handles:
            handle.remove()
        head._spatial_query_active = False
