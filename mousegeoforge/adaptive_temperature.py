"""Optional experimental conditional local-logit scale; upstream source untouched."""
import math
from types import MethodType
import torch
from torch import nn
from torch.nn import functional as F
from vggt_omega.models.heads.instance_pose_head import MultiInstanceMouseHead

PREFIX = 'instance_pose_head.adaptive_temperature.'


class ConditionalTemperature(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # No random draws: installing this branch must not perturb experiment RNG.
        self.weight = nn.Parameter(torch.zeros(1, dim))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, tokens):
        log_scale = F.linear(F.normalize(tokens, dim=-1, eps=1e-6), self.weight, self.bias)
        return log_scale.clamp(math.log(.5), math.log(8.)).exp().squeeze(-1)


def adaptive_local_refinement(self, pixels, tokens, coarse):
    logits, original = MultiInstanceMouseHead._local_refinement(self, pixels, tokens, coarse)
    if not self.adaptive_temperature_enabled:
        self.last_temperature_scale = torch.ones_like(coarse[...,0]).detach()
        return logits, original
    scale = self.adaptive_temperature(tokens)
    self.last_temperature_scale = scale.detach()
    logits = logits * scale[..., None, None]
    axis = torch.linspace(-self.local_window_radius, self.local_window_radius, self.local_heatmap_size,
                          dtype=coarse.dtype, device=coarse.device)
    y, x = torch.meshgrid(axis, axis, indexing='ij')
    offsets = torch.stack((x, y), dim=-1)
    probabilities = logits.flatten(-2).softmax(dim=-1).view_as(logits)
    delta = (probabilities[...,None] * offsets).sum(dim=(-3,-2))
    return logits, (coarse + delta).clamp(0.,1.)


def install_adaptive_temperature(model):
    head = model.instance_pose_head if hasattr(model, 'instance_pose_head') else model
    if hasattr(head, 'adaptive_temperature'):
        raise ValueError('Temperature branch already installed')
    reference = next(head.parameters())
    head.adaptive_temperature = ConditionalTemperature(head.hidden_dim).to(reference)
    head.adaptive_temperature_enabled = True
    head._local_refinement = MethodType(adaptive_local_refinement, head)
    return model


def restore_adaptive(model, state):
    """Original state implies exact zero branch; extended state must be complete."""
    current = model.state_dict()
    extras = {k for k in current if k.startswith(PREFIX)}
    if len(extras) != 2:
        raise ValueError('Expected installed temperature branch')
    provided = extras & set(state)
    if provided and provided != extras:
        raise ValueError('Partial temperature state')
    merged = dict(state)
    if not provided:
        merged.update({k: torch.zeros_like(current[k]) for k in extras})
    loaded = model.load_state_dict(merged, strict=False)
    if loaded.unexpected_keys or any(not k.startswith('aggregator.') for k in loaded.missing_keys):
        raise ValueError(f'Invalid checkpoint: {loaded}')


def original_state_projection(state):
    return {k:v for k,v in state.items() if not k.startswith(PREFIX)}
