"""Matched-initialization local-context ablation for the frozen center probe."""
import torch
from torch import nn
from torch.nn import functional as F
from mousegeoforge.dense_center_probe import DenseCenterProbe


class ContextCenterProbe(DenseCenterProbe):
    def __init__(self, channels=256, hidden=64, context_enabled=False):
        super().__init__(channels, hidden)
        self.context_enabled = bool(context_enabled)
        self.context = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        with torch.no_grad():
            self.context.weight.zero_()
            self.context.weight[:, 0, 1, 1] = 1
            self.context.bias.zero_()
        mask = torch.zeros_like(self.context.weight)
        mask[:, :, 1, 1] = 1
        self.register_buffer('center_mask', mask, persistent=False)

    def forward(self, features):
        hidden = F.relu(self.shared(F.normalize(features, dim=1, eps=1e-6)))
        # OFF retains trainable pointwise scale/bias and identical registered
        # parameter count; its 512 off-center weights remain zero and unused.
        weight = self.context.weight if self.context_enabled else self.context.weight * self.center_mask
        hidden = F.relu(F.conv2d(hidden, weight, self.context.bias, padding=1, groups=hidden.shape[1]))
        out = self.output(hidden)
        return dict(heat_logits=out[:, 0], offset=out[:, 1:3].sigmoid(), size=out[:, 3:5].sigmoid())
