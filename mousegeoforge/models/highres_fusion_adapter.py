"""Isolated copy of a native HRNet fusion unit, with differentiable pre-ReLU tap."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from .highres_instance_readout import HighResolutionInstanceReadout

class _CaptureReLU(nn.Module):
    def __init__(self):super().__init__();self.values=[]
    def forward(self,x):
        if x.shape[1]==32:self.values.append(x)
        return F.relu(x,inplace=False)

class HighresFusionAdapter(nn.Module):
    def __init__(self,native_unit,*,adapt=False):
        super().__init__();self.unit=copy.deepcopy(native_unit);self.unit.fuse_act=_CaptureReLU();self.unit.requires_grad_(adapt);self.unit.eval();self.readout=HighResolutionInstanceReadout();self.adapt=adapt
    def train(self,mode=True):
        super().train(mode);self.unit.eval()  # BN running statistics are never adapted.
        return self
    def features(self,branches):
        tap=self.unit.fuse_act;tap.values.clear()
        # Native timm modifies the input list and may apply in-place activations.
        _=self.unit([v.clone() for v in branches]);assert len(tap.values)==1
        feature=tap.values.pop();return feature[0]
    def forward(self,branches,tokens,boxes,baseline_xy,image_hw,*,enabled=False):
        if not enabled:return dict(xy=baseline_xy)
        return self.readout(self.features(branches),tokens,boxes,baseline_xy,image_hw,enabled=True)

def buffer_digest(module):
    import hashlib
    h=hashlib.sha256()
    for name,value in module.named_buffers():h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
