"""Parameter-count-matched pointwise/spatial residual neck on frozen ROI features."""
import torch
from torch import nn
from torch.nn import functional as F
from mousegeoforge.models.calibrated_mouse_depth_roi import CalibratedRayDepthROI
class CalibratedRayDepthSpatial(CalibratedRayDepthROI):
 def __init__(self,in_channels=1024,channels=32,planes=96,neck_kernel=3):
  if neck_kernel not in (1,3):raise ValueError('Only registered1x1/3x3 controls')
  super().__init__(in_channels,channels,planes)
  self.neck_kernel=neck_kernel
  hidden=channels*(9 if neck_kernel==1 else 1)
  self.neck=nn.Sequential(nn.Conv2d(channels,hidden,neck_kernel,padding=neck_kernel//2,bias=False),nn.GELU(),nn.Conv2d(hidden,channels,neck_kernel,padding=neck_kernel//2,bias=False))
  nn.init.zeros_(self.neck[2].weight)
 def encode(self,features):
  projected=self.project(features.float())
  return F.normalize(projected+self.neck(projected),dim=1)
