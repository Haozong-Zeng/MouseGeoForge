"""Spatial-neck v2: freeze the unidentifiable additive softmax score bias in both arms."""
from mousegeoforge.models.calibrated_mouse_depth_spatial_20261025 import CalibratedRayDepthSpatial
class CalibratedRayDepthSpatialV2(CalibratedRayDepthSpatial):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  # One global additive scalar cancels in plane softmax; preserve its seeded value.
  self.score[4].bias.requires_grad_(False)
