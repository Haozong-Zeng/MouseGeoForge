"""Separate real point and synthetic surface-curve targets sharing image keys."""
from torch import nn
from mousegeoforge.models.back_point_integral import BackPointIntegralProbe
from mousegeoforge.models.surface_curve_integral import SurfaceCurveIntegralProbe


class JointSurfacePointIntegralProbe(nn.Module):
    def __init__(self,point_state=None,curve_state=None):
        super().__init__();self.point=BackPointIntegralProbe('spatial');self.curve=SurfaceCurveIntegralProbe('spatial')
        if point_state is not None:self.point.load_state_dict(point_state)
        if curve_state is not None:self.curve.load_state_dict(curve_state)
        # The real+synthetic trained point branch supplies shared spatial keys.
        # Curve query weights retained, but curve predictions can initially
        # change. Both controls must use this EXACT initialization.
        self.curve.points.key=self.point.key
        self.curve.points.position=self.point.position

    def forward_curve(self,pose,memory,controls):return self.curve(pose,memory,controls)
    def forward_points(self,pose,memory,coarse):return self.point(pose,coarse,memory)
