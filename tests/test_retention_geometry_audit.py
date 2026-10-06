import unittest
import torch
from mousegeoforge.retention_geometry_audit import symmetric_epipolar_distance


class RetentionGeometryTests(unittest.TestCase):
    def cameras(self):
        return torch.tensor([[[100.,0,0,0],[0,100.,0,0],[0,0,1.,0]],
                             [[100.,0,0,-100.],[0,100.,0,0],[0,0,1.,0]]])

    def test_consistent_horizontal_epipolar_lines(self):
        self.assertLess(symmetric_epipolar_distance(torch.tensor([[20.,10.],[0.,10.]]),self.cameras()),1e-5)

    def test_vertical_mismatch_measured_in_pixels(self):
        self.assertAlmostEqual(symmetric_epipolar_distance(torch.tensor([[20.,10.],[0.,30.]]),self.cameras()),20.,places=5)

    def test_camera_projective_scale_invariance(self):
        p=self.cameras();p[0]*=7;p[1]*=.1
        self.assertAlmostEqual(symmetric_epipolar_distance(torch.tensor([[20.,10.],[0.,30.]]),p),20.,places=5)

    def test_reject_wrong_view_count(self):
        with self.assertRaises(ValueError):symmetric_epipolar_distance(torch.zeros(3,2),torch.zeros(3,3,4))


if __name__=='__main__':unittest.main()
