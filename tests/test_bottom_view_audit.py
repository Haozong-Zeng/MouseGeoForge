import unittest
import numpy as np
from tools.audit_bottom_view_coverage import orientation_from_extrinsic


class CameraOrientationTests(unittest.TestCase):
    def test_bottom_looks_up(self):
        self.assertEqual(orientation_from_extrinsic(np.eye(4)), (-90., 'bottom'))

    def test_top_looks_down(self):
        self.assertEqual(orientation_from_extrinsic(np.diag([1., -1., -1., 1.])), (90., 'top'))

    def test_side(self):
        e = np.eye(4)
        e[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
        self.assertEqual(orientation_from_extrinsic(e), (0., 'side'))

    def test_translation_does_not_define_view_role(self):
        e = np.eye(4)
        e[:3, 3] = [1000, -2000, 500]
        self.assertEqual(orientation_from_extrinsic(e[:3]), (-90., 'bottom'))

    def test_rejects_scaled_or_reflected_camera(self):
        for e in (np.diag([20., 20., 20., 1.]), np.diag([-1., 1., 1., 1.])):
            with self.assertRaises(ValueError):
                orientation_from_extrinsic(e)


if __name__ == '__main__':
    unittest.main()
