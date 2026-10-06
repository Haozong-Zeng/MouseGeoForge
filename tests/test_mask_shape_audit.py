import unittest
import numpy as np
from mousegeoforge.mask_shape_audit import mask_shape_statistics


class MaskShapeAuditTests(unittest.TestCase):
    def test_remote_pixel_changes_box_not_main_mass(self):
        a=np.zeros((10,10),bool);a[2:5,2:5]=True;a[9,9]=True
        r=mask_shape_statistics(a)
        self.assertEqual(r['components'],2);self.assertEqual(r['area'],10)
        self.assertEqual(r['largest_component_fraction'],.9);self.assertTrue(r['touches_boundary'])
        self.assertAlmostEqual(r['largest_to_full_box_area'],9/64)
    def test_diagonal_is_connected_and_input_unchanged(self):
        a=np.eye(3,dtype=bool);before=a.copy();r=mask_shape_statistics(a)
        self.assertEqual(r['components'],1);np.testing.assert_array_equal(a,before)
    def test_empty_and_rectangle(self):
        self.assertEqual(mask_shape_statistics(np.zeros((3,4)))['area'],0)
        r=mask_shape_statistics(np.ones((3,4)))
        self.assertEqual(r['fill_fraction'],1);self.assertEqual(r['full_box_cxcywh'],[.5,.5,1.,1.])


if __name__=='__main__':unittest.main()
