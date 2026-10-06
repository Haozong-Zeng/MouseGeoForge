import sys,unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from reconstruct_pymousestudio_unified import seam_stats
from add_dannce_real_multiview_pseudo_depth import rasterize_depth


class UnifiedRenderTest(unittest.TestCase):
    def test_seam_ignores_background_and_measures_interface(self):
        before=np.array([[0.,100.,120.],[0.,100.,120.]],np.float32)
        after=np.array([[0.,100.,101.],[0.,100.,101.]],np.float32)
        added=np.array([[False,False,True],[False,False,True]])
        s=seam_stats(before,after,added)
        self.assertEqual(s['pairs'],2)
        self.assertEqual(s['before_p95_mm'],20.)
        self.assertEqual(s['after_p95_mm'],1.)

    def test_no_interface(self):
        d=np.ones((3,3),np.float32)
        self.assertIsNone(seam_stats(d,d,np.zeros_like(d,bool))['after_p95_mm'])

    def test_zbuffer_selects_front_and_is_order_independent(self):
        k=np.array([[100.,0,20],[0,100.,20],[0,0,1.]])
        e=np.eye(4);mask=np.full((40,40),255,np.uint8)
        points=np.array([[0.,0.,200.],[0.,0.,100.]],np.float32)
        a=rasterize_depth(points,mask,k,e,1.)
        b=rasterize_depth(points[::-1],mask,k,e,1.)
        np.testing.assert_array_equal(a,b)
        self.assertAlmostEqual(float(a[20,20]),99.5,places=3)
        self.assertTrue(np.isfinite(a).all())


if __name__=='__main__':unittest.main()
