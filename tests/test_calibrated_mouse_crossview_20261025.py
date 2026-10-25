import pathlib,sys,unittest,numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from audit_calibrated_mouse_crossview_20261025 import direction,bilinear
class TestCrossview(unittest.TestCase):
 def setUp(self):self.z=np.full((16,16),300.);self.b=np.ones((16,16),bool);self.k=np.array([[60.,0,7.5],[0,60.,7.5],[0,0,1.]]);self.e=np.eye(4);self.s=self.e.copy();self.s[0,3]=-10
 def score(self,pr=None,ps=None,mr=None,gs=None):return direction(self.z,self.b,self.z if gs is None else gs,self.b,self.z if pr is None else pr,self.b if mr is None else mr,self.z if ps is None else ps,self.b,self.k,self.e,self.k,self.s)
 def test_stereo_plane_exact(self):
  x=self.score();self.assertEqual(x['GT_covisible_pixels'],210);self.assertEqual(x['prediction_coverage'],1);self.assertAlmostEqual(x['mean_absolute_consistency_mm_on_covered'],0)
 def test_source_bias_detected(self):
  x=self.score(ps=self.z+10);self.assertAlmostEqual(x['mean_absolute_consistency_mm_on_covered'],10)
 def test_shared_wrong_plane_shows_limit(self):
  x=self.score(pr=self.z+50,ps=self.z+50);self.assertAlmostEqual(x['mean_absolute_consistency_mm_on_covered'],0);self.assertGreater(x['covered_predictions'],0)
 def test_missing_predictions_retain_GT_denominator(self):
  m=self.b.copy();m[:8]=False;x=self.score(mr=m);self.assertEqual(x['GT_covisible_pixels'],210);self.assertLess(x['prediction_coverage'],1)
 def test_GT_occlusion_excluded(self):self.assertEqual(self.score(gs=self.z-50)['GT_covisible_pixels'],0)
 def test_bilinear_requires_all_four_valid(self):
  m=self.b.copy();m[1,1]=False;values,ok=bilinear(self.z,m,np.array([[.5,.5],[2.5,2.5],[-1.,0.]]));self.assertEqual(ok.tolist(),[False,True,False])
if __name__=='__main__':unittest.main()
