import sys,pathlib,unittest,numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from audit_calibrated_mouse_surface_20261025 import surface_metrics
class TestNativeSurface(unittest.TestCase):
 def setUp(self):self.z=np.full((8,8),300.);self.b=np.ones((8,8),bool);self.k=np.array([[300.,0,3.5],[0,300.,3.5],[0,0,1.]])
 def test_exact_plane(self):
  s=surface_metrics(self.z,self.b,self.z,self.b,self.k);self.assertEqual(s['gt_eligible_adjacent_pairs'],112);self.assertEqual(s['gt_eligible_normals'],49);self.assertEqual(s['gradient_mae_mm_on_covered_pairs'],0);self.assertAlmostEqual(s['normal_angle_mean_deg_on_covered'],0,places=6)
 def test_missing_column_keeps_GT_denominators(self):
  m=self.b.copy();m[:,-1]=False;s=surface_metrics(self.z,m,self.z,self.b,self.k);self.assertEqual(s['gt_eligible_adjacent_pairs'],112);self.assertEqual(s['covered_adjacent_pairs'],97);self.assertEqual(s['gt_eligible_normals'],49);self.assertEqual(s['covered_normals'],42);self.assertAlmostEqual(s['normal_angle_mean_deg_missing180'],180/7)
 def test_stripes_detected(self):
  p=self.z.copy();p[:,::2]+=10;p[:,1::2]-=10;s=surface_metrics(p,self.b,self.z,self.b,self.k);self.assertGreater(s['gradient_mae_mm_on_covered_pairs'],0);self.assertEqual(s['false_steps_gt_le1_pred_gt5_mm'],56);self.assertGreater(s['normal_angle_mean_deg_on_covered'],10)
 def test_GT_discontinuity_not_scored_as_local_surface(self):
  z=self.z.copy();z[:,4:]+=30;s=surface_metrics(z,self.b,z,self.b,self.k);self.assertEqual(s['gt_eligible_adjacent_pairs'],104);self.assertEqual(s['gt_eligible_normals'],42)
 def test_nonfinite_prediction_is_missing(self):
  p=self.z.copy();p[:,7]=np.nan;s=surface_metrics(p,self.b,self.z,self.b,self.k);self.assertEqual(s['covered_normals'],42);self.assertTrue(np.isfinite(s['normal_angle_mean_deg_missing180']));self.assertEqual(s['covered_adjacent_pairs'],97)
if __name__=='__main__':unittest.main()
