import importlib.util
from pathlib import Path
import unittest
import numpy as np
P=Path(__file__).resolve().parents[1]/'tools/audit_calibrated_mouse_confidence_filter_20261025_v2.py'
spec=importlib.util.spec_from_file_location('filter_audit',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class FilterAuditTest(unittest.TestCase):
 def test_percentile_is_rank_not_probability(self):
  c=np.array([2.,3.,4.,5.,6.]);v=np.ones(5,dtype=bool)
  keep,t=m.percentile_mask(c,v,20);self.assertAlmostEqual(t,2.8);self.assertEqual(keep.tolist(),[False,True,True,True,True])
 def test_ties_are_retained(self):
  keep,_=m.percentile_mask(np.ones(9),np.ones(9,dtype=bool),20);self.assertTrue(keep.all())
 def test_zero_background_edge_effect(self):
  z=np.full((1,5,5),300.);z[0,0,:]=0
  edge=m.upstream_edge()(z);self.assertTrue(edge[0,1,2]);self.assertFalse(edge[0,3,2])
 def test_relative_threshold_uses_depth_units(self):
  a=np.full((1,5,5),300.);a[0,2,2]=312.
  np.testing.assert_array_equal(m.upstream_edge()(a),m.upstream_edge()(a/1000))
 def test_random_count_and_determinism(self):
  z=np.full((3,5,5),300.);c=np.arange(1,76).reshape(3,5,5)/100;v=np.ones_like(z,dtype=bool);rgb=np.full((3,5,5,3),128,dtype=np.uint8)
  a,_,_=m.filters(z,c,v,rgb,4);b,_,_=m.filters(z,c,v,rgb,4)
  self.assertEqual(a['confidence20'].sum(),a['random_same_count_as_confidence20'].sum());np.testing.assert_array_equal(a['random_same_count_as_confidence20'],b['random_same_count_as_confidence20'])
 def test_complete_truth_recall_and_empty_failure(self):
  truth=np.array([[0.,0,0],[10.,0,0]])
  r=m.cloud_stats(truth[:1],truth,np.array([0.]));self.assertEqual(r['precision']['1'],1.);self.assertEqual(r['recall']['1'],.5);self.assertAlmostEqual(r['fscore']['1'],2/3)
  empty=m.cloud_stats(truth[:0],truth,np.array([]));self.assertEqual(empty['fscore']['5'],0.)
if __name__=='__main__':unittest.main()
