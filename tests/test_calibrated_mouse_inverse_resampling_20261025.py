import importlib.util,unittest
from pathlib import Path
import numpy as np
s=importlib.util.spec_from_file_location('audit',Path(__file__).resolve().parents[1]/'tools/audit_calibrated_mouse_inverse_resampling_20261025.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class OperatorTests(unittest.TestCase):
 def test_strict_half_and_float16_ties(self):
  p=np.array([[.4999,.5,.5005]],dtype=np.float16)
  np.testing.assert_array_equal(m.threshold_saved_probability(p),[[0,0,1]])
 def test_invalid_probability_rejected(self):
  for p in [np.array([[np.nan]]),np.array([[1.1]]),np.array([[-.01]])]:
   with self.assertRaises(AssertionError):m.threshold_saved_probability(p)
 def test_roundtrip_identity_at_native_resolution(self):
  a=np.array([[0,1,0],[1,255,0],[1,1,1]],np.uint8)
  np.testing.assert_array_equal(m.target_roundtrip(a,3),a)
 def test_subsampling_can_remove_thin_structure(self):
  a=np.zeros((8,8),np.uint8);a[:,1]=1
  b=m.target_roundtrip(a,4)
  self.assertEqual(int(b.sum()),0)
  self.assertLess(m.mask_metrics(b,a)['iou'],1)
if __name__=='__main__':unittest.main()
