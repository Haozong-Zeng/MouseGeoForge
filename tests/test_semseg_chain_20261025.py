import importlib.util,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'));sys.path.insert(0,str(ROOT))
from evaluate_calibrated_mouse_semseg_chain_20261025 import compare_common_depth,verify_new_masks
import evaluate_calibrated_mouse_semseg_chain_20261025 as engine
from evaluate_calibrated_mouse_spatial_pipeline_20261025 import cloud_metrics
from mousegeoforge.semseg_training import digest
class Checks(unittest.TestCase):
 def test_common_pixels_missing_remain_missing(self):
  a=np.array([[2.,0.],[8.,0.]],np.float32);b=np.array([[2.,5.],[0.,0.]],np.float32);m=np.ones((2,2),bool)
  r=compare_common_depth(a,m,b,m);self.assertEqual(r['common_valid_pixels'],1);self.assertEqual(r['maximum_abs_difference_mm'],0)
 def test_common_replay_rejects_changed_depth(self):
  with self.assertRaises(AssertionError):compare_common_depth(np.array([[10.01]]),np.ones((1,1),bool),np.array([[10.]]),np.ones((1,1),bool))
 def test_full_gt_recall_includes_missing_points(self):
  r=cloud_metrics(np.array([[0.,0.,0.]]),np.array([[0.,0.,0.],[50.,0.,0.]]));self.assertEqual(r['recall']['5'],.5);self.assertEqual(r['precision']['5'],1)
 def test_empty_prediction_failure_retained(self):
  r=cloud_metrics(np.empty((0,3)),np.array([[0.,0.,0.]]));self.assertTrue(r['failed']);self.assertEqual(r['fscore']['5'],0)
 def test_lineage_requires_correct_final_crop_and_checkpoint(self):
  with tempfile.TemporaryDirectory() as d:
   base=Path(d);run=base/'semseg_roi_pair_v2/full_006000';run.mkdir(parents=True);directory=base/'semseg_roi_pair_v2/full_dev';directory.mkdir();ck=run/'step_006000.pt';ck.write_bytes(b'fixture final checkpoint');h=digest(ck)
   protocol=dict(crop_mode='full',steps=6000,label_policy='center_exact',precision='fp32',roi_warmup=600,roi_cosine_floor=.1,head_phase_exposure_offset=18000)
   def write(p,x):p.write_text(json.dumps(x))
   write(run/'protocol.json',protocol);write(directory/'evaluation.json',dict(checkpoint_sha256=h));write(base/'semseg_roi_native_full_v2.json',dict(denominator=879,checkpoint_sha256=h,evaluation_receipt_sha256=digest(directory/'evaluation.json')));write(base/'semseg_roi_pair_report_v2.json',dict(comparable=True,campaign=dict(full_paired_endpoint=6000),arms=dict(full=dict(checkpoint_sha256=h))))
   with patch.object(engine,'NEW',base):
    self.assertEqual(verify_new_masks('full',directory)['checkpoint_sha256'],h)
    protocol['crop_mode']='roi';write(run/'protocol.json',protocol)
    with self.assertRaises(AssertionError):verify_new_masks('full',directory)
    protocol['crop_mode']='full';write(run/'protocol.json',protocol);ck.write_bytes(b'wrong checkpoint')
    with self.assertRaises(AssertionError):verify_new_masks('full',directory)
if __name__=='__main__':unittest.main()
