from pathlib import Path
import copy,hashlib,json,sys,tempfile,unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from report_calibrated_mouse_next12h_evidence_20261025 import components,read_snapshot,worker_snapshot
class Checks(unittest.TestCase):
 def test_partial_spatial_never_averages_favorable_seeds(self):
  self.assertIsNone(components(dict(spatial=dict(all_three_pairs_comparable=False,equal_seed_mean=None,comparable_complete_seeds=[17])))['spatial_depth']['equal_seed_mean'])
  with self.assertRaises(AssertionError):components(dict(spatial=dict(all_three_pairs_comparable=True,equal_seed_mean={},comparable_complete_seeds=[17,29],arms={})))
 def test_readiness_pass_is_not_convergence_or_unfreeze(self):
  r=dict(historical_gate_failures_unchanged=True,no_DEV_reads=True,arms={a:dict(status='TRAIN_window_passed') for a in ['full','roi']},FULL_numerical_repeat=dict(status='passed'),FULL_ready_for_trial_design=True,adaptation_job_registered=False)
  x=components(dict(readiness=r))['TRAIN_readiness'];self.assertFalse(x['convergence_claim']);self.assertFalse(x['automatic_unfreeze'])
 def test_partial_chain_no_complete_claim(self):
  x=components(dict(chain=dict(all_three_complete=False,paired_full_roi_complete=False,arms={'anchor':dict(status='complete')})))['mask_replacement_chain'];self.assertEqual(x['status'],'missing_or_partial');self.assertTrue(x['missing_zero_is_sentinel_not_valid_point_error'])
 def test_segmentation_requires879_native_final_receipts(self):
  r=dict(comparable=True,campaign=dict(status='completed',full_paired_endpoint=6000),arms={a:dict(status='complete',groups=dict(all=dict(images=878))) for a in ['full','roi']},roi_minus_full={})
  with self.assertRaises(AssertionError):components(dict(semseg=r))
 def test_train_test_overlap_never_heldout_accuracy(self):
  r=dict(loader_path_replay=dict(train_count=480,test_count=18,train_test_overlap=18,disjoint_test_count=0),stored_render_scores=dict(heldout_accuracy_evidence=False),current_source_path=dict(exists=False));self.assertFalse(components(dict(dannce=r))['DANNCE_GS']['independent_heldout_geometry_proven'])
  r['stored_render_scores']['heldout_accuracy_evidence']=True
  with self.assertRaises(AssertionError):components(dict(dannce=r))
 def test_exact_parsed_JSON_bytes_are_hashed(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'data.json';raw=b'{ "b":2, "a":1 }\n';p.write_bytes(raw);data,receipt=read_snapshot(p);self.assertEqual(data,dict(a=1,b=2));self.assertEqual(receipt['sha256'],hashlib.sha256(raw).hexdigest())
 def test_empty_identity_cannot_match(self):self.assertFalse(worker_snapshot({})['matches'])
if __name__=='__main__':unittest.main()
