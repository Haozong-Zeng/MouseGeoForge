import importlib.util,json,hashlib,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('report',ROOT/'tools/report_calibrated_mouse_evidence_status_20261025.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class EvidenceContractTests(unittest.TestCase):
 def test_legacy_metrics_never_substitute_for_native(self):
  groups={k:dict(images=n,native_original_iou=.86,legacy_roundtrip_iou=.99,native_original_boundary_f1_tol1=.4) for k,n in [('all',879),('real/True',71),('real/False',376),('synthetic/True',144),('synthetic/False',288)]}
  data={'denominator':879,'arms':{a:dict(checkpoint_sha256=a,rows=[{'id':str(i)} for i in range(879)],groups=groups) for a in ['L','M']}}
  value=m.original_native(data);self.assertEqual(value['arms']['M']['groups']['all']['native_original_iou'],.86);self.assertNotIn('legacy_roundtrip_iou',value['arms']['M']['groups']['all'])
 def test_missing_sentinel_and_overlap_denominators_stay_separate(self):
  views=[dict(rmse_mm_on_overlap=2 if i else None,rmse_mm_missing_as_zero=75) for i in range(3)]
  d=dict(oracle_mask_inference=False,calibration_fixed=True,gt_scale_alignment=False,independent_real_mm_accuracy_claim=False,multiple_mouse_instance_detection_claim=False,scenes=[{'views':views}],full_scene_failure_denominator=1,summary={})
  value=m.full_chain(d,1);self.assertEqual(value['nonempty_overlap_views'],2);self.assertEqual(value['mean_view_overlap_RMSE_mm'],2);self.assertEqual(value['mean_view_missing_sentinel_zero_RMSE_mm'],75)
  d['gt_scale_alignment']=True
  with self.assertRaises(AssertionError):m.full_chain(d,1)
 def test_no_averaging_only_successful_seed_subset(self):
  with self.assertRaises(AssertionError):m.matched_roi({'pairs':{'17':{},'29':{}}})
 def test_no_pair_claim_when_one_seed_incomparable(self):
  with self.assertRaises(AssertionError):m.matched_roi({'pairs':{s:dict(comparable=False) for s in ['17','29','43']}})
 def test_partial_surface_audit_cannot_be_complete(self):
  with self.assertRaises(AssertionError):m.audited_pipelines({'pipelines':{'only_one':{}}},6,'views',432,'surface')
 def test_gate_failure_and_partial_encoder_pair_are_distinct(self):
  self.assertEqual(m.adapter_status({'head_converged':False},{},{}),'not_admitted_TRAIN_gate')
  self.assertEqual(m.adapter_status({'head_converged':True,'F_U_completed':False},{},{'F':{}}),'partial_pair_do_not_claim_comparable')
  self.assertEqual(m.adapter_status({'head_converged':False},{},{'F':{}}),'protocol_contradiction_F_U_without_gate')
 def test_snapshot_hash_matches_the_exact_parsed_bytes(self):
  with tempfile.TemporaryDirectory(dir=ROOT/'output') as directory:
   p=Path(directory)/'snapshot.json';raw=b'{"complete": false}\n';p.write_bytes(raw);snapshot=m.Snapshot();value=snapshot.read(p);p.write_text('{"complete": true}')
   self.assertFalse(value['complete']);self.assertEqual(snapshot.inputs[str(p.relative_to(ROOT))]['sha256'],hashlib.sha256(raw).hexdigest())
if __name__=='__main__':unittest.main()
