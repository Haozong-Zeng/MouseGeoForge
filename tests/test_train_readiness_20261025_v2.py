from pathlib import Path
import copy,sys,unittest
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from check_calibrated_mouse_train_readiness_20261025_v2 import assess,compare_repeat,GROUPS,METRICS,frozen_model,parameter_versions
class Checks(unittest.TestCase):
 def setUp(self):
  self.rows=[dict(id=str(i),domain='real' if i<128 else 'synthetic',bottom=i%128<64,group='fixture',native=dict(iou=.9,dice=.95,boundary_f1=.7),loss=.03) for i in range(256)];self.ids=[r['id'] for r in self.rows];self.bands={g:{m:dict(tolerance=.001 if m!='boundary_f1' else .002) for m in METRICS} for g in GROUPS}
 def test_declining_loss_and_improving_quality_pass(self):
  w=[copy.deepcopy(self.rows) for _ in range(4)]
  for step,rows in enumerate(w):
   for r in rows:r['native']['iou']+=.002*step;r['loss']*=1-.2*step
  self.assertTrue(assess(w,self.ids,self.bands)['passed'])
 def test_single_stratum_regression_blocks(self):
  w=[copy.deepcopy(self.rows) for _ in range(4)]
  for r in w[-1][:64]:r['native']['iou']-=.01
  self.assertIn('real/True/iou',assess(w,self.ids,self.bands)['failed'])
 def test_late_peak_drop_blocks_even_net_improvement(self):
  w=[copy.deepcopy(self.rows) for _ in range(4)]
  for r in w[2]:r['native']['iou']+=.02
  self.assertFalse(assess(w,self.ids,self.bands)['passed'])
 def test_nearzero_loss_relative_change_is_not_flatness_gate(self):
  w=[copy.deepcopy(self.rows) for _ in range(4)]
  for step,rows in enumerate(w):
   for r in rows:r['loss']=.0001*(step+1)
  self.assertTrue(assess(w,self.ids,self.bands)['passed'])
 def test_loss_blowup_blocks(self):
  w=[copy.deepcopy(self.rows) for _ in range(4)]
  for r in w[-1]:r['loss']=.15
  self.assertFalse(assess(w,self.ids,self.bands)['passed'])
 def test_missing_window_or_wrong_ids_rejected(self):
  with self.assertRaises(AssertionError):assess([self.rows]*3,self.ids,self.bands)
  r=copy.deepcopy(self.rows);r[0]['id']='wrong'
  with self.assertRaises(AssertionError):assess([self.rows,self.rows,self.rows,r],self.ids,self.bands)
 def test_numeric_repeat_has_separate_strict_tolerance(self):
  self.assertEqual(compare_repeat(self.rows,self.rows,self.ids)['loss_max_abs_difference'],0)
  r=copy.deepcopy(self.rows);r[0]['native']['boundary_f1']+=.0001
  with self.assertRaises(AssertionError):compare_repeat(self.rows,r,self.ids)
 def test_frozen_factory_preserves_version_tracking(self):
  with torch.inference_mode():model=frozen_model(lambda:torch.nn.Linear(2,2))
  before=parameter_versions(model)
  self.assertFalse(model.training);self.assertTrue(all(not p.requires_grad for p in model.parameters()))
  with torch.no_grad():model(torch.ones(1,2))
  self.assertEqual(before,parameter_versions(model))
  with torch.no_grad():model.weight.add_(1)
  self.assertNotEqual(before,parameter_versions(model))
if __name__=='__main__':unittest.main()
