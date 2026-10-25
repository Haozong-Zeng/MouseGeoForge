import importlib.util
from pathlib import Path
import math
import unittest
import torch
P=Path(__file__).resolve().parents[1]/'tools/train_mouse_semseg_roi_20261025_v2.py'
spec=importlib.util.spec_from_file_location('roi_v2',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class LRSchedule(unittest.TestCase):
 def test_actual_registered_six_thousand_schedule(self):
  o=torch.optim.AdamW([dict(params=[torch.nn.Parameter(torch.ones(1))],owner='head',peak_lr=1e-5)],lr=1e-5)
  for step in [0,99,100,475,599,600,2999,5999]:
   m.set_roi_learning_rates(o,step,6000);expected=1e-5*(.1+.9*.5*(1+math.cos(math.pi*step/6000)))*min(1.,(step+1)/600)
   self.assertAlmostEqual(o.param_groups[0]['lr'],expected,places=18)
  self.assertEqual(m.ROI_WARMUP,600);self.assertEqual(m.ROI_FLOOR,.1)
 def test_smoke_schedule_uses_same_floor(self):
  o=torch.optim.AdamW([dict(params=[torch.nn.Parameter(torch.ones(1))],owner='head',peak_lr=1e-5)],lr=1e-5)
  for step in range(4):
   m.set_roi_learning_rates(o,step,4,True);expected=1e-5*(.1+.9*.5*(1+math.cos(math.pi*step/4)))*min(1.,(step+1)/2)
   self.assertAlmostEqual(o.param_groups[0]['lr'],expected,places=18)
if __name__=='__main__':unittest.main()
