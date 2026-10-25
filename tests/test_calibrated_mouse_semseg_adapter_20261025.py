import importlib.util,sys,unittest,copy,math
from pathlib import Path
import torch
from mousegeoforge.models.calibrated_mouse_semseg_adapter_20261025 import MouseSemsegFeatureAdapter,configure,optimizer
from mousegeoforge.semseg_training import make_encoder,seed_all
from mousegeoforge.semseg_adapter_safety_20261025 import grouped,violations,persistent_stop
p=Path(__file__).resolve().parents[1]/'tools/train_calibrated_mouse_semseg_adapter_20261025.py';spec=importlib.util.spec_from_file_location('adapter_engine',p);e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
torch.set_num_threads(2)
class Checks(unittest.TestCase):
 def test_actual_group_LRs(self):
  for mode in ['control','adapter']:
   seed_all(17);m=MouseSemsegFeatureAdapter(make_encoder(fixture=True),dim=32,queries=4,heads=4,pixel_layers=1,query_layers=1);m.adapter_mode=mode;configure(m,'M',0,2000);o=optimizer(m,'M',3e-6)
   for step in [0,99,100,299,300,1499,2999]:
    e.set_roi_learning_rates(o,step,3000)
    for group in o.param_groups:self.assertAlmostEqual(group['lr'],group['peak_lr']*min(1,(step+1)/300)*(.1+.9*.5*(1+math.cos(math.pi*step/3000))),places=20)
 def test_initial_identity_and_frozen_backbone(self):
  seed_all(17);m=MouseSemsegFeatureAdapter(make_encoder(fixture=True),dim=32,queries=4,heads=4,pixel_layers=1,query_layers=1);m.adapter_mode='adapter';configure(m,'M',0,2000);x=torch.rand(1,3,64,64)
  with torch.no_grad():
   normal=(x-m.image_mean)/m.image_std;direct=m.encoder.get_intermediate_layers(normal,n=[20,21,22,23],norm=True,reshape=True);actual=m.features(x,'M')
  self.assertTrue(all(torch.equal(a,b) for a,b in zip(actual,direct)));self.assertTrue(all(not p.requires_grad for p in m.encoder.parameters()))
 def test_parameter_count(self):
  from torch import nn
  class Encoder(nn.Module):
   embed_dim=1024
  m=MouseSemsegFeatureAdapter(Encoder());self.assertEqual(sum(p.numel() for p in m.adapters.parameters()),262144)
 def test_persistent_safety_direction_and_denominators(self):
  rows=[dict(id=str(i),domain=d,bottom=b,loss=.02,native=dict(iou=.9,dice=.95,boundary_f1=.8)) for j,(d,b) in enumerate([('real',True),('real',False),('synthetic',True),('synthetic',False)]) for i in range(j*64,(j+1)*64)];self.assertEqual(violations(rows,rows),[])
  bad=copy.deepcopy(rows)
  for r in bad[:64]:r['native']['iou']=.88
  v=violations(bad,rows);self.assertIn('real/True/iou',v);self.assertTrue(persistent_stop(v,v));self.assertFalse(persistent_stop(v,[]))
  improved=copy.deepcopy(rows)
  for r in improved:r['native']['iou']=.92
  self.assertEqual(violations(improved,rows),[])
  with self.assertRaises(AssertionError):grouped(rows[:-1])
if __name__=='__main__':unittest.main()
