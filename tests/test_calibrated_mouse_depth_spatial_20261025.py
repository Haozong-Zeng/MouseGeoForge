import copy
import importlib.util
import io
import math
from pathlib import Path
import unittest
import sys
import torch
from mousegeoforge.models.calibrated_mouse_depth_roi import CalibratedRayDepthROI,supervised_loss
from mousegeoforge.models.calibrated_mouse_depth_spatial_20261025 import CalibratedRayDepthSpatial
P=Path(__file__).resolve().parents[1]/'tools/train_calibrated_mouse_depth_spatial_20261025.py'
sys.path.insert(0,str(P.parent))
spec=importlib.util.spec_from_file_location('spatialtrainer',P);trainer=importlib.util.module_from_spec(spec);spec.loader.exec_module(trainer)
torch.set_num_threads(2)
def fixture(kernel):
 torch.manual_seed(17);model=CalibratedRayDepthSpatial(in_channels=8,channels=4,planes=8,neck_kernel=kernel)
 torch.manual_seed(29);features=torch.randn(3,8,8,8);k=torch.tensor([[100.,0,192],[0,100,192],[0,0,1]]).repeat(3,1,1);e=torch.eye(4).repeat(3,1,1);e[:,0,3]=torch.tensor([-1.,0,1.]);bounds=torch.tensor([[-10.,-10.,10.],[10.,10.,40.]]);xy=torch.tensor([[192.,192.],[190.,193.],[195.,189.]])
 return model,features,k,e,bounds,xy
class Spatial(unittest.TestCase):
 def test_parameter_count_exact_match(self):
  counts=[sum(p.numel() for p in CalibratedRayDepthSpatial(neck_kernel=k).neck.parameters()) for k in (1,3)];self.assertEqual(counts,[18432,18432])
 def test_base_weights_and_initial_prediction_exact(self):
  a,f,k,e,b,x=fixture(1);c,_,_,_,_,_=fixture(3);torch.manual_seed(17);base=CalibratedRayDepthROI(in_channels=8,channels=4,planes=8)
  for name,v in base.state_dict().items():self.assertTrue(torch.equal(v,a.state_dict()[name]));self.assertTrue(torch.equal(v,c.state_dict()[name]))
  enc=base.encode(f);self.assertTrue(torch.equal(enc,a.encode(f)));self.assertTrue(torch.equal(enc,c.encode(f)))
  z=base(enc,x,1,k,e,b)[0];self.assertTrue(torch.equal(z,a(a.encode(f),x,1,k,e,b)[0]));self.assertTrue(torch.equal(z,c(c.encode(f),x,1,k,e,b)[0]))
 def test_neck_nonzero_gradients_after_identity_start(self):
  for kernel in (1,3):
   m,f,k,e,b,x=fixture(kernel);o=torch.optim.AdamW(m.parameters(),lr=.001)
   for step in range(2):
    o.zero_grad();z,_,valid=m(m.encode(f),x,1,k,e,b);supervised_loss(z,torch.full_like(z,21.),valid).backward();self.assertGreater(float(m.neck[2].weight.grad.norm()),0)
    if step:self.assertGreater(float(m.neck[0].weight.grad.norm()),0)
    o.step()
 def test_spatial_support_differs_with_same_parameter_count(self):
  a,_,_,_,_,_=fixture(1);b,_,_,_,_,_=fixture(3)
  for m in (a,b):
   with torch.no_grad():m.neck[0].weight.fill_(.1);m.neck[2].weight.fill_(.1)
  x=torch.zeros(1,4,7,7);y=x.clone();y[:,:,3,4]=1
  self.assertTrue(torch.equal(a.neck(x)[:,:,3,3],a.neck(y)[:,:,3,3]));self.assertGreater(float((b.neck(y)-b.neck(x))[:,:,3,3].abs().sum()),0)
 def test_CPU_serialization_next_updates_exact(self):
  def update(m,o,ema,f,k,e,b,x,step):
   o.zero_grad();z,_,valid=m(m.encode(f),x,1,k,e,b);supervised_loss(z,torch.full_like(z,21.+step),valid).backward();o.step()
   with torch.no_grad():
    for p,q in zip(m.parameters(),ema.parameters()):q.lerp_(p,.005)
  full,f,k,e,b,x=fixture(3);o=torch.optim.AdamW(full.parameters(),lr=.001);ema=copy.deepcopy(full)
  for step in range(2):update(full,o,ema,f,k,e,b,x,step)
  buf=io.BytesIO();torch.save(dict(model=full.state_dict(),optimizer=o.state_dict(),ema=ema.state_dict()),buf);buf.seek(0);s=torch.load(buf,weights_only=False)
  restored,_,_,_,_,_=fixture(3);ro=torch.optim.AdamW(restored.parameters(),lr=.001);re=copy.deepcopy(restored);restored.load_state_dict(s['model']);ro.load_state_dict(s['optimizer']);re.load_state_dict(s['ema'])
  for step in (2,3):update(full,o,ema,f,k,e,b,x,step);update(restored,ro,re,f,k,e,b,x,step)
  for n,v in full.state_dict().items():self.assertTrue(torch.equal(v,restored.state_dict()[n]),n)
  for n,v in ema.state_dict().items():self.assertTrue(torch.equal(v,re.state_dict()[n]),n)
  for key,value in o.state_dict()['state'].items():
   for name,v in value.items():self.assertTrue(torch.equal(v,ro.state_dict()['state'][key][name]))
 def test_registered_actual_LR(self):
  for step in [0,99,100,2999,5999]:self.assertAlmostEqual(trainer.registered_lr(step,6000),.001*min(1,(step+1)/100)*(.1+.9*.5*(1+math.cos(math.pi*step/6000))),places=18)
if __name__=='__main__':unittest.main()
