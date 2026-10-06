import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from mousegeoforge.models.highres_instance_readout import HighResolutionInstanceReadout,highres_localization_loss,KP

class HighresTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(17);self.m=HighResolutionInstanceReadout();self.f=torch.randn(32,8,8);self.t=torch.randn(6,5,256);self.b=torch.tensor([[.5,.5,.5,.5]]).expand(6,4).clone();self.xy=torch.rand(6,23,2);self.hw=(28,32)
    def run_model(self):return self.m(self.f,self.t,self.b,self.xy,self.hw,enabled=True)
    def test_defaultoff(self):self.assertIs(self.m(None,None,None,self.xy,None)['xy'],self.xy)
    def test_zero_initial_exact(self):self.assertTrue(torch.equal(self.run_model()['xy'],self.xy))
    def test_query_permutation(self):
        torch.nn.init.normal_(self.m.query[-1].weight);p=torch.tensor([5,1,3,2,0,4]);a=self.run_model()['xy'];b=self.m(self.f,self.t[p],self.b[p],self.xy[p],self.hw,enabled=True)['xy'];self.assertTrue(torch.allclose(a[p],b,atol=1e-7))
    def test_non_target_exact(self):
        torch.nn.init.normal_(self.m.query[-1].weight);ids=[k for k in range(23) if k not in KP];self.assertTrue(torch.equal(self.run_model()['xy'][:,ids],self.xy[:,ids]))
    def test_finite_grad_and_padding(self):
        out=self.run_model();known=torch.ones(6,5,dtype=torch.bool);target=torch.rand(6,5,2);loss=highres_localization_loss(out,target,known,self.hw);loss.backward();self.assertTrue(torch.isfinite(loss));self.assertFalse(out['spatial_valid'][-1].any());self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.m.parameters() if p.grad is not None));self.assertGreater(float(self.m.query[-1].weight.grad.norm()),0)
    def test_missing_nan_excluded(self):
        out=self.run_model();target=torch.rand(6,5,2);known=torch.zeros(6,5,dtype=torch.bool);known[0,0]=True;a=highres_localization_loss(out,target,known,self.hw);target[~known]=float('nan');b=highres_localization_loss(out,target,known,self.hw);self.assertEqual(float(a),float(b))
    def test_all_unknown_zero_gradient(self):
        out=self.run_model();loss=highres_localization_loss(out,torch.full((6,5,2),float('nan')),torch.zeros(6,5,dtype=torch.bool),self.hw);loss.backward();self.assertEqual(float(loss),0);self.assertTrue(all(p.grad is None or bool((p.grad==0).all()) for p in self.m.parameters()))
    def test_soft_box_not_hard_boundary(self):
        self.b[:,2:]=.001;out=self.run_model();self.assertTrue(torch.isfinite(out['logits'][...,0,0]).all());self.assertTrue(torch.equal(out['xy'],self.xy))
    def test_zeroappearance_ignores_image_values(self):
        self.m.appearance=False;torch.nn.init.normal_(self.m.query[-1].weight);a=self.run_model()['xy'];self.f*=9;b=self.run_model()['xy'];self.assertTrue(torch.equal(a,b))
if __name__=='__main__':unittest.main()
