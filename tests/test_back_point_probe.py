import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import torch
from probe_uncertain_back_points_20261003 import BackPointProbe


class BackPointProbeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17);torch.set_num_threads(2)
        self.pose=torch.randn(3,1024,requires_grad=True);self.coarse=torch.rand(3,2,2,requires_grad=True)
    def test_zero_init_exact(self):
        model=BackPointProbe();self.assertTrue(torch.equal(model(self.pose,self.coarse),self.coarse))
    def test_frozen_inputs_gradient(self):
        model=BackPointProbe();model(self.pose,self.coarse).square().mean().backward()
        self.assertIsNone(self.pose.grad);self.assertIsNone(self.coarse.grad);self.assertGreater(float(model.delta[-1].weight.grad.abs().sum()),0)
    def test_permutation_and_single_instance(self):
        model=BackPointProbe();torch.nn.init.normal_(model.delta[-1].weight,std=.02);order=torch.tensor([2,0,1])
        expected=model(self.pose,self.coarse);actual=model(self.pose[order],self.coarse[order]);self.assertTrue(torch.allclose(actual,expected[order],atol=1e-6))
        self.assertEqual(model(self.pose[:1],self.coarse[:1]).shape,(1,2,2))


if __name__=='__main__':unittest.main()
