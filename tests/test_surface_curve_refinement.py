import unittest
import torch
from mousegeoforge.models.surface_curve_refinement import SurfaceCurveSpatialRefiner
from mousegeoforge.surface_curve_supervision import cubic_bspline


class RefinementTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17);torch.set_num_threads(2)
        self.pose=torch.randn(3,1024,requires_grad=True)
        self.memory=torch.randn(3,256,12,16,requires_grad=True)
        self.controls=torch.rand(3,6,2,requires_grad=True)

    def test_initialization_and_parameter_match(self):
        torch.manual_seed(2);local=SurfaceCurveSpatialRefiner('local')
        torch.manual_seed(2);glob=SurfaceCurveSpatialRefiner('global')
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(local.state_dict().values(),glob.state_dict().values())))
        expected=cubic_bspline(self.controls,33)
        for model in [local,glob]:self.assertTrue(torch.equal(model(self.pose,self.memory,self.controls),expected))

    def test_frozen_inputs_and_trainable_update(self):
        model=SurfaceCurveSpatialRefiner();model(self.pose,self.memory,self.controls).square().mean().backward()
        for x in [self.pose,self.memory,self.controls]:self.assertIsNone(x.grad)
        self.assertGreater(float(model.update[-1].weight.grad.abs().sum()),0)

    def test_instance_permutation_equivariance(self):
        model=SurfaceCurveSpatialRefiner()
        torch.nn.init.normal_(model.update[-1].weight,std=.01)
        order=torch.tensor([2,0,1]);a=model(self.pose,self.memory,self.controls)
        b=model(self.pose[order],self.memory[order],self.controls[order])
        self.assertTrue(torch.allclose(a[order],b,atol=1e-6))

    def test_spatial_shuffle_local_vs_global(self):
        local=SurfaceCurveSpatialRefiner('local');torch.nn.init.normal_(local.update[-1].weight,std=.02)
        glob=SurfaceCurveSpatialRefiner('global');glob.load_state_dict(local.state_dict())
        shuffled=self.memory.flip(-1)
        a=local(self.pose,self.memory,self.controls);b=local(self.pose,shuffled,self.controls)
        self.assertGreater(float((a-b).detach().abs().max()),1e-5)
        self.assertTrue(torch.allclose(glob(self.pose,self.memory,self.controls),glob(self.pose,shuffled,self.controls),atol=1e-6))

    def test_single_instance_and_outside_controls_finite(self):
        model=SurfaceCurveSpatialRefiner();controls=self.controls[:1]*3-1
        out=model(self.pose[:1],self.memory[:1],controls)
        self.assertEqual(out.shape,(1,33,2));self.assertTrue(torch.isfinite(out).all())


if __name__=='__main__':unittest.main()
