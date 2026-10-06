import unittest
import torch
from torch.nn import functional as F
from mousegeoforge.deterministic_local_sampling import frozen_bilinear_sample


class TestSampling(unittest.TestCase):
    def test_cpu_forward_and_grid_gradient(self):
        torch.manual_seed(17)
        p=torch.randn(2,3,7,9,dtype=torch.float64)
        g=(torch.rand(2,5,6,2,dtype=torch.float64)*3-1.5).requires_grad_(True)
        ref=F.grid_sample(p,g,align_corners=False);actual=frozen_bilinear_sample(p,g)
        torch.testing.assert_close(actual,ref,atol=1e-12,rtol=1e-12)
        grad0=torch.autograd.grad(ref.square().sum(),g)[0];grad1=torch.autograd.grad(actual.square().sum(),g)[0]
        torch.testing.assert_close(grad1,grad0,atol=1e-11,rtol=1e-11)

    def test_centers_edges_outside(self):
        p=torch.arange(12,dtype=torch.float64).reshape(1,1,3,4)
        g=torch.tensor([[[[-1.,-1.],[1.,1.],[-2.,0.],[0.,2.],[0.,0.]]]],dtype=torch.float64)
        torch.testing.assert_close(frozen_bilinear_sample(p,g),F.grid_sample(p,g,align_corners=False),atol=0,rtol=0)

    def test_rejects_feature_training(self):
        with self.assertRaises(ValueError):frozen_bilinear_sample(torch.ones(1,1,2,2,requires_grad=True),torch.zeros(1,1,1,2))


if __name__=='__main__':unittest.main()
