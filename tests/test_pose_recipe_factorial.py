import importlib.util
from pathlib import Path
import torch
import unittest
path=Path(__file__).resolve().parents[1]/'tools/train_pose_recipe_factorial_20261005.py'
spec=importlib.util.spec_from_file_location('recipe',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class RecipeTests(unittest.TestCase):
    def test_mse_exact_reference(self):
        x=torch.rand(6,5,2,requires_grad=True);g=torch.rand_like(x);v=torch.rand(6,5)>.4
        self.assertTrue(torch.equal(module.objective(x,g,v,torch.tensor([704.,768.]),'mse'),(x-g).square().sum(-1)[v].mean()))

    def test_missing_nan_not_loss_and_no_gradient(self):
        x=torch.zeros(6,5,2,requires_grad=True);g=torch.full_like(x,float('nan'));v=torch.zeros(6,5,dtype=torch.bool);v[0,0]=True;g[0,0]=.1
        loss=module.objective(x,g,v,torch.tensor([768.,432.]),'pixel_huber');self.assertTrue(torch.isfinite(loss));loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all() and torch.equal(x.grad[~v],torch.zeros_like(x.grad[~v])))

    def test_huber_aspect_scaling_and_gradient(self):
        x=torch.ones(1,1,2,requires_grad=True);g=torch.zeros_like(x);v=torch.ones(1,1,dtype=torch.bool)
        loss=module.objective(x,g,v,torch.tensor([768.,384.]),'pixel_huber');self.assertAlmostEqual(float(loss.detach()),.745,places=6)
        loss.backward();self.assertTrue(torch.allclose(x.grad,torch.tensor([[[.5,.25]]])))

    def test_unknown_recipe_rejected(self):
        with self.assertRaises(ValueError):module.objective(torch.zeros(1),torch.zeros(1),torch.ones(1,dtype=torch.bool),torch.ones(2),'unknown')

if __name__=='__main__':unittest.main()
