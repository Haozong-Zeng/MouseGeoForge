import unittest
import torch
from mousegeoforge.surface_curve_supervision import material_curve_loss

class TestMaterialLoss(unittest.TestCase):
    def inputs(self):
        u=torch.linspace(0,1,65);g=torch.stack([500*u,100+20*u],-1);return g[::2].clone(),g,torch.ones(65,dtype=torch.bool),u
    def test_exact_and_translation_gradients(self):
        p,g,v,u=self.inputs();self.assertEqual(float(material_curve_loss(p,g,v,u,material_parameter_known=True)),0)
        p=(p+torch.tensor([8.,0.])).requires_grad_(True);loss=material_curve_loss(p,g,v,u,material_parameter_known=True);loss.backward()
        self.assertTrue(torch.isfinite(p.grad).all());self.assertGreater(p.grad[:,0].sum(),0)
    def test_requires_complete_known_material_not_manual_points(self):
        p,g,v,u=self.inputs()
        with self.assertRaises(ValueError):material_curve_loss(p,g,v,u)
        v[2]=False
        with self.assertRaises(ValueError):material_curve_loss(p,g,v,u,material_parameter_known=True)
        v[2]=True;u[5]=u[4]
        with self.assertRaises(ValueError):material_curve_loss(p,g,v,u,material_parameter_known=True)
    def test_reversal_penalized_and_nonuniform_sampling(self):
        p,g,v,u=self.inputs();self.assertGreater(float(material_curve_loss(p.flip(0),g,v,u,material_parameter_known=True)),.1)
        uu=u.square();gg=torch.stack([500*uu,100+20*uu],-1)
        torch.testing.assert_close(material_curve_loss(p,gg,v,uu,material_parameter_known=True),torch.tensor(0.),atol=1e-12,rtol=0)
if __name__=='__main__':unittest.main()
