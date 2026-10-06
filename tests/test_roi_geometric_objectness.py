import unittest
import torch
from mousegeoforge.roi_set_objectness import SetObjectnessProbe
from mousegeoforge.roi_geometric_objectness import GeometricObjectnessProbe,pair_geometry


class GeometryObjectnessTests(unittest.TestCase):
    def data(self,views=2):
        x=torch.randn(views,6,37)
        x[...,-5:-3]=torch.rand(views,6,2)*.8+.1
        x[...,-3:-1]=torch.rand(views,6,2)*.3+.1
        x[...,-1]=torch.rand(views,6)
        return x

    def test_pair_self_translation_scale(self):
        b=torch.tensor([[[.2,.3,.2,.1],[.4,.3,.2,.1]]]);s=torch.tensor([[.1,.9]])
        g=pair_geometry(b,s);self.assertEqual(float(g[0,0,0,0]),1.)
        torch.testing.assert_close(g[0,0,0,1:],torch.zeros(5),atol=0,rtol=0)
        translated=b.clone();translated[...,:2]+=.2
        torch.testing.assert_close(g,pair_geometry(translated,s),atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(g,pair_geometry(b*2,s),atol=1e-6,rtol=1e-6)

    def test_off_and_zero_initial_on(self):
        old=SetObjectnessProbe(16,8,True);model=GeometricObjectnessProbe(16,8,False)
        model.load_state_dict(old.state_dict(),strict=False);x=self.data()
        torch.testing.assert_close(model(x),old(x),atol=0,rtol=0)
        model.geometry_enabled=True
        torch.testing.assert_close(model(x),old(x),atol=1e-6,rtol=1e-6)

    def test_permutation(self):
        model=GeometricObjectnessProbe(16,8,True);x=self.data();order=torch.tensor([4,2,5,1,0,3])
        with torch.no_grad():model.geometry.weight.normal_(0,.1)
        torch.testing.assert_close(model(x[:,order]),model(x)[:,order],atol=1e-6,rtol=1e-6)

    def test_gradients_views(self):
        for views in (1,2,4):
            model=GeometricObjectnessProbe(16,8,True);x=self.data(views)
            torch.nn.functional.binary_cross_entropy_with_logits(model(x),torch.rand(views,6)).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0 for p in model.parameters()))

    def test_invalid_geometry(self):
        with self.assertRaises(ValueError):pair_geometry(torch.zeros(1,6,4),torch.ones(1,6))


if __name__=='__main__':unittest.main()
