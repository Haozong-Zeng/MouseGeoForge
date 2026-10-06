import unittest
import torch
from mousegeoforge.roi_candidate_quality import region_features,RegionQualityProbe,apply_quality_scores


class RegionQualityTests(unittest.TestCase):
    def test_shape_gradient_views(self):
        for v in (1,2,4):
            x=torch.randn(v,16,8,12)
            boxes=torch.tensor([[[.3,.4,.3,.2],[.7,.7,.2,.2]]]).expand(v,-1,-1).clone()
            scores=torch.full((v,2),.2);before=boxes.clone()
            d=region_features(x,boxes,scores);self.assertEqual(d.shape,(v,2,37))
            m=RegionQualityProbe(16,8);torch.nn.functional.binary_cross_entropy_with_logits(m(d),torch.full((v,2),.7)).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0 for p in m.parameters()))
            torch.testing.assert_close(before,boxes,atol=0,rtol=0)

    def test_off_exact_passthrough(self):
        s=torch.rand(2,6)
        self.assertIs(apply_quality_scores(None,None,s,False),s)

    def test_view_isolation(self):
        x=torch.randn(2,16,8,12);b=torch.tensor([[[.5,.5,.5,.5]]]).expand(2,-1,-1);s=torch.ones(2,1)
        d=region_features(x,b,s)
        expected=torch.cat([region_features(x[i:i+1],b[i:i+1],s[i:i+1]) for i in range(2)])
        torch.testing.assert_close(d,expected,atol=0,rtol=0)

    def test_constant_region(self):
        x=torch.ones(1,4,8,12);b=torch.tensor([[[.5,.5,.4,.4]]]);s=torch.ones(1,1)
        torch.testing.assert_close(region_features(x,b,s)[0,0,:8],torch.full((8,),.5),atol=1e-6,rtol=0)

    def test_bad_roi(self):
        with self.assertRaises(ValueError):region_features(torch.ones(1,4,8,12),torch.tensor([[[2.,2.,.2,.2]]]),torch.ones(1,1))


if __name__=='__main__':unittest.main()
