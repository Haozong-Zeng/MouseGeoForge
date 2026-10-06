import unittest
import torch
from mousegeoforge.models.back_point_integral import BackPointIntegralProbe
from mousegeoforge.models.joint_surface_point_integral import JointSurfacePointIntegralProbe


class TestJointIntegral(unittest.TestCase):
    def setUp(self):torch.manual_seed(4);torch.set_num_threads(2)

    def test_point_initialization_preserved(self):
        base=BackPointIntegralProbe();base.query[-1].weight.data.normal_(0,.1)
        joint=JointSurfacePointIntegralProbe(base.state_dict());pose=torch.randn(2,1024);memory=torch.randn(2,256,8,9);coarse=torch.rand(2,2,2)
        torch.testing.assert_close(base(pose,coarse,memory),joint.forward_points(pose,memory,coarse),atol=0,rtol=0)

    def test_shared_weights_deduplicated_and_roundtrip(self):
        m=JointSurfacePointIntegralProbe();self.assertIs(m.point.key,m.curve.points.key);self.assertIs(m.point.position,m.curve.points.position)
        params=list(m.parameters());self.assertEqual(len(params),len({id(p) for p in params}))
        other=JointSurfacePointIntegralProbe();other.load_state_dict(m.state_dict())
        self.assertIs(other.point.key,other.curve.points.key)

    def test_real_point_loss_does_not_supervise_curve_query(self):
        m=JointSurfacePointIntegralProbe();m.point.query[-1].weight.data.normal_(0,.1)
        pose=torch.randn(2,1024);memory=torch.randn(2,256,8,9);coarse=torch.rand(2,2,2)
        m.forward_points(pose,memory,coarse).square().mean().backward()
        self.assertTrue(all(p.grad is None for p in m.curve.points.query.parameters()))
        self.assertGreater(float(sum(p.grad.abs().sum() for p in m.point.key.parameters() if p.grad is not None)),0)


if __name__=='__main__':unittest.main()
