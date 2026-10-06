import unittest
import torch
from mousegeoforge.models.core23_spatial_readout import Core23SpatialReadout


class ReadoutTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(2)
        self.pose = torch.randn(2, 23, 256, requires_grad=True)
        self.coarse = torch.rand(2, 23, 2, requires_grad=True)
        self.memory = torch.randn(2, 256, 4, 5, requires_grad=True)

    def test_shape_probability_and_frozen_features(self):
        model = Core23SpatialReadout()
        xy, p = model(self.pose, self.coarse, self.memory, True)
        self.assertEqual(xy.shape, (2, 23, 2))
        torch.testing.assert_close(p.sum((-1, -2)), torch.ones(2, 23))
        self.assertTrue(((xy >= 0) & (xy <= 1)).all())
        xy.sum().backward()
        self.assertIsNone(self.pose.grad)
        self.assertIsNone(self.coarse.grad)
        self.assertIsNone(self.memory.grad)
        self.assertGreater(model.query[-1].weight.grad.abs().sum().item(), 0)

    def test_matched_initialization(self):
        spatial = Core23SpatialReadout("spatial")
        global_ = Core23SpatialReadout("global")
        global_.load_state_dict(spatial.state_dict(), strict=True)
        self.assertEqual(sum(p.numel() for p in spatial.parameters()),
                         sum(p.numel() for p in global_.parameters()))
        torch.testing.assert_close(spatial(self.pose, self.coarse, self.memory),
                                   global_(self.pose, self.coarse, self.memory), atol=0, rtol=0)

    def test_keypoint_permutation_equivariance(self):
        model = Core23SpatialReadout()
        nn = torch.nn
        nn.init.normal_(model.query[-1].weight, std=.05)
        perm = torch.randperm(23)
        torch.testing.assert_close(model(self.pose[:, perm], self.coarse[:, perm], self.memory),
                                   model(self.pose, self.coarse, self.memory)[:, perm])

    def test_global_control_ignores_pixel_permutation(self):
        model = Core23SpatialReadout("global")
        torch.nn.init.normal_(model.query[-1].weight, std=.05)
        shuffled = self.memory.flatten(2)[:, :, torch.randperm(20)].reshape_as(self.memory)
        torch.testing.assert_close(model(self.pose, self.coarse, self.memory),
                                   model(self.pose, self.coarse, shuffled))

    def test_reject_old_four_back_token_input(self):
        with self.assertRaises(ValueError):
            Core23SpatialReadout()(torch.randn(2, 1024), self.coarse, self.memory)

    def test_prediction_only_box_prior_and_gradient(self):
        model = Core23SpatialReadout()
        boxes = torch.tensor([[.2,.3,.1,.1],[.2,.3,.1,.1]], requires_grad=True)
        xy, prob = model(self.pose, self.coarse, self.memory, True, predicted_boxes=boxes)
        self.assertTrue(torch.isfinite(xy).all())
        self.assertTrue((prob > 0).all())  # soft, not hard GT crop
        xy.sum().backward()
        self.assertIsNone(boxes.grad)

    def test_bad_box_shape(self):
        with self.assertRaises(ValueError):
            Core23SpatialReadout()(self.pose,self.coarse,self.memory,predicted_boxes=torch.zeros(1,4))

    def test_optional_log_probability_preserves_default(self):
        model=Core23SpatialReadout()
        xy,p,logp=model(self.pose,self.coarse,self.memory,return_log_probability=True)
        torch.testing.assert_close(p,logp.exp())
        torch.testing.assert_close(xy,model(self.pose,self.coarse,self.memory),atol=0,rtol=0)


if __name__ == "__main__":
    unittest.main()
