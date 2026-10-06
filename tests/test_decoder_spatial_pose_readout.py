import unittest
import torch
from mousegeoforge.models.decoder_spatial_pose_readout import DecoderSpatialPoseReadout
from mousegeoforge.models.core23_spatial_readout import Core23SpatialReadout

class DecoderSpatialTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(17)
    def test_all138_queries_and_shape(self):
        model=DecoderSpatialPoseReadout().eval();seen=[];h=model.keypoint_decoder.register_forward_pre_hook(lambda m,a:seen.append(a[0].shape))
        with torch.no_grad():p=model(torch.randn(2,6,256),torch.randn(2,256,3,4),torch.full((2,6,4),.4))
        h.remove();self.assertEqual(seen,[torch.Size([2,138,256])]);self.assertEqual(p['xy'].shape,(2,6,23,2));self.assertTrue(torch.isfinite(p['xy']).all())
    def test_opt_in_gradients_only_decoder_and_readout(self):
        for enabled in [False,True]:
            model=DecoderSpatialPoseReadout().eval();model.set_decoder_trainable(enabled);torch.nn.init.normal_(model.spatial_readout.query[-1].weight,std=.01)
            inst=torch.randn(1,6,256,requires_grad=True);memory=torch.randn(1,256,3,4,requires_grad=True);boxes=torch.full((1,6,4),.4,requires_grad=True)
            model(inst,memory,boxes)['xy'].square().sum().backward()
            grads=[p.grad for p in model.keypoint_decoder.parameters()]
            if enabled:self.assertTrue(any(g is not None and g.abs().sum()>0 for g in grads))
            else:self.assertTrue(all(g is None for g in grads))
            self.assertIsNone(inst.grad);self.assertIsNone(memory.grad);self.assertIsNone(boxes.grad)
            self.assertTrue(all(p.grad is None for p in model.keypoint_queries.parameters()));self.assertTrue(all(p.grad is None for p in model.coarse_keypoint_xy.parameters()))
    def test_default_readout_still_detaches_pose(self):
        model=Core23SpatialReadout();torch.nn.init.normal_(model.query[-1].weight,std=.01)
        p=torch.randn(1,23,256,requires_grad=True);coarse=torch.rand(1,23,2);mem=torch.randn(1,256,3,4)
        model(p,coarse,mem).sum().backward();self.assertIsNone(p.grad)
    def test_freeze_toggle_preserves_outputs(self):
        model=DecoderSpatialPoseReadout().eval();x=torch.randn(1,6,256);m=torch.randn(1,256,3,4);b=torch.full((1,6,4),.4)
        with torch.no_grad():a=model(x,m,b);model.set_decoder_trainable(True);c=model(x,m,b)
        # CPU attention kernel selection can vary with requires_grad flags,
        # even inside no_grad. Permit float32 rounding, not geometric drift.
        for k in a:torch.testing.assert_close(a[k],c[k],atol=2e-6 if k=='tokens' else 2e-7,rtol=0)

if __name__=='__main__':unittest.main()
