import unittest
import torch
from mousegeoforge.dense_center_context import ContextCenterProbe
from mousegeoforge.dense_center_probe import DenseCenterProbe, center_targets, center_loss


class ContextProbeTests(unittest.TestCase):
    def pair(self):
        torch.manual_seed(17)
        off = ContextCenterProbe(16, 8, False)
        torch.manual_seed(17)
        on = ContextCenterProbe(16, 8, True)
        return off, on

    def test_initial_equivalence(self):
        off, on = self.pair()
        torch.manual_seed(17)
        old = DenseCenterProbe(16, 8)
        x = torch.randn(2, 16, 8, 12)
        for key, value in off(x).items():
            torch.testing.assert_close(value, on(x)[key], atol=0, rtol=0)
            torch.testing.assert_close(value, old(x)[key], atol=0, rtol=0)
        self.assertEqual(sum(p.numel() for p in off.parameters()), sum(p.numel() for p in on.parameters()))

    @torch.no_grad()
    def test_spatial_receptive_field(self):
        off, on = self.pair()
        with torch.no_grad():
            for model in (off, on):
                model.shared.weight.fill_(1); model.shared.bias.fill_(1)
                model.output.weight.fill_(1)
                model.context.weight.fill_(1)
        x = torch.ones(1, 16, 7, 7)
        changed = x.clone(); changed[:, :, 3, 4] = 0
        self.assertEqual(float(off(x)['heat_logits'][0, 3, 3]), float(off(changed)['heat_logits'][0, 3, 3]))
        self.assertGreater(abs(float(on(x)['heat_logits'][0, 3, 3] - on(changed)['heat_logits'][0, 3, 3])), .1)

    def test_gradient_and_dormant_weights(self):
        for views in (1, 2, 4):
            off, on = self.pair()
            boxes = torch.tensor([[[.3,.3,.2,.2],[.7,.7,.2,.2]]]).expand(views,-1,-1)
            target = center_targets(boxes, torch.ones(views,2,dtype=torch.bool), (8,12))
            x = torch.randn(views,16,8,12)
            for model in (off, on):
                optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
                loss,_ = center_loss(model(x), target); loss.backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0 for p in model.parameters()))
                neighbors = model.context.weight.grad * (1-model.center_mask)
                self.assertEqual(bool(neighbors.abs().sum()>0), model.context_enabled)
                optimizer.step()
                self.assertEqual(bool((model.context.weight*(1-model.center_mask)).abs().sum()>0), model.context_enabled)

    def test_invalid_channels_no_cross_view_mixing(self):
        off,on = self.pair()
        x=torch.randn(4,16,8,12)
        for model in (off,on):
            y=model(x)['heat_logits']
            torch.testing.assert_close(y,torch.cat([model(v[None])['heat_logits'] for v in x]),atol=1e-6,rtol=1e-6)


if __name__ == '__main__': unittest.main()
