import io
import unittest
import torch
from vggt_omega.models.heads.instance_pose_head import MultiInstanceMouseHead
from mousegeoforge.adaptive_temperature import ConditionalTemperature, install_adaptive_temperature,restore_adaptive,PREFIX


class AdaptiveTemperatureTests(unittest.TestCase):
    def head_inputs(self):
        torch.manual_seed(17)
        head=MultiInstanceMouseHead(dim_in=32,hidden_dim=32,num_heads=4,num_queries=2)
        pixels=torch.randn(2,32,8,8,requires_grad=True)
        tokens=torch.randn(2,2,23,32,requires_grad=True)
        coarse=torch.rand(2,2,23,2,requires_grad=True)
        return head,pixels,tokens,coarse

    def test_zero_initialization_exact_output_and_old_gradient(self):
        head,pixels,tokens,coarse=self.head_inputs()
        a= head._local_refinement(pixels,tokens,coarse)
        ga=torch.autograd.grad(a[1].sum(),[pixels,tokens,coarse],retain_graph=True)
        rng=torch.random.get_rng_state().clone();install_adaptive_temperature(head)
        torch.testing.assert_close(rng,torch.random.get_rng_state(),atol=0,rtol=0)
        b=head._local_refinement(pixels,tokens,coarse)
        for x,y in zip(a,b):torch.testing.assert_close(x,y,atol=0,rtol=0)
        gb=torch.autograd.grad(b[1].sum(),[pixels,tokens,coarse],retain_graph=True)
        for x,y in zip(ga,gb):torch.testing.assert_close(x,y,atol=0,rtol=0)
        gradients=torch.autograd.grad(b[1].sum(),list(head.adaptive_temperature.parameters()))
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients));self.assertGreater(sum(float(g.abs().sum()) for g in gradients),0)

    def test_bounded_and_query_permutation(self):
        torch.manual_seed(17)
        temp=ConditionalTemperature(8);tokens=torch.randn(2,3,23,8)
        with torch.no_grad():temp.weight.fill_(.3);temp.bias.fill_(.2)
        # Different vectorized row layouts can differ by one float32 ULP.
        torch.testing.assert_close(temp(tokens.flip(1)),temp(tokens).flip(1),atol=1e-7,rtol=1e-6)
        with torch.no_grad():temp.bias.fill_(100)
        torch.testing.assert_close(temp(tokens),torch.full((2,3,23),8.))
        with torch.no_grad():temp.bias.fill_(-100)
        torch.testing.assert_close(temp(tokens),torch.full((2,3,23),.5))

    def test_reload_and_disabled_path(self):
        head,pixels,tokens,coarse=self.head_inputs();original=head._local_refinement(pixels,tokens,coarse)
        install_adaptive_temperature(head)
        with torch.no_grad():head.adaptive_temperature.bias.fill_(.5)
        modified=head._local_refinement(pixels,tokens,coarse)
        saved=io.BytesIO();torch.save(head.state_dict(),saved);saved.seek(0)
        state=torch.load(saved,weights_only=True)
        head.load_state_dict(state)
        for a,b in zip(modified,head._local_refinement(pixels,tokens,coarse)):torch.testing.assert_close(a,b,atol=0,rtol=0)
        head.adaptive_temperature_enabled=False
        for a,b in zip(original,head._local_refinement(pixels,tokens,coarse)):torch.testing.assert_close(a,b,atol=0,rtol=0)

    def test_original_restore_resets_branch_and_rejects_partial_state(self):
        head,_,_,_=self.head_inputs();model=torch.nn.Module();model.instance_pose_head=head
        original={k:v.clone() for k,v in model.state_dict().items()}
        install_adaptive_temperature(model)
        with torch.no_grad():head.adaptive_temperature.bias.fill_(.7)
        restore_adaptive(model,original)
        self.assertEqual(float(head.adaptive_temperature.bias.detach().sum()),0.)
        partial=dict(original);partial[PREFIX+'bias']=torch.zeros(1)
        with self.assertRaises(ValueError):restore_adaptive(model,partial)
        with self.assertRaises(ValueError):restore_adaptive(model,dict(original,unknown=torch.zeros(1)))


if __name__=='__main__':unittest.main()
