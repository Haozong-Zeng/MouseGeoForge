import unittest
import torch
from torch import nn
from mousegeoforge.models.view_token_probe import shared_first_view_tokens
from vggt_omega.models.aggregator import slice_expand_and_flatten

def make_model():
    model=nn.Module();model.camera_token=nn.Parameter(torch.randn(1,2,1,8));model.register_token=nn.Parameter(torch.randn(1,2,3,8));model.other=nn.Parameter(torch.randn(4));return model.eval().requires_grad_(False)

class ViewTokenProbeTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(17)
    def test_default_is_noop(self):
        model=make_model();old={k:v.clone() for k,v in model.state_dict().items()}
        with shared_first_view_tokens(model):
            for k,v in model.state_dict().items():torch.testing.assert_close(v,old[k],atol=0,rtol=0)
    def test_single_unchanged_multiview_shared_and_restore(self):
        model=make_model();old={k:v.clone() for k,v in model.state_dict().items()};single=slice_expand_and_flatten(model.camera_token,2,1).clone()
        with shared_first_view_tokens(model,enabled=True):
            torch.testing.assert_close(single,slice_expand_and_flatten(model.camera_token,2,1),atol=0,rtol=0)
            for name in ['camera_token','register_token']:
                p=getattr(model,name);expanded=slice_expand_and_flatten(p,2,7).reshape(2,7,*p.shape[2:]);torch.testing.assert_close(expanded,old[name][:,:1].expand(2,7,*p.shape[2:]),atol=0,rtol=0)
            torch.testing.assert_close(model.other,old['other'],atol=0,rtol=0)
        for k,v in model.state_dict().items():torch.testing.assert_close(v,old[k],atol=0,rtol=0)
    def test_exception_restores(self):
        model=make_model();old=model.camera_token.clone()
        with self.assertRaisesRegex(RuntimeError,'injected'):
            with shared_first_view_tokens(model,enabled=True):raise RuntimeError('injected')
        torch.testing.assert_close(model.camera_token,old,atol=0,rtol=0)
    def test_reject_training_or_trainable_parameters(self):
        model=make_model().train()
        with self.assertRaises(ValueError):
            with shared_first_view_tokens(model,enabled=True):pass
        model.eval().requires_grad_(True)
        with self.assertRaises(ValueError):
            with shared_first_view_tokens(model,enabled=True):pass
    def test_toy_permutation_with_real_token_expansion(self):
        model=make_model();x=torch.randn(1,3,8);perm=torch.tensor([2,0,1])
        def forward(x):
            y=x+slice_expand_and_flatten(model.camera_token,1,3).reshape(1,3,8)
            return torch.softmax(y@y.transpose(-1,-2),-1)@y
        original=forward(x);reordered=forward(x[:,perm])[:,torch.argsort(perm)]
        self.assertGreater(float((original-reordered).abs().max()),.01)
        with shared_first_view_tokens(model,enabled=True):torch.testing.assert_close(forward(x),forward(x[:,perm])[:,torch.argsort(perm)],atol=2e-6,rtol=0)

if __name__=='__main__':unittest.main()
