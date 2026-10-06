import unittest
import torch
from torch import nn
from mousegeoforge.models.independent_view_pose_probe import POSE_KEYS
from mousegeoforge.models.shared_token_pose_probe import shared_token_pose_probe

class Toy(nn.Module):
    def __init__(self):
        super().__init__();self.aggregator=nn.Module();self.aggregator.camera_token=nn.Parameter(torch.tensor([[[[1.]],[[3.]]]]));self.aggregator.register_token=nn.Parameter(torch.zeros(1,2,1,1));self.fail=False;self.calls=0;self.eval().requires_grad_(False)
    def forward(self,x,coupling='off'):
        self.calls+=1
        if self.fail:raise RuntimeError('injected')
        b,v=x.shape[:2];p=self.aggregator.camera_token
        offset=torch.cat([p[:,:1].flatten(2),p[:,1:].flatten(2).expand(1,v-1,1)],1)
        value=x.mean((2,3,4))[...,None]+offset
        return {**{k:value.clone() for k in POSE_KEYS},'depth':value[...,None],'auxiliary_unvalidated':value}

class SharedPoseProbeTests(unittest.TestCase):
    def test_geometry_identity_and_parameter_restore(self):
        model=Toy();x=torch.randn(2,3,3,4,4);base=model(x);before={k:v.clone() for k,v in model.state_dict().items()};out=shared_token_pose_probe(x,base,model)
        self.assertIs(out['depth'],base['depth']);self.assertIs(out['auxiliary_unvalidated'],base['auxiliary_unvalidated'])
        for key in POSE_KEYS:torch.testing.assert_close(out[key],x.mean((2,3,4))[...,None]+1,atol=0,rtol=0)
        for key in before:torch.testing.assert_close(before[key],model.state_dict()[key],atol=0,rtol=0)
    def test_single_no_extra_pass(self):
        model=Toy();x=torch.randn(2,1,3,4,4);base=model(x);model.fail=True;out=shared_token_pose_probe(x,base,model);self.assertEqual(model.calls,1)
        for k in base:self.assertIs(out[k],base[k])
    def test_exception_restores(self):
        model=Toy();x=torch.randn(1,2,3,4,4);base=model(x);before=model.aggregator.camera_token.clone();model.fail=True
        with self.assertRaisesRegex(RuntimeError,'injected'):shared_token_pose_probe(x,base,model)
        torch.testing.assert_close(before,model.aggregator.camera_token,atol=0,rtol=0)
    def test_invalid_input(self):
        model=Toy();x=torch.randn(1,2,3,4,4)
        with self.assertRaises(ValueError):shared_token_pose_probe(x,{},model)

if __name__=='__main__':unittest.main()
