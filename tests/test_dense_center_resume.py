import copy
import unittest
import torch
from mousegeoforge.dense_center_probe import DenseCenterProbe,center_targets,center_loss


class CenterResumeTests(unittest.TestCase):
    def test_adamw_continuation_exact_cpu(self):
        torch.manual_seed(37)
        features=torch.randn(2,16,8,12)
        boxes=torch.tensor([[[.2,.3,.15,.2]],[[.7,.6,.2,.3]]])
        targets=center_targets(boxes,torch.ones(2,1,dtype=torch.bool),(8,12))
        first=DenseCenterProbe(16,8)
        opt=torch.optim.AdamW(first.parameters(),lr=.001,weight_decay=.01)
        def step(model,optimizer):
            optimizer.zero_grad(set_to_none=True)
            loss,_=center_loss(model(features),targets);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        for _ in range(3):step(first,opt)
        state=copy.deepcopy(first.state_dict());moments=copy.deepcopy(opt.state_dict())
        for _ in range(4):step(first,opt)
        second=DenseCenterProbe(16,8);second.load_state_dict(state)
        resumed=torch.optim.AdamW(second.parameters(),lr=.001,weight_decay=.01)
        resumed.load_state_dict(moments)
        self.assertTrue(all(float(s['step'])==3 for s in resumed.state.values()))
        for _ in range(4):step(second,resumed)
        for name,value in first.state_dict().items():torch.testing.assert_close(value,second.state_dict()[name],atol=0,rtol=0)


if __name__=='__main__':unittest.main()
