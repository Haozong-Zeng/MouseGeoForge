import copy
import unittest
import torch
from torch.nn import functional as F
from mousegeoforge.roi_set_objectness import SetObjectnessProbe


class RoiPoolResumeTests(unittest.TestCase):
    def check_resume(self, candidates):
        torch.manual_seed(17)
        model=SetObjectnessProbe(relation_enabled=True)
        features=torch.randn(2,candidates,517)
        targets=torch.zeros(2,candidates);targets[:,0]=1
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
        def step(head,opt):
            opt.zero_grad(set_to_none=True)
            F.binary_cross_entropy_with_logits(head(features),targets).backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
            opt.step()
        for _ in range(3):step(model,optimizer)
        state=copy.deepcopy(model.state_dict());moments=copy.deepcopy(optimizer.state_dict())
        for _ in range(4):step(model,optimizer)
        resumed=SetObjectnessProbe(relation_enabled=True);resumed.load_state_dict(state)
        opt=torch.optim.AdamW(resumed.parameters(),lr=.001,weight_decay=.01);opt.load_state_dict(moments)
        self.assertTrue(all(int(s['step'])==3 for s in opt.state.values()))
        for _ in range(4):step(resumed,opt)
        for name,value in model.state_dict().items():
            torch.testing.assert_close(value,resumed.state_dict()[name],atol=0,rtol=0)
        self.assertTrue(all(int(s['step'])==7 for s in opt.state.values()))

    def test_center6(self):self.check_resume(6)
    def test_union12(self):self.check_resume(12)


if __name__=='__main__':unittest.main()
