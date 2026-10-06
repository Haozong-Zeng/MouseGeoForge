import unittest
import torch
from torch.nn import functional as F
from mousegeoforge.body_tail_loss_diagnostic import partition_coordinate_loss


class LossPartitionTest(unittest.TestCase):
    def run_case(self, mode):
        torch.manual_seed(17)
        coarse = torch.rand(3, 6, 23, 2, requires_grad=True)
        refined = torch.rand(3, 6, 23, 2, requires_grad=True)
        gt = torch.rand(1, 3, 4, 23, 2)
        valid = torch.rand(1, 3, 4, 23) > .3
        if mode == 'body': valid[..., 7:14] = False
        if mode == 'tail': valid[..., :7] = False; valid[..., 14:] = False
        matches = [([4, 1], [2, 0]), ([0], [3]), ([5, 2, 1], [3, 0, 2])]
        sample = dict(matches=matches, target=dict(keypoint_valid=valid, keypoint_xy=gt))
        reference = []
        for v, (qs, ms) in enumerate(matches):
            for q, m in zip(qs, ms):
                k = valid[0, v, m]
                if k.any(): reference.append(F.smooth_l1_loss(refined[v,q,k],gt[0,v,m,k],beta=.05) + .5*F.smooth_l1_loss(coarse[v,q,k],gt[0,v,m,k],beta=.05))
        total = torch.stack(reference).mean()
        parts = partition_coordinate_loss(refined, coarse, sample)
        torch.testing.assert_close(parts['body']+parts['tail'], total, atol=1e-7, rtol=1e-6)
        a = torch.autograd.grad(total, (coarse,refined), retain_graph=True)
        b = torch.autograd.grad(parts['body']+parts['tail'], (coarse,refined))
        for x,y in zip(a,b): torch.testing.assert_close(x,y,atol=1e-8,rtol=1e-6)
        if mode in parts: self.assertEqual(float(parts['tail' if mode=='body' else 'body'].detach()),0.)

    def test_mixed_visibility_noncontinuous_ids(self): self.run_case('mixed')
    def test_body_only(self): self.run_case('body')
    def test_tail_only(self): self.run_case('tail')
    def test_empty_rejected(self):
        with self.assertRaises(ValueError): partition_coordinate_loss(torch.zeros(1,6,23,2),torch.zeros(1,6,23,2),dict(matches=[([],[])],target={}))


if __name__ == '__main__': unittest.main()
