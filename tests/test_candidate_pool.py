import unittest
import torch
from mousegeoforge.candidate_pool import merge_candidate_pools


class CandidatePoolTests(unittest.TestCase):
    def test_off_zero_copy(self):
        b=torch.rand(2,6,4);s=torch.rand(2,6);x,y=merge_candidate_pools(None,None,b,s,False)
        self.assertIs(x,b);self.assertIs(y,s)

    def test_keeps_duplicates_and_order(self):
        b=torch.rand(4,6,4);s=torch.rand(4,6)
        x,y=merge_candidate_pools(b,s,b,s,True)
        self.assertEqual(x.shape,(4,12,4));torch.testing.assert_close(x[:,:6],x[:,6:],atol=0,rtol=0)
        torch.testing.assert_close(y,torch.cat((s,s),1),atol=0,rtol=0)

    def test_rejects_mismatched_views(self):
        with self.assertRaises(ValueError):merge_candidate_pools(torch.rand(2,6,4),torch.rand(2,6),torch.rand(1,6,4),torch.rand(1,6),True)


if __name__=='__main__':unittest.main()
