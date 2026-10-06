import unittest
import torch
from mousegeoforge.candidate_ceiling import geometric_ceiling


class CandidateCeilingTests(unittest.TestCase):
    def test_distinct_objects(self):
        boxes=torch.tensor([[.2,.3,.2,.2],[.8,.7,.2,.2],[.5,.5,.1,.1]])
        r=geometric_ceiling(boxes,boxes[:2]);self.assertEqual(r['covered'],2)

    def test_one_candidate_cannot_cover_two(self):
        boxes=torch.tensor([[.2,.3,.2,.2],[.8,.7,.2,.2]])
        gt=boxes[:1].expand(2,-1)
        self.assertEqual(geometric_ceiling(boxes,gt)['covered'],1)

    def test_permutation_and_empty(self):
        b=torch.tensor([[.2,.3,.2,.2],[.8,.7,.2,.2],[.5,.5,.1,.1]])
        self.assertEqual(geometric_ceiling(b[[2,0,1]],b[:2])['covered'],2)
        self.assertEqual(geometric_ceiling(b,b[:0])['covered'],0)


if __name__=='__main__':unittest.main()
