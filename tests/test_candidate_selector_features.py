import unittest
import numpy as np
import torch
from mousegeoforge.candidate_selector_features import build_selector_inputs


class CandidateFeatureTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(17);v=4
        self.p=dict(candidate_xy=rng.uniform(.2,.8,(v,6,5,6,2)),candidate_mass=rng.uniform(.01,.8,(v,6,5,6)),xy=rng.uniform(.2,.8,(v,6,23,2)),objectness_logits=np.tile([4,-4,-4,-4,-4,-4],(v,1)),tokens=rng.normal(size=(v,6,5,256)),candidate_memory=rng.normal(size=(v,6,5,6,256)),probability_entropy=np.ones((v,6,5))*.5,probability_variance=np.ones((v,6,5,2))*.01)
        k=np.array([[600.,0,320],[0,600,240],[0,0,1]])
        self.cameras=np.stack([k@np.column_stack((np.eye(3),[-x,0,0])) for x in range(v)])
    def build(self,ids):return build_selector_inputs({k:v[ids] for k,v in self.p.items()},self.cameras[ids],ids,[640,480])
    def test_query_eligibility(self):
        x=self.build([0,1,2,3]);self.assertTrue(x['eligible'][:,0].all());self.assertFalse(x['eligible'][:,1:].any())
    def test_permutation(self):
        x=self.build([0,1,2,3]);y=self.build([3,1,0,2]);order=[2,1,3,0]
        for key in x:torch.testing.assert_close(x[key],y[key][order],atol=1e-7,rtol=0)
    def test_duplicate(self):
        x=self.build([0,1,2,3]);y=self.build([0,1,2,3,1,2])
        for key in x:torch.testing.assert_close(x[key],y[key][:4],atol=1e-7,rtol=0)
    def test_single_paired_fallback(self):
        for ids in [[0],[0,1]]:self.assertFalse(self.build(ids)['eligible'].any())
    def test_missing_support(self):
        self.p['objectness_logits'][2:]=-4
        self.assertFalse(self.build([0,1,2,3])['eligible'].any())
    def test_ambiguous_query(self):
        self.p['objectness_logits'][0,1]=4;x=self.build([0,1,2,3]);self.assertFalse(x['eligible'][0].any());self.assertTrue(x['eligible'][1:,0].all())
    def test_conflicting_duplicates_abstain(self):
        ids=[0,1,2,3,1];p={k:v[ids].copy() for k,v in self.p.items()};p['candidate_xy'][-1]+=.1
        x=build_selector_inputs(p,self.cameras[ids],ids,[640,480]);self.assertFalse(x['eligible'][[1,4]].any())


if __name__=='__main__':unittest.main()
