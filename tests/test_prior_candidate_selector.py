import unittest
import torch
from mousegeoforge.models.prior_candidate_selector import PriorCandidateSelector


class PriorCandidateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.model=PriorCandidateSelector()
        self.args=dict(semantic=torch.randn(2,5,256),local=torch.randn(2,5,6,256),scalars=torch.rand(2,5,6,9),geometry=torch.randn(2,5,6,6),candidate_xy=torch.rand(2,5,6,2),baseline_xy=torch.rand(2,5,2),eligible=torch.ones(2,5,dtype=torch.bool))
        self.args['scalars'][...,0]=-torch.rand(2,5,6)
        self.args['candidate_xy'][:,:,0]=self.args['baseline_xy']

    def test_off_and_initial_exact(self):
        self.assertIs(self.model(**self.args)['xy'],self.args['baseline_xy'])
        torch.testing.assert_close(self.model(**self.args,enabled=True)['xy'],self.args['baseline_xy'],atol=0,rtol=0)

    def test_no_support_nan_fallback(self):
        self.args['eligible'].zero_()
        for key in ['semantic','local','scalars','geometry']:self.args[key].fill_(float('nan'))
        out=self.model(**self.args,enabled=True)
        self.assertTrue(torch.isfinite(out['logits']).all())
        torch.testing.assert_close(out['xy'],self.args['baseline_xy'],atol=0,rtol=0)

    def test_weak_candidates_cannot_beat_baseline(self):
        torch.nn.init.normal_(self.model.score[-1].weight,std=100)
        self.args['scalars'][...,1:,0]=-.11 # logmass=-2.2 < -bound2
        out=self.model(**self.args,enabled=True)
        self.assertTrue((out['choice']==0).all())
        self.assertTrue((out['logits'][...,1:]<out['logits'][...,:1]).all())

    def test_gradient_only_selector(self):
        self.args['local'].requires_grad_();self.args['scalars'].requires_grad_()
        out=self.model(**self.args,enabled=True)
        (-out['logits'].log_softmax(-1)[...,1].mean()).backward()
        self.assertGreater(float(self.model.score[-1].weight.grad.abs().sum()),0)
        self.assertIsNone(self.args['local'].grad);self.assertIsNone(self.args['scalars'].grad)

    def test_unary_ignores_geometry(self):
        m=PriorCandidateSelector('unary');torch.nn.init.normal_(m.score[-1].weight)
        a=m(**self.args,enabled=True);self.args['geometry'].fill_(float('nan'));b=m(**self.args,enabled=True)
        torch.testing.assert_close(a['logits'],b['logits'],atol=0,rtol=0)

    def test_permutation(self):
        torch.nn.init.normal_(self.model.score[-1].weight)
        a=self.model(**self.args,enabled=True);b=self.model(**{k:v.flip(0) for k,v in self.args.items()},enabled=True)
        torch.testing.assert_close(a['xy'],b['xy'].flip(0),atol=0,rtol=0)

    def test_bounds_validated(self):
        for bound in [0,-1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):PriorCandidateSelector(residual_bound=bound)


if __name__=='__main__':unittest.main()
