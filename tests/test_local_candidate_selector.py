import unittest
import torch
from mousegeoforge.models.local_candidate_selector import LocalCandidateSelector


class LocalCandidateSelectorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17);self.model=LocalCandidateSelector();self.args=dict(semantic=torch.randn(2,5,256),local=torch.randn(2,5,6,256),scalars=torch.randn(2,5,6,9),geometry=torch.randn(2,5,6,6),candidate_xy=torch.rand(2,5,6,2),baseline_xy=torch.rand(2,5,2),eligible=torch.ones(2,5,dtype=torch.bool))
        self.args['candidate_xy'][:,:,0]=self.args['baseline_xy']
    def test_off_is_exact_object(self):self.assertIs(self.model(**self.args)['xy'],self.args['baseline_xy'])
    def test_initial_exact_baseline(self):
        out=self.model(**self.args,enabled=True);torch.testing.assert_close(out['xy'],self.args['baseline_xy'],atol=0,rtol=0);self.assertFalse(out['active'].any())
    def test_unavailable_and_nan_fallback(self):
        self.args['eligible'].zero_();self.args['local'].fill_(float('nan'));self.args['geometry'].fill_(float('nan'))
        out=self.model(**self.args,enabled=True);self.assertTrue(torch.isfinite(out['logits']).all());torch.testing.assert_close(out['xy'],self.args['baseline_xy'],atol=0,rtol=0)
    def test_parameter_matched(self):
        other=LocalCandidateSelector('unary');self.assertEqual(sum(p.numel() for p in self.model.parameters()),sum(p.numel() for p in other.parameters()))
    def test_unary_ignores_geometry(self):
        model=LocalCandidateSelector('unary');torch.nn.init.normal_(model.score[-1].weight)
        a=model(**self.args,enabled=True);self.args['geometry'].fill_(float('nan'));b=model(**self.args,enabled=True)
        torch.testing.assert_close(a['logits'],b['logits'],atol=0,rtol=0)
    def test_loss_gradients_only_readout(self):
        self.args['local'].requires_grad_();out=self.model(**self.args,enabled=True);loss=torch.nn.functional.cross_entropy(out['logits'].flatten(0,1),torch.ones(10,dtype=torch.long));loss.backward()
        self.assertGreater(float(self.model.score[-1].weight.grad.abs().sum()),0);self.assertIsNone(self.args['local'].grad)
    def test_batch_permutation(self):
        torch.nn.init.normal_(self.model.score[-1].weight);a=self.model(**self.args,enabled=True)
        b=self.model(**{k:v.flip(0) for k,v in self.args.items()},enabled=True);torch.testing.assert_close(a['xy'],b['xy'].flip(0),atol=0,rtol=0)


if __name__=='__main__':unittest.main()
