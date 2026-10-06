import unittest
import torch
from torchvision.ops import sigmoid_focal_loss
from mousegeoforge.models.proposal_objectness_losses import balanced_objectness,grouping_objectness_loss
from tests.test_proposal_instance_grouping import ProposalTests

class ObjectnessTests(unittest.TestCase):
    def test_focal_matches_reference_with_same_balance(self):
        x=torch.tensor([2.,-1.,0.,3.],requires_grad=True);valid=torch.tensor([True,True,True,False]);pos=torch.tensor([0]);y=torch.tensor([1.,0.,0.,0.])
        ref=sigmoid_focal_loss(x,y,alpha=-1,gamma=2,reduction='none');expected=(ref[0]+ref[1:3].mean())/2
        loss=balanced_objectness(x,valid,pos,gamma=2);torch.testing.assert_close(loss,expected);loss.backward();self.assertTrue(torch.isfinite(x.grad).all());self.assertEqual(float(x.grad[3]),0.)
    def test_unknown_negatives_ignored_and_empty_finite(self):
        x=torch.tensor([2.,3.],requires_grad=True);v=torch.ones(2,dtype=torch.bool);p=torch.tensor([],dtype=torch.long)
        loss=balanced_objectness(x,v,p,gamma=2,annotation_complete=False);self.assertEqual(float(loss),0);loss.backward();self.assertEqual(float(x.grad.abs().sum()),0)
    def test_gamma0_original_exact_and_matching_unchanged(self):
        fixture=ProposalTests();fixture.setUp();out,_,_=fixture.output();args=fixture.target()
        from mousegeoforge.models.proposal_instance_grouping import proposal_grouping_loss
        original=proposal_grouping_loss(out,*args);control=grouping_objectness_loss(out,*args,gamma=0);focal=grouping_objectness_loss(out,*args,gamma=2)
        for key in original:self.assertTrue(torch.equal(original[key],control[key]))
        for key in ['assignment','matched_queries','matched_targets','missing_center_proposals']:self.assertTrue(torch.equal(original[key],focal[key]))
    def test_hard_negative_gradient_exceeds_easy(self):
        x=torch.tensor([3.,-3.],requires_grad=True);loss=balanced_objectness(x,torch.ones(2,dtype=torch.bool),torch.tensor([],dtype=torch.long),gamma=2);loss.backward();self.assertGreater(abs(float(x.grad[0])),abs(float(x.grad[1]))*10)

if __name__=='__main__':unittest.main()
