import unittest
import torch
from mousegeoforge.coverage_objectness import coverage_objectness_targets
from mousegeoforge.roi_set_objectness import set_objectness_targets


class CoverageObjectnessTests(unittest.TestCase):
    def test_score_biased_assignment_can_strand_valid_candidate(self):
        boxes=torch.tensor([[[.58,.5,.2,.2],[.5,.5,.2,.2]]])
        target=boxes[:,1:2].clone();valid=torch.ones(1,1,dtype=torch.bool)
        old=set_objectness_targets(boxes,torch.tensor([[1.,0.]]),target,valid)
        self.assertEqual(int(old.sum()),0)
        self.assertEqual(coverage_objectness_targets(boxes,target,valid).tolist(),[[0.,1.]])

    def test_two_targets_are_unique_and_permutation_equivariant(self):
        boxes=torch.tensor([[[.2,.2,.2,.2],[.7,.7,.2,.2],[.21,.2,.2,.2]]])
        targets=boxes[:,:2].clone();valid=torch.ones(1,2,dtype=torch.bool)
        label=coverage_objectness_targets(boxes,targets,valid)
        self.assertEqual(label.tolist(),[[1.,1.,0.]])
        order=torch.tensor([2,0,1])
        torch.testing.assert_close(coverage_objectness_targets(boxes[:,order],targets,valid),label[:,order])

    def test_empty_targets(self):
        label=coverage_objectness_targets(torch.ones(2,6,4),torch.empty(2,0,4),torch.empty(2,0,dtype=torch.bool))
        self.assertEqual(int(label.sum()),0)

    def test_gt_and_boxes_receive_no_gradients(self):
        boxes=torch.tensor([[[.5,.5,.2,.2]]],requires_grad=True)
        labels=coverage_objectness_targets(boxes,boxes.clone(),torch.ones(1,1,dtype=torch.bool))
        self.assertFalse(labels.requires_grad)


if __name__=='__main__':unittest.main()
