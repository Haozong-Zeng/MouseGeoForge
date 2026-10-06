import unittest
from unittest.mock import patch
import torch
from mousegeoforge.replay_retention import build_teacher_targets,matched_replay_retention,replay_retention_pose_loss,upstream


class ReplayRetentionTests(unittest.TestCase):
    def sample(self):
        boxes=torch.tensor([[[[.25,.5,.2,.4],[.75,.5,.2,.4]]]],requires_grad=True)
        xy=boxes[...,:2].detach().unsqueeze(-2).expand(1,1,2,23,2).clone().requires_grad_()
        p=dict(instance_boxes_cxcywh=boxes,instance_keypoint_xy=xy,instance_objectness_logits=torch.ones(1,1,2)*8)
        t=dict(instance_ids=torch.tensor([[[10,20]]]),box_valid=torch.ones(1,1,2,dtype=torch.bool),
            boxes_cxcywh=boxes.detach().clone(),keypoint_valid=torch.ones(1,1,2,23,dtype=torch.bool))
        m=[[(torch.tensor([0,1]),torch.tensor([0,1]))]]
        return p,t,m

    def test_identical_initialization_zero(self):
        p,t,m=self.sample();teacher=build_teacher_targets(p,t,m)
        loss,c=matched_replay_retention(p,t,m,teacher);self.assertEqual(float(loss.detach()),0)
        loss.backward();self.assertEqual(float(p['instance_keypoint_xy'].grad.abs().sum()),0)
        self.assertEqual(c['retained_points'],46)

    def test_query_permutation_identity_alignment(self):
        p,t,m=self.sample();teacher=build_teacher_targets(p,t,m)
        p2={k:v.flip(2) for k,v in p.items()};m2=[[(torch.tensor([1,0]),torch.tensor([0,1]))]]
        loss,c=matched_replay_retention(p2,t,m2,teacher);self.assertEqual(float(loss.detach()),0)

    def test_quality_gate_and_missing_labels(self):
        p,t,m=self.sample();p['instance_objectness_logits'][0,0,0]=-8
        with torch.no_grad():p['instance_boxes_cxcywh'][0,0,1,0]=.1
        teacher=build_teacher_targets(p,t,m);loss,c=matched_replay_retention(p,t,m,teacher)
        self.assertEqual(c['retained_instance_views'],0);self.assertEqual(float(loss.detach()),0)

    def test_teacher_stop_gradient_tail_included(self):
        p,t,m=self.sample();teacher=build_teacher_targets(p,t,m)
        teacher[0][0][10]['xy'].requires_grad_()
        with torch.no_grad():p['instance_keypoint_xy'][0,0,0,7:14]+=.05
        loss,c=matched_replay_retention(p,t,m,teacher);loss.backward()
        self.assertIsNone(teacher[0][0][10]['xy'].grad)
        self.assertGreater(float(p['instance_keypoint_xy'].grad[...,7:14,:].abs().sum()),0)

    def test_empty_nonreplay_and_duplicates(self):
        p,t,m=self.sample();loss,c=matched_replay_retention(p,t,m,None)
        self.assertEqual(float(loss.detach()),0)
        t['instance_ids'][:]=10
        with self.assertRaises(ValueError):build_teacher_targets(p,t,m)

    def test_zero_weight_one_student_match(self):
        p,t,m=self.sample();namespace=dict(hungarian_match_instances=None)
        exec('def stub(predictions,targets,weights):\n hungarian_match_instances(predictions,targets)\n return predictions["instance_keypoint_xy"].sum(), {}',namespace)
        with patch.object(upstream,'instance_pose_loss',namespace['stub']),patch.object(upstream,'hungarian_match_instances',return_value=m) as matcher:
            loss,_=replay_retention_pose_loss(p,t,None,aux_weight=0)
            self.assertEqual(matcher.call_count,1);torch.testing.assert_close(loss,p['instance_keypoint_xy'].sum(),atol=0,rtol=0)


if __name__=='__main__':unittest.main()
