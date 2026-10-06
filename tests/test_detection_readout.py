import unittest
import torch
from mousegeoforge.models.instance_detection_readout import InstanceDetectionReadout
from mousegeoforge.detection_readout_loss import detection_readout_loss,synthetic_detection_retention
from vggt_omega.models.heads.instance_pose_head import MultiInstanceMouseHead


class DetectionReadoutTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(17)
    def test_exact_original_heads_and_frozen_inputs(self):
        head=MultiInstanceMouseHead();model=InstanceDetectionReadout();model.box.load_state_dict(head.box.state_dict());model.objectness.load_state_dict(head.objectness.state_dict())
        x=torch.randn(2,6,256,requires_grad=True);p=model(x);raw=head.box(x)
        torch.testing.assert_close(p['instance_boxes_cxcywh'],torch.cat([raw[...,:2].sigmoid(),.02+.98*raw[...,2:].sigmoid()],-1),atol=0,rtol=0)
        torch.testing.assert_close(p['instance_objectness_logits'],head.objectness(x).squeeze(-1),atol=0,rtol=0)
        p['instance_boxes_cxcywh'].sum().backward();self.assertIsNone(x.grad)
    def test_incomplete_frames_no_negative_loss(self):
        model=InstanceDetectionReadout();p=model(torch.randn(2,6,256));boxes=torch.rand(2,2,4);valid=torch.ones(2,2,dtype=torch.bool)
        loss,_=detection_readout_loss(p,boxes,valid,torch.zeros(2,dtype=torch.bool));self.assertEqual(float(loss.detach()),0);loss.backward();self.assertEqual(float(model.objectness.weight.grad.abs().sum()),0)
    def test_complete_single_dual_finite_gradients(self):
        model=InstanceDetectionReadout();p=model(torch.randn(2,6,256));boxes=torch.tensor([[[.3,.4,.2,.3],[0,0,0,0]],[[.3,.4,.2,.3],[.7,.5,.2,.3]]]);valid=torch.tensor([[True,False],[True,True]])
        loss,_=detection_readout_loss(p,boxes,valid,torch.ones(2,dtype=torch.bool));self.assertTrue(torch.isfinite(loss));loss.backward();self.assertTrue(torch.isfinite(model.box.layers[-1].weight.grad).all())
    def test_invalid_shape(self):
        with self.assertRaises(ValueError):InstanceDetectionReadout()(torch.randn(6,256))
    def test_retention_real_excluded_teacher_detached(self):
        logits=torch.zeros(2,2,requires_grad=True);boxes=torch.zeros(2,2,4,requires_grad=True)
        teacher_logits=torch.tensor([[2.,-2.],[2.,-2.]],requires_grad=True);teacher_boxes=torch.ones(2,2,4,requires_grad=True)
        loss,_=synthetic_detection_retention(dict(instance_objectness_logits=logits,instance_boxes_cxcywh=boxes),teacher_boxes,teacher_logits,torch.tensor([False,True]));loss.backward()
        self.assertEqual(float(logits.grad[0].abs().sum()),0);self.assertEqual(float(boxes.grad[0].abs().sum()),0)
        self.assertGreater(float(logits.grad[1].abs().sum()),0);self.assertGreater(float(boxes.grad[1,0].abs().sum()),0);self.assertEqual(float(boxes.grad[1,1].abs().sum()),0)
        self.assertIsNone(teacher_logits.grad);self.assertIsNone(teacher_boxes.grad)
    def test_retention_identical_stationary(self):
        logits=torch.randn(2,6,requires_grad=True);boxes=torch.rand(2,6,4,requires_grad=True)
        loss,_=synthetic_detection_retention(dict(instance_objectness_logits=logits,instance_boxes_cxcywh=boxes),boxes.clone(),logits.clone(),torch.ones(2,dtype=torch.bool));loss.backward()
        self.assertLess(float(logits.grad.abs().max()),1e-7);self.assertEqual(float(boxes.grad.abs().sum()),0)
    def test_retention_empty_mask(self):
        logits=torch.randn(2,6,requires_grad=True);boxes=torch.rand(2,6,4,requires_grad=True)
        loss,_=synthetic_detection_retention(dict(instance_objectness_logits=logits,instance_boxes_cxcywh=boxes),boxes,logits,torch.zeros(2,dtype=torch.bool));loss.backward();self.assertEqual(float(loss.detach()),0)


if __name__=='__main__':unittest.main()
