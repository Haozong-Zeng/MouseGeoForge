from pathlib import Path
import sys
import unittest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vggt-omega'))
sys.path.insert(0,str(ROOT))
from mousegeoforge.tail_supervision import sample_tail,tail_loss,matched_tail_loss


class TailTests(unittest.TestCase):
    def fixture(self):
        gt=torch.stack((torch.linspace(.1,.7,7),torch.ones(7)*.4),-1)[None]
        return gt,torch.ones(1,7,dtype=torch.bool),torch.tensor([[.4,.4,.8,.5]])

    def test_identity_and_boundary(self):
        gt,valid,box=self.fixture()
        curve,mask=sample_tail(gt,valid)
        torch.testing.assert_close(curve[0,0,0],gt[0,0])
        torch.testing.assert_close(curve[0,-1,-1],gt[0,-1])
        self.assertTrue(mask.all())
        for mode in ('points','curve'):
            self.assertEqual(float(tail_loss(gt,gt,valid,box,(432,768),mode)),0.)

    def test_partial_nan_no_bridge_and_finite_gradient(self):
        gt,valid,box=self.fixture(); valid[0,3]=False; gt[0,3]=float('nan')
        pred=torch.zeros_like(gt,requires_grad=True)
        _,mask=sample_tail(torch.nan_to_num(gt),valid)
        self.assertFalse(mask[0,1:5].any())
        loss=tail_loss(pred,gt,valid,box,(432,768),'curve'); loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertEqual(float(pred.grad[0,3].abs().sum()),0.)

    def test_empty_and_order_sensitive(self):
        gt,valid,box=self.fixture(); pred=gt.flip(-2).clone().requires_grad_()
        for mode in ('points','curve'):
            self.assertGreater(float(tail_loss(pred,gt,valid,box,(432,768),mode).detach()),0)
            loss=tail_loss(pred,gt,torch.zeros_like(valid),box,(432,768),mode)
            self.assertEqual(float(loss.detach()),0)

    def test_pixel_scale_invariance(self):
        gt,valid,box=self.fixture(); pred=gt+.03
        for mode in ('points','curve'):
            a=tail_loss(pred,gt,valid,box,(432,768),mode)
            b=tail_loss(pred,gt,valid,box,(864,1536),mode)
            torch.testing.assert_close(a,b)

    def test_instance_permutation_and_occluded_labels(self):
        xy=torch.zeros(1,2,2,23,2)
        xy[:,:,0]=.25; xy[:,:,1]=.75
        boxes=torch.tensor([.25,.25,.3,.3,.75,.75,.3,.3]).reshape(1,1,2,4).expand(1,2,2,4)
        target=dict(instance_valid=torch.ones(1,2,2,dtype=torch.bool),
                    boxes_cxcywh=boxes,keypoint_xy=xy,keypoint_valid=torch.ones(1,2,2,23,dtype=torch.bool),
                    keypoint_visible=torch.zeros(1,2,2,23,dtype=torch.bool))
        pred=dict(instance_objectness_logits=torch.ones(1,2,2),
                  instance_boxes_cxcywh=boxes,instance_keypoint_xy=(xy+.01).requires_grad_())
        for mode in ('points','curve'):
            loss=matched_tail_loss(pred,target,(432,768),mode)
            permuted={k:v.flip(2) for k,v in pred.items()}
            torch.testing.assert_close(loss,matched_tail_loss(permuted,target,(432,768),mode))
            self.assertGreater(float(loss.detach()),0)
        loss.backward()
        self.assertGreater(float(pred['instance_keypoint_xy'].grad[...,7:14,:].abs().sum()),0)

    def test_curve_controls_receive_finite_gradient(self):
        gt,valid,box=self.fixture(); pred=(gt+.01).requires_grad_()
        tail_loss(pred,gt,valid,box,(432,768),'curve').backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertTrue((pred.grad.abs().sum(-1)>0).all())
