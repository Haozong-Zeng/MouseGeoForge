import unittest
import torch
from mousegeoforge.detection_readout_loss import detection_readout_loss,match_detection
from mousegeoforge.detection_negative_control import background_control_loss

class NegativeControlTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.boxes=torch.tensor([[[.2,.3,.2,.2],[.8,.7,.2,.2]],[[.2,.3,.2,.2],[.8,.7,.2,.2]]])
        self.valid=torch.tensor([[True,False],[True,True]])
        self.complete=torch.ones(2,dtype=torch.bool)
    def pred(self):
        boxes=torch.cat([self.boxes,torch.tensor([[[.5,.5,.2,.2]],[[.5,.5,.2,.2]]])],1).clone().requires_grad_(True)
        return dict(instance_boxes_cxcywh=boxes,instance_objectness_logits=torch.zeros(2,3,requires_grad=True))
    def test_default_exact(self):
        p=self.pred();a,_=detection_readout_loss(p,self.boxes,self.valid,self.complete);b,_=background_control_loss(p,self.boxes,self.valid,self.complete)
        torch.testing.assert_close(a,b,atol=0,rtol=0)
    def test_positive_and_box_gradients_unchanged(self):
        a,b=self.pred(),self.pred();la,_=background_control_loss(a,self.boxes,self.valid,self.complete);lb,_=background_control_loss(b,self.boxes,self.valid,self.complete,additional_negative_weight=.4);la.backward();lb.backward()
        torch.testing.assert_close(a['instance_boxes_cxcywh'].grad,b['instance_boxes_cxcywh'].grad,atol=0,rtol=0)
        matches=match_detection(a,self.boxes,self.valid)
        for i,(qs,_) in enumerate(matches):
            torch.testing.assert_close(a['instance_objectness_logits'].grad[i,qs],b['instance_objectness_logits'].grad[i,qs],atol=0,rtol=0)
            neg=torch.ones(3,dtype=torch.bool);neg[qs]=False
            self.assertTrue((b['instance_objectness_logits'].grad[i,neg]>a['instance_objectness_logits'].grad[i,neg]).all())
        self.assertEqual(len(matches[1][0]),2)
    def test_incomplete_excluded(self):
        p=self.pred();loss,_=background_control_loss(p,self.boxes,self.valid,torch.tensor([False,True]),additional_negative_weight=.4);loss.backward()
        self.assertEqual(float(p['instance_objectness_logits'].grad[0].abs().sum()),0)
    def test_no_complete_is_zero(self):
        p=self.pred();loss,_=background_control_loss(p,self.boxes,self.valid,torch.zeros(2,dtype=torch.bool),additional_negative_weight=.4);self.assertEqual(float(loss.detach()),0);loss.backward()
    def test_empty_scene_negatives(self):
        p=self.pred();loss,logs=background_control_loss(p,self.boxes,torch.zeros_like(self.valid),self.complete,additional_negative_weight=.4);self.assertGreater(float(logs['additional_negative'].detach()),0);loss.backward();self.assertTrue((p['instance_objectness_logits'].grad>0).all())
    def test_invalid_weight(self):
        for w in [-1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):background_control_loss(self.pred(),self.boxes,self.valid,self.complete,additional_negative_weight=w)

if __name__=='__main__':unittest.main()
