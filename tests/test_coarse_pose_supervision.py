import unittest
from unittest.mock import patch
import torch
from mousegeoforge.coarse_pose_supervision import matched_coarse_body_loss,coarse_aux_pose_loss,upstream


class CoarseSupervisionTests(unittest.TestCase):
    def sample(self):
        xy=torch.full((1,1,2,23,2),.5,requires_grad=True)
        pred={'instance_keypoint_coarse_xy':xy}
        gt=torch.zeros((1,1,1,23,2));valid=torch.ones((1,1,1,23),dtype=torch.bool)
        target={'keypoint_xy':gt,'keypoint_valid':valid,'keypoint_visible':torch.zeros_like(valid)}
        matches=[[(torch.tensor([1]),torch.tensor([0]))]]
        return pred,target,matches

    def test_body_occluded_supervised_tail_unmatched_untouched(self):
        p,t,m=self.sample();loss,count=matched_coarse_body_loss(p,t,m);loss.backward()
        self.assertEqual(count,16)
        grad=p['instance_keypoint_coarse_xy'].grad
        self.assertEqual(float(grad[0,0,0].abs().sum()),0.)
        self.assertEqual(float(grad[0,0,1,7:14].abs().sum()),0.)
        self.assertGreater(float(grad[0,0,1,:7].abs().sum()),0.)

    def test_missing_labels_and_empty_matches(self):
        p,t,m=self.sample();t['keypoint_valid'][:]=False
        t['keypoint_xy'][:]=float('nan')
        loss,count=matched_coarse_body_loss(p,t,m)
        self.assertEqual(count,0);self.assertEqual(float(loss.detach()),0.)

    def test_valid_nan_rejected(self):
        p,t,m=self.sample();t['keypoint_xy'][0,0,0,0]=float('nan')
        with self.assertRaises(ValueError):matched_coarse_body_loss(p,t,m)

    def test_single_matching_and_zero_weight_parity(self):
        # Base stub exercises the cloned function lookup and preserved keyword defaults.
        p,t,m=self.sample()
        def base(predictions,targets,weights,*,image_size=None):
            matches=upstream.hungarian_match_instances(predictions,targets)
            return predictions['instance_keypoint_coarse_xy'].sum(),{}
        # Use a global-name lookup just like the real upstream function.
        namespace=dict(base.__globals__,hungarian_match_instances=lambda p,t:m)
        exec('def stub(predictions,targets,weights,*,image_size=None):\n matches=hungarian_match_instances(predictions,targets)\n return predictions["instance_keypoint_coarse_xy"].sum(), {}',namespace)
        with patch.object(upstream,'instance_pose_loss',namespace['stub']),patch.object(upstream,'hungarian_match_instances',return_value=m) as matcher:
            total,logs=coarse_aux_pose_loss(p,t,None,aux_weight=0.,image_size=(768,768))
            self.assertEqual(matcher.call_count,1)
            torch.testing.assert_close(total,p['instance_keypoint_coarse_xy'].sum(),atol=0,rtol=0)


if __name__=='__main__':unittest.main()
