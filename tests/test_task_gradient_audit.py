import unittest
from unittest.mock import patch
from types import SimpleNamespace
import torch
from mousegeoforge.task_gradient_audit import compare_gradient_sets, matched_task_terms, upstream


class TaskGradientTests(unittest.TestCase):
    def test_opposite_and_zero_gradients(self):
        result=compare_gradient_sets([torch.tensor([1.,0.])]*2,[torch.tensor([-1.,0.])]*3)
        self.assertEqual(result['mean_gradient_cosine'],-1.)
        self.assertEqual(result['negative_pair_fraction'],1.)
        self.assertEqual(result['valid_pairs'],6)
        zero=compare_gradient_sets([torch.zeros(2)],[torch.ones(2)])
        self.assertIsNone(zero['mean_gradient_cosine']);self.assertIsNone(zero['negative_pair_fraction'])

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):compare_gradient_sets([torch.tensor([float('nan')])],[torch.ones(1)])

    def test_single_matching_and_additive_body_tail(self):
        xy=torch.full((1,1,1,23,2),.1,requires_grad=True)
        pred={'instance_keypoint_xy':xy};target={'keypoint_xy':torch.zeros_like(xy),'keypoint_valid':torch.ones(1,1,1,23,dtype=torch.bool)}
        target['keypoint_valid'][0,0,0,3]=False
        namespace=dict(hungarian_match_instances=None,torch=torch)
        exec('def stub(p,t,w):\n hungarian_match_instances(p,t)\n loss=torch.nn.functional.smooth_l1_loss(p["instance_keypoint_xy"][t["keypoint_valid"]],t["keypoint_xy"][t["keypoint_valid"]],beta=.01)\n return 5*loss,{"loss_instance_keypoint_l1":loss}',namespace)
        matches=[[(torch.tensor([0]),torch.tensor([0]))]]
        with patch.object(upstream,'instance_pose_loss',namespace['stub']),patch.object(upstream,'hungarian_match_instances',return_value=matches) as matcher:
            loss,_,parts=matched_task_terms(pred,target,SimpleNamespace(keypoint_l1=5.))
        self.assertEqual(matcher.call_count,1)
        self.assertEqual(parts['valid_points'],22)
        torch.testing.assert_close(parts['body']+parts['tail'],loss)
        self.assertAlmostEqual(float((parts['body']/loss).detach()),15/22,places=6)


if __name__=='__main__':unittest.main()
