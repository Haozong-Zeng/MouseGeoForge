import unittest
import torch
from mousegeoforge.part_balanced_depth import keypoint_regions, part_balanced_depth


class PartBalanceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_off_is_exact_baseline_and_keeps_empty_views_out_of_mean(self):
        e=torch.rand(3,12,12,requires_grad=True);v=torch.ones_like(e,dtype=torch.bool);v[1]=False
        r=torch.rand(3,5,12,12)
        got=part_balanced_depth(e,v,r,strength=0)
        expected=torch.stack([e[0].mean(),e[2].mean()]).mean()
        torch.testing.assert_close(got['loss'],expected,rtol=0,atol=0)
        got['loss'].backward();self.assertEqual(float(e.grad[1].sum()),0)

    def test_small_part_has_more_influence_but_gradient_is_bounded(self):
        e=torch.ones(1,16,16,requires_grad=True);e.data[:,:2,:4]=8
        r=torch.zeros(1,1,16,16);r[:,:,:2,:4]=1
        out=part_balanced_depth(e,torch.ones_like(e,dtype=torch.bool),r,strength=10,max_pixel_boost=4)
        self.assertGreater(float(out['loss'].detach()),float(out['baseline'].detach()))
        out['loss'].backward()
        self.assertGreater(float(e.grad[0,0,0]),float(e.grad[0,8,8]))
        self.assertLessEqual(float(e.grad.max())*256,4.000001)
        self.assertLessEqual(float(out['gradient_ratio'].max()),4.000001)

    def test_each_region_averaged_before_combination(self):
        e=torch.ones(1,10,10);e[:,:2,:4]=5
        r=torch.zeros(1,2,10,10);r[:,0,:2,:4]=1;r[:,1,2:,:]=1
        got=part_balanced_depth(e,torch.ones_like(e,dtype=torch.bool),r)
        self.assertAlmostEqual(float(got['part_mean']),3)

    def test_missing_and_tiny_regions_fall_back_without_dropping_sample(self):
        e=torch.rand(2,8,8);r=torch.zeros(2,3,8,8);r[0,0,0,0]=1
        out=part_balanced_depth(e,torch.ones_like(e,dtype=torch.bool),r)
        self.assertFalse(bool(out['active_regions'].any()))
        torch.testing.assert_close(out['loss'],e.mean())
        self.assertTrue(bool(out['valid_views'].all()))

    def test_invalid_nan_padding_empty_batch_and_no_roi_gradients(self):
        e=torch.full((1,8,8),float('nan'),requires_grad=True);v=torch.zeros_like(e,dtype=torch.bool)
        r=torch.ones(1,2,8,8,requires_grad=True)
        out=part_balanced_depth(e,v,r);out['loss'].backward()
        self.assertEqual(float(out['loss'].detach()),0);self.assertTrue(torch.isfinite(e.grad).all())
        self.assertEqual(float(e.grad.abs().sum()),0);self.assertIsNone(r.grad)

    def test_confidence_controls_support_and_cannot_receive_gradients(self):
        e=torch.ones(1,8,8,requires_grad=True);e.data[0,0,0]=100
        c=torch.ones_like(e,requires_grad=True);c.data[0,0,0]=0
        r=torch.ones(1,1,8,8)
        out=part_balanced_depth(e,torch.ones_like(e,dtype=torch.bool),r,confidence=c)
        self.assertAlmostEqual(float(out['loss'].detach()),1)
        out['loss'].backward();self.assertEqual(float(e.grad[0,0,0]),0);self.assertIsNone(c.grad)

    def test_part_weights_and_minimum_support(self):
        e=torch.ones(1,8,8);e[:,:2,:]=9
        r=torch.zeros(1,2,8,8);r[:,0,:2,:]=1;r[:,1,2:,:]=1
        out=part_balanced_depth(e,torch.ones_like(e,dtype=torch.bool),r,part_weights=[0,1])
        self.assertAlmostEqual(float(out['part_mean']),1)
        for bad in [[0,0],[-1,1],[1], [float('nan'),1]]:
            with self.assertRaises(ValueError):part_balanced_depth(e,torch.ones_like(e,dtype=torch.bool),r,part_weights=bad)

    def test_valid_nonfinite_and_negative_errors_rejected(self):
        for value in [float('nan'),float('inf'),-1]:
            e=torch.full((1,8,8),value)
            with self.assertRaises(ValueError):part_balanced_depth(e,torch.ones_like(e,dtype=torch.bool),torch.ones(1,1,8,8))

    def test_roi_visibility_crop_coordinates_and_detach(self):
        pts=torch.tensor([[[5.,6.],[float('nan'),0.],[16.,4.],[3.,4.]]],requires_grad=True)
        visible=torch.tensor([[True,True,True,False]])
        r=keypoint_regions(pts,visible,torch.tensor([[0.,0.,16.,16.]]),(16,16),radius_fraction=.25)
        self.assertEqual(float(r[0,0,6,5]),1)
        self.assertEqual(float(r[:,1:].sum()),0)
        self.assertFalse(r.requires_grad)
        with self.assertRaises(ValueError):keypoint_regions(pts,visible.float(),torch.tensor([[0.,0.,16.,16.]]),(16,16))

    def test_rois_do_not_create_a_depth_target(self):
        pts=torch.tensor([[[4.,4.]]]);v=torch.tensor([[True]])
        r=keypoint_regions(pts,v,torch.tensor([[0.,0.,8.,8.]]),(8,8),radius_fraction=.5)
        error=torch.zeros(1,8,8,requires_grad=True)
        out=part_balanced_depth(error,torch.ones_like(error,dtype=torch.bool),r)
        out['loss'].backward();self.assertEqual(float(out['loss'].detach()),0)

    def test_positive_focus_auxiliary_preserves_global_silog_off(self):
        e=torch.rand(1,16,16,requires_grad=True);v=torch.ones_like(e,dtype=torch.bool)
        r=torch.zeros(1,1,16,16);r[:,:,:2,:4]=1
        off=part_balanced_depth(e,v,r,strength=0)
        self.assertEqual(float(off['focus_loss'].detach()),0)
        on=part_balanced_depth(e,v,r,strength=10,max_pixel_boost=4)
        self.assertGreaterEqual(float(on['focus_loss'].detach()),0)
        on['focus_loss'].backward()
        self.assertLessEqual(float(e.grad.max())*256,3.000001)
        self.assertEqual(float(e.grad[0,8,8]),0)


if __name__=='__main__':unittest.main()
