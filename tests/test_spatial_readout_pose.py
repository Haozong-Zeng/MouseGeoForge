import unittest
from unittest.mock import patch
import torch
from torch import nn
from mousegeoforge.models.spatial_readout_pose import SpatialReadoutDepthPoseModel,_cudnn_tf32_only
from mousegeoforge.models.surface_curve_pose import SurfaceCurveDepthPoseModel


class Observe(nn.Module):
    def forward(self, query, memory=None):return query


def make_model():
    dense=nn.Module();dense.norm=nn.LayerNorm(8);dense.intermediate_layer_idx=(4,11,17,23);dense.proj=nn.Conv2d(256,1,1)
    head=nn.Module();head.intermediate_layer_idx=dense.intermediate_layer_idx;head.hidden_dim=256;head.patch_size=16
    head.instance_decoder=Observe();head.keypoint_decoder=Observe()
    return SpatialReadoutDepthPoseModel(nn.Identity(),dense,head)


def fake_decode(model,images,cached,patch_start,*,coupling=None):
    model.instance_pose_head.instance_decoder(cached['instance_tokens'],cached['memory'])
    model.instance_pose_head.keypoint_decoder(cached['tokens'])
    if cached.get('fail'):raise RuntimeError('injected')
    return cached['pred']


class SpatialWrapperTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(17)

    def inputs(self,views,batch=1):
        images=torch.randn(batch,views,3,32,48);q,k=2,23
        data=dict(memory=torch.randn(batch*views,6,256),tokens=torch.randn(batch*views,q*k,256),instance_tokens=torch.randn(batch*views,q,256),pred={
            'instance_keypoint_coarse_xy':torch.rand(batch,views,q,k,2),
            'instance_keypoint_xy':torch.rand(batch,views,q,k,2),
            'instance_boxes_cxcywh':torch.rand(batch,views,q,4),
            'instance_objectness_logits':torch.randn(batch,views,q),
            'depth':torch.randn(batch,views,32,48)})
        return images,data

    def test_disabled_returns_exact_original_dictionary(self):
        model=make_model();images,data=self.inputs(2)
        with patch.object(SurfaceCurveDepthPoseModel,'_decode',fake_decode):out=model.forward_cached(images,data,0)
        self.assertIs(out,data['pred']);self.assertNotIn('instance_keypoint_original_xy',out)

    def test_all_queries_one_two_seven_views_and_batches(self):
        model=make_model();model.spatial_pose_enabled=True
        for views,batch in [(1,1),(2,2),(7,1)]:
            images,data=self.inputs(views,batch)
            with patch.object(SurfaceCurveDepthPoseModel,'_decode',fake_decode):out=model.forward_cached(images,data,0)
            self.assertEqual(out['instance_keypoint_xy'].shape,(batch,views,2,23,2))
            self.assertTrue(torch.isfinite(out['instance_keypoint_xy']).all())
            self.assertIs(out['depth'],data['pred']['depth'])
            self.assertIs(out['instance_boxes_cxcywh'],data['pred']['instance_boxes_cxcywh'])
            torch.testing.assert_close(out['instance_keypoint_original_xy'],data['pred']['instance_keypoint_xy'],atol=0,rtol=0)
            self.assertFalse(model.instance_pose_head.instance_decoder._forward_pre_hooks)
            self.assertFalse(model.instance_pose_head.keypoint_decoder._forward_hooks)

    def test_exception_removes_hooks(self):
        model=make_model();model.spatial_pose_enabled=True;images,data=self.inputs(2);data['fail']=True
        with patch.object(SurfaceCurveDepthPoseModel,'_decode',fake_decode):
            with self.assertRaises(RuntimeError):model.forward_cached(images,data,0)
        self.assertFalse(model.instance_pose_head.instance_decoder._forward_pre_hooks)
        self.assertFalse(model.instance_pose_head.keypoint_decoder._forward_hooks)
        self.assertFalse(model.instance_pose_head.instance_decoder._forward_hooks)

    def test_detector_only_and_combined_all_views(self):
        for pose_enabled in [False,True]:
            model=make_model();model.spatial_detector_enabled=True;model.spatial_pose_enabled=pose_enabled
            for views,batch in [(1,1),(2,2),(7,1)]:
                images,data=self.inputs(views,batch)
                data['pred']['instance_boxes_cxcywh']=.05+.9*data['pred']['instance_boxes_cxcywh']
                with patch.object(SurfaceCurveDepthPoseModel,'_decode',fake_decode):out=model.forward_cached(images,data,0)
                torch.testing.assert_close(out['instance_boxes_cxcywh'],data['pred']['instance_boxes_cxcywh'],atol=1e-6,rtol=0)
                self.assertIs(out['instance_boxes_original_cxcywh'],data['pred']['instance_boxes_cxcywh'])
                self.assertIs(out['depth'],data['pred']['depth'])
                if not pose_enabled:self.assertIs(out['instance_keypoint_xy'],data['pred']['instance_keypoint_xy'])
                self.assertFalse(model.instance_pose_head.instance_decoder._forward_hooks)

    def test_unvalidated_joint_mode_rejected_only_when_enabled(self):
        model=make_model();images,data=self.inputs(1)
        with patch.object(SurfaceCurveDepthPoseModel,'_decode',fake_decode):model.forward_cached(images,data,0,coupling='local_depth_features')
        model.spatial_pose_enabled=True
        with self.assertRaises(ValueError):model.forward_cached(images,data,0,coupling='local_depth_features')

    def test_precision_context_preserves_cudnn_enabled_and_restores(self):
        original=(torch.backends.cudnn.enabled,torch.backends.cudnn.benchmark,torch.backends.cudnn.deterministic,torch.backends.cudnn.allow_tf32)
        with self.assertRaises(RuntimeError):
            with _cudnn_tf32_only(not original[3]):
                self.assertEqual((torch.backends.cudnn.enabled,torch.backends.cudnn.benchmark,torch.backends.cudnn.deterministic),original[:3])
                self.assertEqual(torch.backends.cudnn.allow_tf32,not original[3])
                raise RuntimeError('injected')
        self.assertEqual((torch.backends.cudnn.enabled,torch.backends.cudnn.benchmark,torch.backends.cudnn.deterministic,torch.backends.cudnn.allow_tf32),original)


if __name__=='__main__':unittest.main()
