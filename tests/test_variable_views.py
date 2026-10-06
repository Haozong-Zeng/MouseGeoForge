import tempfile
import unittest
from pathlib import Path

import torch
from mousegeoforge.view_sampling import sample_view_ids, subset_record, view_cache_key
from training.finetune_blendedmvs import BlendedMVSDataset, load_scene
from training.test_blendedmvs_multi_instance_loader import write_scene
from training.test_instance_pose_geometry_losses import _problem
from training.instance_pose_losses import _matched_multiview_geometry_losses
from vggt_omega.utils.robust_pose_geometry import triangulate_robust


class VariableViewTests(unittest.TestCase):
    def test_balanced_counts_and_unique_cameras(self):
        views = [sample_view_ids([0,1,2,3],sample_index=i) for i in range(100)]
        self.assertEqual([sum(len(v)==n for v in views) for n in range(1,5)],[25]*4)
        self.assertTrue(all(len(v)==len(set(v)) for v in views))
        self.assertEqual({v[0] for v in views},set(range(4)))
        self.assertEqual(len({tuple(sorted(v)) for v in views if len(v)==2}),6)

    def test_determinism_and_available_cap(self):
        self.assertEqual(sample_view_ids([0,1,2],sample_index=10),sample_view_ids([2,1,0],sample_index=10))
        self.assertEqual(sample_view_ids([9],sample_index=15,max_views=8),[9])
        self.assertEqual(len(sample_view_ids(range(6),sample_index=5)),6)

    def test_invalid_view_inputs(self):
        for ids in ([],[1,1]):
            with self.assertRaises(ValueError): sample_view_ids(ids,sample_index=0)
        with self.assertRaises(ValueError): sample_view_ids([0],sample_index=-1)
        with self.assertRaises(ValueError): sample_view_ids([0],sample_index=0,max_views=0)

    def test_subset_does_not_mutate_record(self):
        record = dict(views=[0,2],scene='example',split='train')
        result = subset_record(record,[0,1,2,3],sample_index=0,scene_scale=500.)
        self.assertEqual(record['views'],[0,2])
        self.assertEqual(len(result['views']),1)
        self.assertEqual(result['views'],result['diagnostic_view_order'])
        self.assertEqual(result['split'],'train')

    def test_cache_identity_includes_order_context_and_rgb(self):
        args = dict(scene='scene',view_ids=[0,2],rgb_revision='rgb',encoder_sha256='encoder',preprocessing=dict(hw=[432,768]))
        key = view_cache_key(**args)
        for changes in (dict(view_ids=[2,0]),dict(view_ids=[0]),dict(rgb_revision='changed'),dict(encoder_sha256='new')):
            self.assertNotEqual(key,view_cache_key(**dict(args,**changes)))

    def test_ragged_views_match_independent_calls(self):
        from test_depth_pose import tiny_model
        model = tiny_model().eval()
        images = [torch.rand(v,3,64,64) for v in (1,2,3,4,6)]
        with torch.no_grad():
            result = model.forward_ragged(images)
            for image,out in zip(images,result):
                self.assertEqual(out['instance_keypoint_xy'].shape[:2],(1,len(image)))
                torch.testing.assert_close(out['instance_keypoint_xy'],model(image[None])['instance_keypoint_xy'])
        self.assertEqual(len(result),5)
        with self.assertRaises(ValueError): model.forward_ragged([])
        with self.assertRaises(ValueError): model(torch.empty(1,0,3,64,64))

    def test_single_view_backward(self):
        from test_depth_pose import tiny_model
        model = tiny_model().train()
        output = model(torch.rand(1,1,3,64,64))
        loss = output['instance_keypoint_xy'].square().mean() + output['depth'].square().mean()
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.instance_pose_head.parameters()))
        self.assertTrue(all(p.grad is None for p in model.aggregator.parameters()))

    def test_single_view_never_fabricates_triangulation(self):
        result = triangulate_robust(torch.tensor([[30.,20.]]),torch.eye(3,4)[None])
        self.assertFalse(result.supported)
        self.assertTrue(torch.isnan(result.xyz).all())
        pred,target,_,hw = _problem()
        pred = {k:v[:,:1] for k,v in pred.items()}
        target = {k:(v[:,:1] if v.ndim>1 and v.shape[1]==3 else v) for k,v in target.items()}
        matches = [[(torch.tensor([0,1]),torch.tensor([0,1]))]]
        a,b,logs = _matched_multiview_geometry_losses(pred,target,matches,image_size=hw,world_scale=1.)
        self.assertEqual(float(a.detach()),0.)
        self.assertEqual(float(b.detach()),0.)
        self.assertEqual(float(logs['instance_3d_points']),0.)

    def test_fixed_scene_scale_survives_view_dropout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'scene'; write_scene(root,2)
            scene = load_scene(root)
            def sample(n):
                return BlendedMVSDataset([scene],n,96,64,1,17,False,False,False,
                    'median_baseline',2,True,normalization_scale_override=500.)[0]
            a,b = sample(1),sample(2)
            self.assertEqual(float(a['scale']),500.)
            self.assertEqual(float(b['scale']),500.)
            torch.testing.assert_close(a['instance_keypoint_world_xyz'],b['instance_keypoint_world_xyz'])


if __name__=='__main__': unittest.main()
