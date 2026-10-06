import copy
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from evaluate_view_redundancy import view_indices, verify_shared_targets
from run_depth_pose_ablation import load_batch


class ViewRedundancyTests(unittest.TestCase):
    def pair(self):
        fields = dict(keypoint_xy=torch.arange(16.).reshape(1,4,1,2,2),
            keypoint_valid=torch.ones(1,4,1,2,dtype=torch.bool),
            keypoint_visible=torch.ones(1,4,1,2,dtype=torch.bool),
            instance_ids=torch.zeros(1,4,1,dtype=torch.long),
            intrinsics=torch.eye(3).repeat(1,4,1,1),
            extrinsics=torch.eye(4).repeat(1,4,1,1),
            keypoint_world_xyz=torch.ones(1,1,2,3), world_valid=torch.ones(1,1,2,dtype=torch.bool))
        fields['extrinsics'][...,0,3] = torch.arange(4.)
        old = dict(actual_view_ids=['2,0'], scale=torch.tensor([2.]), targets={})
        new = dict(actual_view_ids=['0,1,2,3'], scale=torch.tensor([5.]), targets=copy.deepcopy(fields))
        for key,value in fields.items():
            old['targets'][key] = value[:,[2,0]].clone() if key not in ('keypoint_world_xyz','world_valid') else value.clone()
        # Both represent identical native coordinates despite different scene scaling.
        for item in (old,new):
            item['targets']['extrinsics'][...,:3,3] /= item['scale'][0]
            item['targets']['keypoint_world_xyz'] /= item['scale'][0]
        return old,new

    def test_physical_camera_mapping_not_tensor_index(self):
        old,new = self.pair()
        self.assertEqual(view_indices(old,[0,2]),[1,0])
        self.assertEqual(view_indices(new,[2,0]),[2,0])

    def test_duplicate_and_missing_views_rejected(self):
        old,_ = self.pair()
        for cameras in ([2,2],[1]):
            with self.assertRaises(ValueError): view_indices(old,cameras)
        old['actual_view_ids']=['2,2']
        with self.assertRaises(ValueError): view_indices(old,[2])

    def test_native_units_and_view_permutation(self):
        verify_shared_targets(*self.pair())

    def test_changed_camera_rejected(self):
        old,new = self.pair()
        new['targets']['extrinsics'][0,2,0,3] += .1
        with self.assertRaises(AssertionError): verify_shared_targets(old,new)

    def test_changed_labels_rejected(self):
        old,new = self.pair()
        new['targets']['keypoint_xy'][0,2] += .1
        with self.assertRaises(AssertionError): verify_shared_targets(old,new)

    def test_default_stays_explicitly_two_views(self):
        with self.assertRaises(ValueError): load_batch(dict(views=[0,1,2,3]))
        with self.assertRaises(ValueError): load_batch(dict(views=[0,0]),num_views=2)

    def test_four_view_model_shape(self):
        from test_depth_pose import tiny_model
        model = tiny_model().eval()
        with torch.no_grad(): result = model(torch.rand(1,4,3,64,64))
        self.assertEqual(result['instance_keypoint_xy'].shape[:2],(1,4))
        self.assertTrue(torch.isfinite(result['instance_keypoint_xy']).all())


if __name__=='__main__': unittest.main()
