from __future__ import annotations

import unittest

import torch

from training.geometry_coupled_losses import (
    depth_to_world_points,
    multiview_keypoint_geometry_loss,
    official_depth_loss,
    official_point_loss,
    relative_rig_camera_loss,
    relative_rig_frame_errors,
    shared_rig_temporal_loss,
    token_matching_loss,
)
from training.finetune_blendedmvs import mean_world_point_distance
from vggt_omega.utils.pose_enc import extri_intri_to_pose_encoding


class GeometryCoupledLossTest(unittest.TestCase):
    def setUp(self) -> None:
        self.height = 32
        self.width = 32
        self.extrinsics = torch.eye(4).view(1, 1, 4, 4).repeat(1, 2, 1, 1)
        self.intrinsics = torch.tensor(
            [[24.0, 0.0, 16.0], [0.0, 24.0, 16.0], [0.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3).repeat(1, 2, 1, 1)
        self.target_depth = torch.full((1, 2, self.height, self.width), 2.0)
        self.valid = torch.ones_like(self.target_depth, dtype=torch.bool)
        self.pose = extri_intri_to_pose_encoding(
            self.extrinsics,
            self.intrinsics,
            (self.height, self.width),
        )

    def test_depth_and_point_losses_have_finite_gradients(self) -> None:
        predicted_depth = (self.target_depth + 0.1).requires_grad_()
        predicted_pose = self.pose.detach().clone().requires_grad_()
        depth_logs = official_depth_loss(
            predicted_depth,
            self.target_depth,
            self.valid,
            gradient_weight=1.0,
        )
        point_logs = official_point_loss(
            predicted_depth,
            predicted_pose,
            self.target_depth,
            self.extrinsics,
            self.intrinsics,
            self.valid,
            (self.height, self.width),
            stride=2,
            gradient_weight=1.0,
        )
        total = depth_logs["loss_depth_official"] + point_logs["loss_point"]
        total.backward()
        self.assertTrue(torch.isfinite(total))
        self.assertTrue(torch.isfinite(predicted_depth.grad).all())
        self.assertTrue(torch.isfinite(predicted_pose.grad).all())

    def test_identity_camera_unprojection(self) -> None:
        world = depth_to_world_points(
            self.target_depth,
            self.extrinsics,
            self.intrinsics,
        )
        self.assertTrue(torch.allclose(world[..., 2], self.target_depth))
        self.assertAlmostEqual(float(world[0, 0, 16, 16, 0]), 0.0, places=6)
        self.assertAlmostEqual(float(world[0, 0, 16, 16, 1]), 0.0, places=6)

    def test_mean_point_scene_unit_scales_linearly(self) -> None:
        depths = self.target_depth[0].numpy()
        masks = self.valid[0].numpy()
        extrinsics = self.extrinsics[0].numpy()
        intrinsics = self.intrinsics[0].numpy()
        base = mean_world_point_distance(
            depths,
            masks,
            extrinsics,
            intrinsics,
            stride=2,
        )
        scaled_extrinsics = extrinsics.copy()
        scaled_extrinsics[:, :3, 3] *= 3.0
        scaled = mean_world_point_distance(
            depths * 3.0,
            masks,
            scaled_extrinsics,
            intrinsics,
            stride=2,
        )
        self.assertAlmostEqual(scaled / base, 3.0, places=5)

    def test_matching_loss_backpropagates_to_tokens(self) -> None:
        tokens = torch.randn(1, 2, 4, 8, requires_grad=True)
        batch = {
            "depths": self.target_depth,
            "depth_masks": self.valid,
            "depth_available": torch.ones((1, 2), dtype=torch.bool),
            "extrinsics": self.extrinsics,
            "intrinsics": self.intrinsics,
        }
        logs = token_matching_loss(
            tokens,
            batch,
            patch_size=16,
            max_pairs=16,
            border=0,
        )
        logs["loss_match"].backward()
        self.assertGreater(float(logs["match_positive_pairs"]), 0.0)
        self.assertTrue(torch.isfinite(tokens.grad).all())

    def test_sparse_geometry_is_zero_for_matching_prediction(self) -> None:
        xy = torch.tensor([[[[16.0, 16.0]], [[16.0, 16.0]]]])
        batch = {
            "images": torch.zeros(1, 2, 3, self.height, self.width),
            "extrinsics": self.extrinsics,
            "intrinsics": self.intrinsics,
            "geometry_world": torch.tensor([[[0.0, 0.0, 2.0]]]),
            "geometry_valid": torch.ones((1, 1), dtype=torch.bool),
            "geometry_xy": xy,
            "geometry_view_valid": torch.ones((1, 2, 1), dtype=torch.bool),
            "geometry_weights": torch.ones((1, 2, 1)),
            "scale": torch.ones(1),
        }
        logs = multiview_keypoint_geometry_loss(
            self.target_depth,
            self.pose,
            batch,
            predicted_rig_weight=1.0,
            fixed_rig_weight=1.0,
            pairwise_weight=1.0,
            reprojection_weight=1.0,
        )
        self.assertLess(float(logs["loss_real_geometry"]), 1e-5)
        self.assertLess(float(logs["geometry_reprojection_error_px"]), 1e-4)

    def test_relative_rig_loss_is_scale_and_gauge_invariant(self) -> None:
        extrinsics = torch.eye(4).view(1, 1, 4, 4).repeat(1, 3, 1, 1)
        extrinsics[0, 1, 0, 3] = -1.0
        extrinsics[0, 2, 1, 3] = -1.5
        intrinsics = self.intrinsics[:, :1].repeat(1, 3, 1, 1)
        scaled = extrinsics.clone()
        scaled[..., :3, 3] *= 3.0
        predicted_pose = extri_intri_to_pose_encoding(
            scaled,
            intrinsics,
            (self.height, self.width),
        )
        batch = {
            "images": torch.zeros(1, 3, 3, self.height, self.width),
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "camera_available": torch.ones((1, 3), dtype=torch.bool),
        }
        matching = relative_rig_camera_loss(predicted_pose, batch)
        self.assertLess(float(matching["loss_relative_rig"]), 1e-5)

        perturbed = scaled.clone()
        angle = torch.tensor(0.12)
        rotation_y = torch.tensor(
            [
                [torch.cos(angle), 0.0, torch.sin(angle)],
                [0.0, 1.0, 0.0],
                [-torch.sin(angle), 0.0, torch.cos(angle)],
            ]
        )
        perturbed[0, 1, :3, :3] = rotation_y
        perturbed_pose = extri_intri_to_pose_encoding(
            perturbed,
            intrinsics,
            (self.height, self.width),
        )
        mismatching = relative_rig_camera_loss(perturbed_pose, batch)
        self.assertGreater(
            float(mismatching["loss_relative_rig_rotation"]),
            float(matching["loss_relative_rig_rotation"]) + 1e-4,
        )
        rotation_error, center_error = relative_rig_frame_errors(
            perturbed_pose,
            batch,
        )
        self.assertGreater(float(rotation_error[0]), 1.0)
        self.assertTrue(torch.isfinite(center_error).all())

    def test_shared_rig_temporal_loss_excludes_low_confidence_frame(self) -> None:
        extrinsics = torch.eye(4).view(1, 1, 4, 4).repeat(3, 3, 1, 1)
        extrinsics[:, 1, 0, 3] = -1.0
        extrinsics[:, 2, 1, 3] = -1.5
        angle = torch.tensor(0.2)
        extrinsics[2, 1, :3, :3] = torch.tensor(
            [
                [torch.cos(angle), 0.0, torch.sin(angle)],
                [0.0, 1.0, 0.0],
                [-torch.sin(angle), 0.0, torch.cos(angle)],
            ]
        )
        intrinsics = self.intrinsics[:, :1].repeat(3, 3, 1, 1)
        pose = extri_intri_to_pose_encoding(
            extrinsics,
            intrinsics,
            (self.height, self.width),
        ).requires_grad_()
        batch = {
            "images": torch.zeros(3, 3, 3, self.height, self.width),
            "rig_frame_confidence": torch.tensor([0.95, 0.96, 0.2]),
            "rig_frame_coverage": torch.ones(3),
        }
        filtered = shared_rig_temporal_loss(
            pose,
            batch,
            frame_mask=torch.tensor([True, True, False]),
            outlier_rotation_deg=180.0,
            outlier_center_error=10.0,
        )
        self.assertLess(float(filtered["loss_shared_rig_temporal"]), 1e-6)
        self.assertEqual(float(filtered["shared_rig_accepted_frames"]), 2.0)

        unfiltered = shared_rig_temporal_loss(
            pose,
            batch,
            frame_mask=torch.ones(3, dtype=torch.bool),
            outlier_rotation_deg=180.0,
            outlier_center_error=10.0,
        )
        self.assertGreater(float(unfiltered["loss_shared_rig_temporal"]), 1e-4)
        unfiltered["loss_shared_rig_temporal"].backward()
        self.assertTrue(torch.isfinite(pose.grad).all())


if __name__ == "__main__":
    unittest.main()
