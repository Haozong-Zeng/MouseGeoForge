import importlib.util
from pathlib import Path
import unittest
import sys
import torch
from mousegeoforge.models.calibrated_mouse_depth import CalibratedRayDepth, project_world, ray_bounds, world_rays, supervised_loss


class CalibratedGeometryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.k = torch.tensor([[200., 0, 31.5], [0, 200, 31.5], [0, 0, 1]])
        self.e = torch.eye(4)
        self.e[:3, 3] = torch.tensor([3., 4., 5.])
        self.xy = torch.tensor([[31.5, 31.5], [12., 44.]])
        self.box = torch.tensor([[-50., -50., 40.], [50., 50., 120.]])

    def test_rotated_camera_z_roundtrip(self):
        self.e[:3, :3] = torch.tensor([[0., -1, 0], [1., 0, 0], [0., 0, 1]])
        origin, rays = world_rays(self.xy, self.k, self.e)
        z = torch.tensor([70., 90.])
        world = origin + rays * z[:, None]
        uv, got_z = project_world(world, self.k, self.e)
        torch.testing.assert_close(uv, self.xy)
        torch.testing.assert_close(got_z, z)

    def test_parallel_slabs_and_no_intersection(self):
        _, _, near, far, valid = ray_bounds(self.xy, self.k, self.e, self.box)
        torch.testing.assert_close(near, torch.full_like(near, 45.))
        torch.testing.assert_close(far, torch.full_like(far, 125.))
        self.assertTrue(valid.all())
        _, _, _, _, miss = ray_bounds(self.xy[:1], self.k, self.e, self.box + torch.tensor([1000., 0, 0]))
        self.assertFalse(miss.any())

    def test_metric_plane_recovery_with_analytic_features(self):
        # Three rectified cameras observing a textured frontoparallel plane.
        # Exact synthetic feature maps test source warp and metric depth jointly.
        side = 64; z = 80.
        e = torch.eye(4).repeat(3, 1, 1)
        e[:, 0, 3] = torch.tensor([0., -8., 8.])
        k = self.k.repeat(3, 1, 1)
        y, x = torch.meshgrid(torch.arange(side), torch.arange(side), indexing='ij')
        maps = []
        for camera in range(3):
            world_x = (x-31.5)*z/200 - e[camera, 0, 3]
            world_y = (y-31.5)*z/200
            maps.append(torch.stack([torch.sin(world_x*.4), torch.cos(world_x*.4),
                                     torch.sin(world_y*.3), torch.cos(world_y*.3)]))
        encoded = torch.nn.functional.normalize(torch.stack(maps), dim=1)
        model = CalibratedRayDepth(4, 4, 81)
        model.log_temperature.data.fill_(4.)
        for p in model.score.parameters():
            p.data.zero_()
        box = torch.tensor([[-100., -100., 60.], [100., 100., 100.]])
        prediction, _, valid = model(encoded, self.xy, 0, k, e, box, image_side=64)
        self.assertTrue(valid.all())
        self.assertLess(float((prediction-z).abs().max().detach()), 1.)

    def test_forward_backward_checkpoint_and_no_gt_forward(self):
        model = CalibratedRayDepth(8, 8, 12)
        features = torch.randn(3, 8, 8, 8)
        encoded = model.encode(features)
        pred, conf, valid = model(encoded, self.xy, 0, self.k.repeat(3, 1, 1), self.e.repeat(3, 1, 1), self.box, image_side=64)
        loss = supervised_loss(pred, torch.tensor([80., 85.]), valid)
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None), 0)
        restored = CalibratedRayDepth(8, 8, 12); restored.load_state_dict(model.state_dict())
        got, _, _ = restored(restored.encode(features), self.xy, 0, self.k.repeat(3, 1, 1), self.e.repeat(3, 1, 1), self.box, image_side=64)
        torch.testing.assert_close(pred, got)

    def test_common_train_plateau_rejects_one_stratum_drift(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
        from calibrated_mouse_semseg_campaign_20261025 import plateau
        history=[dict(step=4000+i*500,metrics={group:dict(loss=.5,iou=.9,boundary_f1=.8)
                   for group in ['all','real/True','synthetic/False']}) for i in range(5)]
        self.assertTrue(plateau(history))
        history[-1]['metrics']['real/True']['boundary_f1']=.795
        self.assertFalse(plateau(history))
        self.assertFalse(plateau(history[:4]))

    def test_no_matching_cost_branch_still_trains_reference_projection(self):
        model=CalibratedRayDepth(8,8,12)
        features=torch.randn(3,8,8,8)
        prediction,_,valid=model(model.encode(features),self.xy,0,self.k.repeat(3,1,1),
                                  self.e.repeat(3,1,1),self.box,image_side=64,use_cost=False)
        supervised_loss(prediction,torch.tensor([80.,85.]),valid).backward()
        self.assertGreater(float(model.project[0].weight.grad.abs().sum()),0)


if __name__ == '__main__':
    unittest.main()
