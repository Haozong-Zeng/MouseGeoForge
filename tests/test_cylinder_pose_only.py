import sys
from pathlib import Path
import unittest
import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from train_cylinder_pose_only import freeze_except_pose, PoseOnlyDepthPoseModel


class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.aggregator=nn.Linear(3,3)
        self.shared_adapter=nn.Linear(3,3)
        self.dense_head=nn.Linear(3,1)
        self.depth_pose_bridge=nn.Linear(3,3)
        self.instance_pose_head=nn.Linear(3,2)


class PoseOnlyTests(unittest.TestCase):
    def test_only_pose_trainable(self):
        model=SmallModel()
        self.assertEqual(freeze_except_pose(model),['instance_pose_head.weight','instance_pose_head.bias'])

    def test_optimizer_does_not_change_frozen_parameters(self):
        model=SmallModel(); freeze_except_pose(model)
        before={k:v.detach().clone() for k,v in model.state_dict().items()}
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
        x=model.shared_adapter(model.aggregator(torch.ones(2,3)))
        loss=model.instance_pose_head(x).square().sum()+model.dense_head(x).square().sum()
        loss.backward(); optimizer.step()
        self.assertFalse(torch.equal(before['instance_pose_head.weight'],model.instance_pose_head.weight))
        for name,p in model.named_parameters():
            if not name.startswith('instance_pose_head.'):
                self.assertIsNone(p.grad)
                torch.testing.assert_close(before[name],p,atol=0,rtol=0)

    def test_freezing_preserves_state_schema(self):
        model=SmallModel(); keys=set(model.state_dict())
        freeze_except_pose(model)
        self.assertEqual(set(model.state_dict()),keys)


if __name__=='__main__': unittest.main()
