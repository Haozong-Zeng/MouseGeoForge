import json
from pathlib import Path
import sys
import unittest
import torch
from mousegeoforge.replay_retention import matched_replay_retention

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from summarize_cylinder_replay_retention import summarize


class ReplayRecoveryTests(unittest.TestCase):
    def test_summary_serializable_and_negative_quality_gate(self):
        q = [dict(checkpoint=t, record=0, view=0, identity=1, iou=iou,
                  body_errors=[body], tail_errors=[tail])
             for t, iou, body, tail in [('teacher', .8, .06, .09), ('student', .9, .07, .08)]]
        g = [dict(checkpoint='student', group=group, ratio=.02)
             for group in ('shared_adapter', 'instance_head')]
        report = summarize(q, g)
        self.assertFalse(report['training_supported'])
        self.assertEqual(report['individual_quality_violations'], dict(box=1, body=0, tail=1))
        self.assertEqual(json.loads(json.dumps(report, allow_nan=False)), report)

    def test_instance_balancing_with_unequal_view_support(self):
        # ID 10 has two qualified views; ID 20 only one. Each animal gets 1/2.
        xy = torch.zeros(1, 2, 2, 23, 2, requires_grad=True)
        boxes = torch.zeros(1, 2, 2, 4, requires_grad=True)
        pred = dict(instance_keypoint_xy=xy, instance_boxes_cxcywh=boxes)
        target = dict(instance_ids=torch.tensor([[[10, 20], [10, 20]]]),
                      keypoint_valid=torch.ones(1, 2, 2, 23, dtype=torch.bool))
        matches = [[(torch.tensor([0, 1]), torch.tensor([0, 1])) for _ in range(2)]]
        def teacher(delta):
            return dict(box=torch.full((4,), delta), xy=torch.zeros(23, 2), valid=torch.ones(23, dtype=torch.bool))
        packed = [[{10: teacher(.1), 20: teacher(.3)}, {10: teacher(.1)}]]
        loss, counts = matched_replay_retention(pred, target, matches, packed)
        self.assertAlmostEqual(float(loss.detach()), (.095 + .295) / 2, places=6)
        self.assertEqual(counts['retained_instance_views'], 3)
        loss.backward()
        self.assertAlmostEqual(float(boxes.grad[0, :, 0].abs().sum()), .5, places=6)
        self.assertAlmostEqual(float(boxes.grad[0, :, 1].abs().sum()), .5, places=6)


if __name__ == '__main__':
    unittest.main()
