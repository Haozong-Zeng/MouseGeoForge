import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from analyze_variable_view_lr import assess


class LearningRateControlTests(unittest.TestCase):
    def test_only_lr_changes_in_budget_and_supervision(self):
        old=json.loads((ROOT/'configs/experiments/variable_view_replay_v2.json').read_text())
        new=json.loads((ROOT/'configs/experiments/variable_view_lr_v3.json').read_text())
        for key in ('seed','steps','train_scenes','development_scenes','single_train_per_source',
                    'weight_decay','gradient_clip','long_side','encoder_frozen','coupling',
                    'geometry_loss_weights','uncertainty_loss_weight','local_heatmap_weight',
                    'depth_weight','pose_weight','checkpoint','manifest','replay_policy','acceptance'):
            self.assertEqual(old[key],new[key],key)
        self.assertEqual(set(new['arms']),set(new['arm_learning_rates']))
        self.assertEqual(sorted(new['arm_learning_rates'].values()),[3e-6,1e-5])
        self.assertEqual(new['steps']*len(new['arms']),1600)

    def test_retention_gate_not_weakened(self):
        from test_variable_view_replay import ReplayTests
        fixture=ReplayTests().fixture()
        self.assertTrue(assess(fixture)['arms']['mixed_replay']['development_gate_passed'])
        fixture['arms']['mixed_replay']['summary']['fixed_pair']['box_matched_nme']=.12
        self.assertFalse(assess(fixture)['arms']['mixed_replay']['development_gate_passed'])


if __name__=='__main__': unittest.main()
