import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from train_variable_view_replay import make_schedule


class SeedReplicationTests(unittest.TestCase):
    def test_fresh_seeds_and_final_stage(self):
        cfg=json.loads((ROOT/'configs/experiments/variable_view_seed_v4.json').read_text())
        self.assertEqual(cfg['arm_seeds'],{'replay_seed29':29,'replay_seed43':43})
        self.assertEqual(cfg['learning_rate'],3e-6)
        self.assertEqual(cfg['steps']*len(cfg['arms']),1600)
        self.assertEqual(cfg['followup_stage'],2)
        self.assertEqual(cfg['remaining_followup_stages_after_this'],0)
        self.assertNotEqual(make_schedule(800,29),make_schedule(800,43))
        old=json.loads((ROOT/'configs/experiments/variable_view_lr_v3.json').read_text())
        for key in ('steps','train_scenes','development_scenes','single_train_per_source',
                    'long_side','encoder_frozen','coupling','checkpoint','manifest','acceptance'):
            self.assertEqual(cfg[key],old[key])


if __name__=='__main__': unittest.main()
