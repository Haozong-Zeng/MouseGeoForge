import copy
import json
from pathlib import Path
import sys
import unittest
from collections import Counter

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from train_variable_view_replay import make_schedule
from analyze_variable_view_replay import assess


class ReplayTests(unittest.TestCase):
    def test_balanced_interleaved_schedule(self):
        schedule=make_schedule(800,17)
        self.assertEqual(schedule,make_schedule(800,17))
        self.assertNotEqual(schedule,make_schedule(800,18))
        self.assertEqual(Counter(k for k,n in schedule),dict(shared=400,dual_anchor=200,single_anchor=200))
        for i in range(0,800,16):
            self.assertEqual(Counter(n for k,n in schedule[i:i+16]),{1:4,2:4,3:4,4:4})
            self.assertEqual(Counter(n for k,n in schedule[i:i+16] if k=='shared'),{1:2,2:2,3:2,4:2})
        with self.assertRaises(ValueError): make_schedule(15,17)

    def fixture(self):
        cfg=json.loads((Path(__file__).resolve().parents[1]/'configs/experiments/variable_view_replay_v2.json').read_text())
        groups=('V1','V2','V3','V4','fixed_pair','single_original_pair')
        summary={g:{'box_matched_nme':.1,'detection_aware_pck_0.1':.7,'body_depth_absrel':.2} for g in groups}
        adapted=copy.deepcopy(summary)
        for g in ('V1','V3','V4'): adapted[g]['box_matched_nme']=.08
        return dict(config=cfg,baseline=dict(summary=summary),arms=dict(mixed_replay=dict(summary=adapted)))

    def test_reference_is_original(self):
        result=assess(self.fixture())
        self.assertTrue(result['arms']['mixed_replay']['development_gate_passed'])
        self.assertEqual(result['reference'],'original checkpoint')
        self.assertFalse(result['production_promoted'])

    def test_single_and_pair_regression_fail(self):
        for g in ('fixed_pair','single_original_pair'):
            for metric,value in (('box_matched_nme',.12),('detection_aware_pck_0.1',.65),('body_depth_absrel',.25)):
                result=self.fixture(); result['arms']['mixed_replay']['summary'][g][metric]=value
                self.assertFalse(assess(result)['arms']['mixed_replay']['development_gate_passed'])


if __name__=='__main__': unittest.main()
