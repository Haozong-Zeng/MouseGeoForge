import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from analyze_cylinder_view_balance import gate_results
from train_cylinder_view_balance import CONFIG


class CylinderGateTests(unittest.TestCase):
    def fixture(self):
        metrics = dict(bottom_body_keypoint_nme=1.,top_body_keypoint_nme=1.,
                       box_matched_nme=1.,detection_aware_pck_0_1=.6,
                       body_depth_absrel=.1,bottom_body_depth_absrel=.1)
        metrics['detection_aware_pck_0.1']=metrics.pop('detection_aware_pck_0_1')
        groups=('bottom_only','side_bottom','sides_bottom','sides_bottom_top',
                'old_dual_pair','old_cylinder_side_pair','top_only','sides_top')
        base={g:dict(metrics) for g in groups}
        control,balanced=copy.deepcopy(base),copy.deepcopy(base)
        for g in ('bottom_only','side_bottom'):
            control[g]['bottom_body_keypoint_nme']=.9
            balanced[g]['bottom_body_keypoint_nme']=.8
        return dict(config=json.loads(CONFIG.read_text()),baseline=dict(summary=base),
                    arms=dict(side_control=dict(summary=control),balanced_views=dict(summary=balanced)))

    def test_expected_gates_pass(self):
        report=gate_results(self.fixture())
        self.assertTrue(report['balanced_views']['passed'])
        self.assertEqual(report['balanced_views']['total'],17)

    def test_original_pair_regression_not_hidden_by_bottom_gain(self):
        result=self.fixture()
        result['arms']['balanced_views']['summary']['old_dual_pair']['box_matched_nme']=1.06
        self.assertFalse(gate_results(result)['balanced_views']['passed'])

    def test_improvement_must_exceed_equal_budget_control(self):
        result=self.fixture()
        result['arms']['balanced_views']['summary']['bottom_only']['bottom_body_keypoint_nme']=.88
        self.assertFalse(gate_results(result)['balanced_views']['passed'])

    def test_missing_metric_does_not_pass(self):
        result=self.fixture()
        result['arms']['balanced_views']['summary']['bottom_only']['bottom_body_keypoint_nme']=None
        self.assertFalse(gate_results(result)['balanced_views']['passed'])


if __name__=='__main__': unittest.main()
