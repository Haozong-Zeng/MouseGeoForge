import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from train_variable_view_adaptation import spaced, mean_metrics
from analyze_variable_view_adaptation import assess


class AdaptationTests(unittest.TestCase):
    def config(self):
        return json.loads((Path(__file__).resolve().parents[1]/'configs/experiments/variable_view_adaptation_v1.json').read_text())

    def test_bounded_config(self):
        cfg=self.config()
        self.assertEqual(cfg['steps']*len(cfg['arms']),1200)
        self.assertEqual(cfg['long_side'],768)
        self.assertTrue(cfg['encoder_frozen'])
        self.assertEqual(cfg['geometry_loss_weights'],0)

    def test_spaced_selection_unique_and_ends(self):
        rows=spaced(list(range(200)),32)
        self.assertEqual(len(set(rows)),32)
        self.assertEqual((rows[0],rows[-1]),(0,199))
        with self.assertRaises(ValueError): spaced([1],32)

    def test_groups_do_not_mix_view_counts(self):
        result=mean_metrics([dict(group='V1',metrics={'nme':1.}),dict(group='V4',metrics={'nme':9.}),dict(group='V1',metrics={'nme':3.})])
        self.assertEqual(result,{'V1':{'nme':2.},'V4':{'nme':9.}})

    def fixture(self):
        summary={g:dict(box_matched_nme=.1,detection_aware_pck_0_1=.7,body_depth_absrel=.2) for g in ('V1','V2','V3','V4','fixed_pair')}
        for row in summary.values(): row['detection_aware_pck_0.1']=row.pop('detection_aware_pck_0_1')
        variable=copy.deepcopy(summary)
        for group in ('V1','V3','V4'): variable[group]['box_matched_nme']=.09
        return dict(config=self.config(),arms=dict(fixed_two=dict(summary=summary),variable_one_to_four=dict(summary=variable)))

    def test_improvement_does_not_promote_production(self):
        result=assess(self.fixture())
        self.assertTrue(result['development_gate_passed'])
        self.assertFalse(result['production_promoted'])

    def test_single_view_or_fixed_pair_regression_fails(self):
        for group in ('V1','fixed_pair'):
            result=self.fixture(); result['arms']['variable_one_to_four']['summary'][group]['box_matched_nme']=.2
            self.assertFalse(assess(result)['development_gate_passed'])


if __name__=='__main__': unittest.main()
