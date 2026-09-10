import copy
import json
import sys
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from run_depth_pose_ablation import resize_manifest_records
from pose_diverse_resolution import assert_pair


class TestDiverseResolution(unittest.TestCase):
    def pair(self):
        low=dict(config=dict(long_side=512,steps=15000),
                 source_plan_sha256='plan',grouping_sha256='groups',
                 reference_manifest_sha256='ref',experimental_source_groups={},
                 records=[dict(scene=str(i),image_hw=[288 if i%2 else 512,512]) for i in range(432)])
        high=copy.deepcopy(low)
        high['config']['long_side']=768
        high['records']=resize_manifest_records(low['records'],768)
        return low,high

    def test_pair_and_nonmutating_resize(self):
        low,high=self.pair()
        assert_pair(low,high)
        self.assertEqual(low['records'][1]['image_hw'],[288,512])
        self.assertEqual(high['records'][1]['image_hw'],[432,768])
        self.assertEqual(high['records'][0]['image_hw'],[768,768])
        self.assertEqual(resize_manifest_records(low['records'],512),low['records'])

    def test_no_scene_or_order_change(self):
        low,high=self.pair()
        high['records'].reverse()
        with self.assertRaises(AssertionError): assert_pair(low,high)

    def test_no_aspect_or_camera_change(self):
        for key,value in [('image_hw',[768,768]),('views',[0,1])]:
            low,high=self.pair(); high['records'][1][key]=value
            with self.assertRaises(AssertionError): assert_pair(low,high)

    def test_no_optimizer_change(self):
        low,high=self.pair(); high['config']['steps']=20000
        with self.assertRaises(AssertionError): assert_pair(low,high)

    def test_registered_config_resolution_only(self):
        configs=[json.loads((ROOT/'configs/experiments'/f'pose_diverse_core23_{side}_v1.json').read_text())
                 for side in (512,768)]
        ignored={'purpose','notes','long_side'}
        self.assertEqual(*[{k:v for k,v in c.items() if k not in ignored} for c in configs])

    def test_unregistered_size_rejected(self):
        with self.assertRaises(ValueError): resize_manifest_records([],640)
