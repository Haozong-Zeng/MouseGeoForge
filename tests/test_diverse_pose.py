import copy
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from run_depth_pose_ablation import validate_diverse_records


class TestDiversePose(unittest.TestCase):
    def records(self):
        train=[dict(scene=f't{i}',split='train',source_group='train_source',
                    expected_instances=2 if i<200 else 1) for i in range(400)]
        val=[dict(scene=f'v{i}',split='val',source_group='heldout_source') for i in range(32)]
        return train,val,copy.deepcopy(train[:32]+train[200:232]+val)

    def test_valid_extension(self):
        validate_diverse_records(*self.records())

    def test_no_source_leakage(self):
        train,val,ref=self.records(); train[0]['source_group']='heldout_source'
        with self.assertRaises(AssertionError): validate_diverse_records(train,val,ref)

    def test_no_validation_mutation(self):
        train,val,ref=self.records(); val[0]['scene']='changed'
        with self.assertRaises(AssertionError): validate_diverse_records(train,val,ref)

    def test_reference_training_retained(self):
        train,val,ref=self.records(); train[0]['scene']='new'
        with self.assertRaises(AssertionError): validate_diverse_records(train,val,ref)

    def test_config_keeps_optimizer_and_model(self):
        old=json.loads((ROOT/'configs/experiments/pose_convergence_512_15000_v1.json').read_text())
        new=json.loads((ROOT/'configs/experiments/pose_diverse_core23_512_v1.json').read_text())
        excluded={'purpose','notes','dataset_policy','reference_manifest'}
        self.assertEqual({k:v for k,v in old.items() if k not in excluded},
                         {k:v for k,v in new.items() if k not in excluded})
