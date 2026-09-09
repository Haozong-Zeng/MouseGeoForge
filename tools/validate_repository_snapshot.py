"""Small source-snapshot smoke suite; no datasets, checkpoint or pytest needed.

Requires the installed model dependencies including torch, numpy and scipy.
This does not validate the proposed (not yet implemented) temporal architecture.
"""
from pathlib import Path
import importlib
import inspect
import json
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'vggt-omega'))


def main():
    import torch
    torch.set_num_threads(2)
    spec = json.loads((ROOT/'configs/experiments/temporal_ablation_v1.json').read_text())
    assert spec['status'] == 'design_only_not_consumed_by_training_entrypoint'
    assert {r['id'] for r in spec['variants']} == {'A0', 'A1', 'A2', 'A3', 'C0', 'C1'}
    suite = unittest.defaultTestLoader.loadTestsFromName('training.test_temporal_window_index')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    tested = []
    for name in ('training.test_instance_pose_head', 'training.test_instance_pose_geometry_losses'):
        module = importlib.import_module(name)
        for key, value in vars(module).items():
            if key.startswith('test_') and inspect.isfunction(value):
                assert not inspect.signature(value).parameters, 'Use pytest for tests with fixtures'
                value()
                tested.append(f'{name}.{key}')
                print(f'PASS {tested[-1]}', flush=True)
    print(json.dumps({'passed': True, 'window_tests': result.testsRun,
                      'head_geometry_tests': len(tested), 'experiment_spec_valid': True}))


if __name__ == '__main__':
    main()
