"""Run loader fixture tests with temporary directories, without requiring pytest."""
import sys
import tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'vggt-omega'))
from training import test_blendedmvs_multi_instance_loader as tests

for name in sorted(dir(tests)):
    if name.startswith('test_'):
        with tempfile.TemporaryDirectory(prefix='orbbec-loader-test-') as directory:
            getattr(tests,name)(Path(directory))
        print(f'PASS {name}',flush=True)
