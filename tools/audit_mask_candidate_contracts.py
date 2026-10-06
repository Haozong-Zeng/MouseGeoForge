"""Finite CPU contract tests and grouped recorded timing; no model forward/training."""
from collections import defaultdict
import io
import json
import unittest
import numpy as np
from run_depth_pose_ablation import ROOT,torch,write_json,fingerprint
from mousegeoforge.mask_box_candidate import file_sha256

OUT=ROOT/'output/mask_candidate_contracts_v1'
BUNDLES=ROOT/'output/mask_box_candidates_v1'


def main():
    if (OUT/'report.json').exists():raise RuntimeError('Existing contract audit')
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    inputs={str(p):fingerprint(p) for p in [BUNDLES/'verification.json',BUNDLES/'loading_guard_verification.json',BUNDLES/'latencies.json',*[BUNDLES/s/n for s in ('seed17','seed29') for n in ('manifest.json','refiner.pt')]]}
    assert all(json.loads((BUNDLES/n).read_text())['passed'] for n in ('verification.json','loading_guard_verification.json'))
    checked={}
    for seed in ('seed17','seed29'):
        m=json.loads((BUNDLES/seed/'manifest.json').read_text())
        assert file_sha256(BUNDLES/seed/m['weights'])==m['weights_sha256']
        for path,h in m['source_sha256'].items():assert file_sha256(ROOT/path)==h
        for key in ('base_checkpoint','encoder_checkpoint'):
            p=ROOT/m[key]
            if str(p) not in checked:checked[str(p)]=file_sha256(p)
            assert checked[str(p)]==m[key+'_sha256']
    stream=io.StringIO();suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    result=unittest.TextTestRunner(stream=stream,verbosity=1).run(suite)
    tests=dict(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),passed=result.wasSuccessful(),output=stream.getvalue())
    write_json(OUT/'test_results.json',tests)
    if not result.wasSuccessful():raise RuntimeError('Contract tests failed')
    timings=json.loads((BUNDLES/'latencies.json').read_text());groups=defaultdict(list)
    for r in timings:groups[(r['views'],*r['image_hw'])].append(r['postprocess_ms'])
    grouped=[dict(views=k[0],height=k[1],width=k[2],n=len(v),median_ms=float(np.median(v)),p95_ms=float(np.quantile(v,.95)),min_ms=float(min(v)),max_ms=float(max(v))) for k,v in sorted(groups.items())]
    assert sum(x['n'] for x in grouped)==176
    from pathlib import Path
    assert all(fingerprint(Path(p))==h for p,h in inputs.items())
    report=dict(passed=True,tests=tests,latency_groups=grouped,artifact_sha256=inputs,base_encoder_sha256=checked,
        source_sha256={p:fingerprint(ROOT/p) for p in ('tools/audit_mask_candidate_contracts.py','tests/test_mask_candidate_boundaries.py')},
        model_forwards=0,optimizer_steps=0,real_artifacts_mutated=False,production_promoted=False,
        limitations=['Finite explicit contract tests, not exhaustive fuzzing or a security certification.',
            'Corruption tests use disposable synthetic fixtures, never modify the real candidates.',
            'Recorded timing pools both seeds by V/H/W; observational single-run measurements, not controlled throughput scaling.',
            'Postprocess only including transfers; excludes base forward, loading and context hash checks.'])
    write_json(OUT/'report.json',report);print(json.dumps(dict(passed=True,tests_run=result.testsRun,latency_groups=grouped),indent=2))


if __name__=='__main__':main()
