"""Paired-initialization, equal-update comparison of registered 256 and 512 runs."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_depth_pose_ablation import ROOT,write_json


def main():
    folders=[ROOT/'output'/name for name in ('depth_pose_baseline_5000_v2','depth_pose_resolution_512_v1')]
    reports=[json.loads((p/'results.json').read_text()) for p in folders]
    checks=[json.loads((p/'checkpoint_verification.json').read_text()) for p in folders]
    assert all(c['passed'] and all(r['passed'] and r['validated_scenes']==32 for r in c['runs']) for c in checks)
    runs=[r['runs'][0] for r in reports]
    assert all(len(r['runs'])==1 for r in reports)
    assert all(r['seed']==17 and r['mode']=='off' and r['steps']==5000 for r in runs)
    for key in ('initial_state_sha256','train_order_sha256'):
        assert runs[0][key]==runs[1][key],key
    ignored={'purpose','notes','long_side'}
    configs=[{k:v for k,v in r['config'].items() if k not in ignored} for r in reports]
    assert configs[0]==configs[1]
    assert [r['config']['long_side'] for r in reports]==[256,512]
    deltas={}
    for split in ('training','validation'):
        deltas[split]={}
        for source,metrics in runs[0][split]['per_source'].items():
            high=runs[1][split]['per_source'][source]
            deltas[split][source]={k:high[k]-v for k,v in metrics.items() if v is not None and high[k] is not None}
    audit=json.loads((folders[1]/'resolution_audit.json').read_text())
    assert audit['passed'] and audit['compared_scenes']==96
    write_json(ROOT/'docs/experiments/pose_resolution_512_results.json',dict(
        execution_verified=True,equal_updates_not_equal_compute=True,
        identical_initialization_and_training_order=True,single_seed=True,
        low=reports[0],high=reports[1],checkpoint_verification=checks,
        per_source_high_minus_low=deltas,resolution_geometry_audit=audit,
        limitation='Shared synthetic assets and one seed; no real-animal or causal mechanism claim beyond this resolution intervention.'))
    print(json.dumps(dict(low=runs[0]['validation'],high=runs[1]['validation'],delta=deltas),indent=2))


if __name__=='__main__': main()
