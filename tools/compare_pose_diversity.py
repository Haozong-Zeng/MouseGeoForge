"""Compare the diverse 400-scene run against the fixed 64-scene 512 control."""
import json
from run_depth_pose_ablation import ROOT,write_json


def main():
    names=['pose_convergence_512_15000_v1','pose_diverse_core23_512_v1']
    folders=[ROOT/'output'/n for n in names]
    reports=[json.loads((f/'results.json').read_text()) for f in folders]
    checks=[json.loads((f/'checkpoint_verification.json').read_text()) for f in folders]
    manifests=[json.loads((f/'manifest.json').read_text()) for f in folders]
    assert all(c['passed'] and len(c['runs'])==1 and c['runs'][0]['validated_scenes']==32
               and c['runs'][0]['max_metric_difference']<1e-6 for c in checks)
    assert all(len(r['runs'])==1 for r in reports)
    runs=[r['runs'][0] for r in reports]
    for key in ('initial_state_sha256','lr_schedule_sha256'):
        assert runs[0][key]==runs[1][key],key
    assert all(r['seed']==17 and r['mode']=='off' and r['steps']==15000 for r in runs)
    excluded={'purpose','notes','dataset_policy','reference_manifest'}
    assert {k:v for k,v in reports[0]['config'].items() if k not in excluded}=={
        k:v for k,v in reports[1]['config'].items() if k not in excluded}
    assert runs[0]['initial_validation']==runs[1]['initial_validation']
    assert reports[0]['provenance']['checkpoint_sha256']==reports[1]['provenance']['checkpoint_sha256']
    assert [r for r in manifests[0]['records'] if r['split']=='val']==[r for r in manifests[1]['records'] if r['split']=='val']
    assert runs[1]['reference_training']['scenes']==64
    delta={}
    for split,old,new in [('validation',runs[0]['validation'],runs[1]['validation']),
                         ('original64_training',runs[0]['training'],runs[1]['reference_training'])]:
        delta[split]={source:{key:metrics[key]-old['per_source'][source][key]
            for key in metrics if metrics[key] is not None and old['per_source'][source][key] is not None}
            for source,metrics in new['per_source'].items()}
    audit=json.loads((ROOT/'output/pose_diverse_core23_preflight_v1/geometry_audit.json').read_text())
    assert audit['passed'] and audit['scenes']==432
    write_json(ROOT/'docs/experiments/pose_diverse_core23_results.json',dict(
        execution_verified=True,baseline=reports[0],diverse=reports[1],checks=checks,
        identical_validation=True,identical_initialization_and_lr=True,
        per_source_diverse_minus_baseline=delta,geometry_audit=audit,
        training_curves=[json.loads((f/'seed17_off/training_curve.json').read_text()) for f in folders],
        limitations=['one seed','more data and coverage jointly changed','same updates, fewer epochs',
                     'dual source-session count not increased','synthetic only; no real-animal claim']))
    print(json.dumps(delta,indent=2))


if __name__=='__main__': main()
