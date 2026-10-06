"""Verify source mixture execution, exact baseline, budgets and immutable inputs."""
import fcntl
import json
import math
from train_cylinder_source_mix import ROOT,OUTPUT,PARENT,PREFLIGHT,ARM,build_protocol,validate_sequence
from run_depth_pose_ablation import torch,fingerprint,write_json,state_hash


def main():
    torch.set_num_threads(2)
    spec=json.loads((OUTPUT/'protocol.json').read_text());old=json.loads((PARENT/'protocol.json').read_text())
    result=json.loads((OUTPUT/'results.json').read_text());prior=json.loads((PARENT/'results.json').read_text())
    assert json.loads((OUTPUT/'status.json').read_text())['state']=='completed'
    with (OUTPUT/'running.lock').open() as file:fcntl.flock(file,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert build_protocol(spec['config'])==spec
    assert spec==json.loads(json.dumps(spec))
    assert fingerprint(OUTPUT/'protocol.json')==result['protocol_sha256']
    assert fingerprint(PARENT/'results.json')==spec['parent_results_sha256']
    for key in ('evaluation','schedule','slots','checkpoint_sha256','encoder_sha256','manifest_sha256'):
        assert spec[key]==old[key],key
    assert all(spec['records'][e['key']]==old['records'][e['key']] for e in old['evaluation'])
    seq=[spec['records'][x[0]] for x in spec['arms'][ARM]]
    assert seq==json.loads((PREFLIGHT/'protocol.json').read_text())['training_sequence']
    budget=validate_sequence(seq,old)
    current=result['arms'][ARM];reference=prior['arms']['combination_coverage']
    for key in ('images_seen','patch_tokens_seen','initialization_hash'):assert current[key]==reference[key]
    history=json.loads((OUTPUT/ARM/'history.json').read_text())
    previous=json.loads((PARENT/'combination_coverage/history.json').read_text())
    assert len(history)==len(previous)==800
    for row,before,record in zip(history,previous,seq):
        for key in ('image_hw','images_seen','patch_tokens_seen','slot_kind','step','learning_rate'):
            assert row[key]==before[key],key
        for key in ('scene','views','image_hw'):assert row[key]==record[key]
        if row['slot_kind']!='cylinder':
            assert row['scene']==before['scene'] and row['views']==before['views']
        assert all(math.isfinite(row[k]) for k in ('loss','pose_loss','depth_loss','gradient_norm'))
    parity=[]
    for a,b in zip(result['baseline']['records'],prior['baseline']['records']):
        assert all(a[k]==b[k] for k in ('group','scene','views'))
        parity.extend(abs(a['metrics'][k]-v) for k,v in b['metrics'].items() if v is not None)
    assert len(result['baseline']['records'])==88 and max(parity)<1e-6
    assert json.loads((OUTPUT/'baseline_pretrain_verification.json').read_text())['passed']
    checkpoint=torch.load(OUTPUT/ARM/'final_heads.pt',weights_only=True,map_location='cpu')
    assert state_hash(checkpoint['model'])==checkpoint['trained_hash']!=checkpoint['initial_hash']
    assert checkpoint['protocol_sha256']==result['protocol_sha256']
    assert current['checkpoint_reload_verified'] and current['encoder_gradients_absent']
    report=dict(passed=True,protocol_rebuilt_exactly=True,all_source_code_weights_verified=True,
        **budget,baseline_metric_max_difference=max(parity),baseline_prediction_max_difference=0,
        actual_history_verified=True,training_seconds=current['training_seconds'],
        final_checkpoint_sha256=fingerprint(OUTPUT/ARM/'final_heads.pt'))
    write_json(OUTPUT/'delivery_verification.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
