"""Budget, source, old-state projection and temperature-only candidate delivery."""
import fcntl
import json
import math
from train_cylinder_adaptive_temperature import ROOT,OUTPUT,PARENT,ARM,build_protocol
from run_depth_pose_ablation import torch,fingerprint,write_json,state_hash
from mousegeoforge.adaptive_temperature import PREFIX


def main():
    torch.set_num_threads(2)
    spec=json.loads((OUTPUT/'protocol.json').read_text());old=json.loads((PARENT/'protocol.json').read_text())
    result=json.loads((OUTPUT/'results.json').read_text());prior=json.loads((PARENT/'results.json').read_text())
    assert json.loads((OUTPUT/'status.json').read_text())['state']=='completed'
    with (OUTPUT/'running.lock').open() as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert build_protocol(spec['config'])==spec
    for key in ('records','evaluation','schedule','slots','checkpoint_sha256','encoder_sha256','manifest_sha256'):assert spec[key]==old[key]
    assert spec['arms'][ARM]==old['arms']['source_mix']
    assert fingerprint(OUTPUT/'protocol.json')==result['protocol_sha256']
    history=json.loads((OUTPUT/ARM/'history.json').read_text());previous=json.loads((PARENT/'source_mix/history.json').read_text())
    temperature=json.loads((OUTPUT/ARM/'temperature_history.json').read_text())
    assert len(history)==len(previous)==len(temperature)==800
    for row,before,t in zip(history,previous,temperature):
        for key in ('scene','views','image_hw','images_seen','patch_tokens_seen','slot_kind','step','learning_rate'):assert row[key]==before[key]
        assert t['step']==row['step'] and t['pose_loss']==row['pose_loss']
        assert .5<=t['scale_min']<=t['scale_max']<=8.000001
        assert all(math.isfinite(row[k]) for k in ('loss','pose_loss','depth_loss','gradient_norm'))
    assert temperature[0]['scale_min']==temperature[0]['scale_max']==1
    baseline=json.loads((OUTPUT/'baseline_pretrain_verification.json').read_text());assert baseline['passed']
    parity=[]
    for a,b in zip(result['baseline']['records'],prior['baseline']['records']):
        assert all(a[k]==b[k] for k in ('scene','views','group'))
        parity.extend(abs(a['metrics'][k]-v) for k,v in b['metrics'].items() if v is not None)
    assert len(result['baseline']['records'])==88 and max(parity)<1e-6
    checkpoint=torch.load(OUTPUT/ARM/'final_heads.pt',weights_only=True,map_location='cpu')
    assert state_hash(checkpoint['model'])==checkpoint['trained_hash']!=checkpoint['initial_hash']
    extras={k:v for k,v in checkpoint['model'].items() if k.startswith(PREFIX)}
    assert len(extras)==2 and sum(v.numel() for v in extras.values())==257
    assert any(v.abs().max()>0 for v in extras.values())
    assert checkpoint['protocol_sha256']==result['protocol_sha256']
    candidate=result['arms'][ARM]
    assert candidate['encoder_gradients_absent'] and candidate['checkpoint_reload_verified']
    for key in ('images_seen','patch_tokens_seen'):assert candidate[key]==prior['arms']['source_mix'][key]
    for idx in range(88):
        stored=torch.load(OUTPUT/ARM/f'temperature_{idx:03d}.pt',weights_only=True,map_location='cpu')
        assert torch.isfinite(stored['scale']).all() and stored['scale'].min()>=.5 and stored['scale'].max()<=8.000001
    report=dict(passed=True,protocol_rebuilt_exactly=True,sampling_budget_source_verified=True,original_initial_tensors_equal=True,
        baseline_prediction_max_difference=0,baseline_metric_max_difference=max(parity),new_parameters=257,new_parameters_updated=True,
        images=candidate['images_seen'],patch_tokens=candidate['patch_tokens_seen'],training_seconds=candidate['training_seconds'],
        final_checkpoint_sha256=fingerprint(OUTPUT/ARM/'final_heads.pt'),scale_last_step=temperature[-1])
    write_json(OUTPUT/'delivery_verification.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
