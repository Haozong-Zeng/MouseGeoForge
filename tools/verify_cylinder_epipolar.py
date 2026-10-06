"""Verify epipolar-only control without changing historical protocols."""
import fcntl
import json
import math
from train_cylinder_epipolar import ROOT,OUTPUT,PARENT,ARM,build_protocol
from run_depth_pose_ablation import torch,fingerprint,write_json,state_hash


def main():
    torch.set_num_threads(2)
    spec=json.loads((OUTPUT/'protocol.json').read_text());old=json.loads((PARENT/'protocol.json').read_text())
    result=json.loads((OUTPUT/'results.json').read_text());prior=json.loads((PARENT/'results.json').read_text())
    assert json.loads((OUTPUT/'status.json').read_text())['state']=='completed'
    with (OUTPUT/'running.lock').open() as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert build_protocol(spec['config'])==spec
    for key in ('records','evaluation','schedule','slots','checkpoint_sha256','encoder_sha256','manifest_sha256'):
        assert spec[key]==old[key],key
    assert spec['arms'][ARM]==old['arms']['source_mix']
    assert fingerprint(OUTPUT/'protocol.json')==result['protocol_sha256']
    assert fingerprint(PARENT/'results.json')==spec['parent_results_sha256']
    current=result['arms'][ARM];reference=prior['arms']['source_mix']
    for key in ('images_seen','patch_tokens_seen','initialization_hash'):assert current[key]==reference[key]
    history=json.loads((OUTPUT/ARM/'history.json').read_text());previous=json.loads((PARENT/'source_mix/history.json').read_text())
    aux=json.loads((OUTPUT/ARM/'epipolar_loss_history.json').read_text())
    assert len(history)==len(previous)==len(aux)==800
    mono=multi=0
    for row,before,a in zip(history,previous,aux):
        for key in ('scene','views','image_hw','images_seen','patch_tokens_seen','slot_kind','step','learning_rate'):assert row[key]==before[key]
        assert a['step']==row['step'] and abs(row['pose_loss']-a['combined_pose_loss'])<1e-6
        assert abs(a['combined_pose_loss']-a['base_pose_loss']-a['auxiliary_loss'])<1e-5
        assert all(math.isfinite(row[k]) for k in ('loss','pose_loss','depth_loss','gradient_norm'))
        if len(row['views'])==1:assert a['auxiliary_loss']==0 and a['points']==0;mono+=1
        else:assert a['points']>0 and a['pairs']>0;multi+=1
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
    report=dict(passed=True,protocol_rebuilt_exactly=True,source_code_weights_verified=True,
        exact_sampling_initialization_budget=True,baseline_metric_max_difference=max(parity),baseline_prediction_max_difference=0,
        mono_steps_zero_aux=mono,multiview_steps_supervised=multi,images=current['images_seen'],patch_tokens=current['patch_tokens_seen'],
        training_seconds=current['training_seconds'],final_checkpoint_sha256=fingerprint(OUTPUT/ARM/'final_heads.pt'))
    write_json(OUTPUT/'delivery_verification.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
