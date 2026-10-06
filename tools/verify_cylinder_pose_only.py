"""Frozen-state, source, exact schedule and depth-invariance delivery checks."""
import fcntl
import json
import math

from train_cylinder_pose_only import ROOT,OUTPUT,PARENT,build_protocol,torch
from run_depth_pose_ablation import fingerprint,write_json,state_hash


def main():
    spec=json.loads((OUTPUT/'protocol.json').read_text())
    result=json.loads((OUTPUT/'results.json').read_text())
    parent=json.loads((PARENT/'results.json').read_text())
    oldspec=json.loads((PARENT/'protocol.json').read_text())
    assert json.loads((OUTPUT/'status.json').read_text())['state']=='completed'
    with (OUTPUT/'running.lock').open() as file:fcntl.flock(file,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert build_protocol(spec['config'])==spec
    for key in ('records','evaluation','schedule','slots','checkpoint_sha256','encoder_sha256','manifest_sha256'):
        assert spec[key]==oldspec[key],key
    assert spec['arms']['pose_head_only']==oldspec['arms']['balanced_views']
    assert fingerprint(OUTPUT/'protocol.json')==result['protocol_sha256']
    current=result['arms']['pose_head_only'];previous=parent['arms']['balanced_views']
    assert current['initialization_hash']==previous['initialization_hash']
    for key in ('images_seen','patch_tokens_seen'):assert current[key]==previous[key]
    history=json.loads((OUTPUT/'pose_head_only/history.json').read_text())
    oldhistory=json.loads((PARENT/'balanced_views/history.json').read_text())
    assert len(history)==len(oldhistory)==800
    for i,(a,b) in enumerate(zip(history,oldhistory)):
        for key in ('scene','views','image_hw','images_seen','patch_tokens_seen','slot_kind','learning_rate','step'):
            assert a[key]==b[key],(i,key)
        assert all(math.isfinite(a[k]) for k in ('loss','pose_loss','depth_loss','gradient_norm'))
        assert abs(a['loss']-a['pose_loss']-a['depth_loss'])<1e-5
    original=torch.load(ROOT/spec['config']['checkpoint'],weights_only=True,map_location='cpu')['model']
    checkpoint=torch.load(OUTPUT/'pose_head_only/final_heads.pt',weights_only=True,map_location='cpu')
    assert state_hash(checkpoint['model'])==checkpoint['trained_hash']
    assert checkpoint['protocol_sha256']==result['protocol_sha256']
    changed=[];frozen=[]
    for name,value in checkpoint['model'].items():
        equal=torch.equal(value,original[name])
        if name.startswith('instance_pose_head.'):
            if not equal:changed.append(name)
        else:
            assert equal,name
            frozen.append(name)
    assert changed and frozen
    parity=[];depth_difference=[]
    assert len(result['baseline']['records'])==len(parent['baseline']['records'])==len(current['records'])==len(spec['evaluation'])==88
    for initial,old,final in zip(result['baseline']['records'],parent['baseline']['records'],current['records']):
        assert initial['scene']==old['scene']==final['scene']
        assert initial['views']==old['views']==final['views']
        for key,value in old['metrics'].items():
            if value is not None:parity.append(abs(initial['metrics'][key]-value))
        for key,value in initial['metrics'].items():
            if 'depth' in key and value is not None:
                depth_difference.append(abs(final['metrics'][key]-value))
    assert max(parity)<1e-6
    assert max(depth_difference)<spec['config']['isolation_questions']['frozen_depth_max_difference']
    report=dict(passed=True,protocol_rebuilt_exactly=True,original_data_code_and_weights_unchanged=True,
        parent_initial_metric_max_difference=max(parity),frozen_depth_metric_max_difference=max(depth_difference),
        unchanged_non_pose_tensor_count=len(frozen),changed_pose_tensor_count=len(changed),
        images=current['images_seen'],patch_tokens=current['patch_tokens_seen'],
        identical_slot_by_slot_budget=True,checkpoint_reload_verified=current['checkpoint_reload_verified'],
        final_checkpoint_sha256=fingerprint(OUTPUT/'pose_head_only/final_heads.pt'),
        first100_pose_loss=sum(r['pose_loss'] for r in history[:100])/100,
        last100_pose_loss=sum(r['pose_loss'] for r in history[-100:])/100)
    write_json(OUTPUT/'delivery_verification.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
