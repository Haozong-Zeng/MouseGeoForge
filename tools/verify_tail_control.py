"""Read-only training/source checks; writes only a derived verification report."""
import hashlib
import json
import math
import fcntl

from train_tail_control import ROOT, OUTPUT, torch, fingerprint, state_hash, write_json


def main():
    spec=json.loads((OUTPUT/'protocol.json').read_text())
    result=json.loads((OUTPUT/'results.json').read_text())
    assert json.loads((OUTPUT/'status.json').read_text())['state']=='completed'
    with (OUTPUT/'running.lock').open() as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert fingerprint(ROOT/spec['config']['checkpoint'])==spec['checkpoint_sha256']
    assert fingerprint(ROOT/'vggt-omega/vggt_omega_1b_512.pt')==spec['encoder_sha256']
    for source,digest in spec['code_sha256'].items(): assert fingerprint(ROOT/source)==digest,source
    sources={r['scene']:r['source_revision'] for r in spec['records'].values()}
    for scene,expected in sources.items():
        digest=hashlib.sha256(); root=ROOT/scene
        for path in sorted(root.rglob('*')):
            if path.is_file():
                digest.update(str(path.relative_to(root)).encode()); digest.update(fingerprint(path).encode())
        assert digest.hexdigest()==expected,scene
    assert fingerprint(ROOT/spec['config']['manifest'])==spec['manifest_sha256']
    assert fingerprint(OUTPUT/'protocol.json')==result['protocol_sha256']
    assert not set(spec['train_sources']) & set(spec['development_sources'])
    old=json.loads((ROOT/'output/variable_view_adaptation_v1_retry1/initial/evaluation.json').read_text())
    for group,metrics in old['summary'].items():
        for name in ('box_matched_nme','detection_aware_pck_0.1','body_depth_absrel'):
            assert abs(metrics[name]-result['baseline']['summary'][group][name])<1e-6,(group,name)
    parent=ROOT/spec['config']['parent_run']
    assert fingerprint(parent/'protocol.json')==spec['parent_protocol_sha256']
    assert fingerprint(parent/'results.json')==spec['parent_results_sha256']
    parent_spec=json.loads((parent/'protocol.json').read_text())
    from train_tail_control import build_protocol
    assert build_protocol(spec['config'])==spec,'Tail protocol not reproducible'
    parent_result=json.loads((parent/'results.json').read_text())
    for key in ('evaluation','schedule','checkpoint_sha256','encoder_sha256','manifest_sha256'):
        assert spec[key]==parent_spec[key],key
    for group,metrics in parent_result['baseline']['summary'].items():
        for name in ('box_matched_nme','detection_aware_pck_0.1','body_depth_absrel'):
            assert abs(metrics[name]-result['baseline']['summary'][group][name])<1e-6
    checks={}; initial_hashes=set()
    for arm in spec['config']['arms']:
        checkpoint=torch.load(OUTPUT/arm/'final_heads.pt',weights_only=True,map_location='cpu')
        assert state_hash(checkpoint['model'])==checkpoint['trained_hash']
        assert checkpoint['trained_hash']!=checkpoint['initial_hash']
        assert not any(k.startswith('aggregator.') for k in checkpoint['model'])
        initial_hashes.add(checkpoint['initial_hash'])
        history=json.loads((OUTPUT/arm/'history.json').read_text())
        assert len(history)==spec['config']['steps']
        assert checkpoint['steps']==len(history)
        assert spec['arms'][arm]==parent_spec['arms']['replay_seed29']
        assert result['arms'][arm]['patch_tokens_seen']==2633184
        assert all(math.isfinite(r['tail_aux_loss']) and r['tail_aux_loss']>=0 for r in history)
        assert all(abs(r['loss']-r['pose_loss']-r['depth_loss']-
                       spec['config']['tail_aux_weight']*r['tail_aux_loss'])<1e-5 for r in history)
        assert all(r['learning_rate']==spec['config']['learning_rate'] for r in history)
        assert result['arms'][arm]['images_seen']==1800
        assert all(r['training_seed']==spec['config']['arm_seeds'][arm] for r in history)
        assert all(r['slot_kind']==spec['arm_slots'][arm][i]['kind'] for i,r in enumerate(history))
        assert checkpoint['initial_hash']==parent_result['arms']['replay_seed29']['initialization_hash']
        assert checkpoint['protocol_sha256']==result['protocol_sha256']
        tokens=sum(len(r['views'])*(r['image_hw'][0]//16)*(r['image_hw'][1]//16) for r in history)
        assert tokens==result['arms'][arm]['patch_tokens_seen']
        for step,row in enumerate(history):
            expected=spec['records'][spec['arms'][arm][step][0]]
            assert row['views']==expected['views'] and row['image_hw']==expected['image_hw']
            assert expected['split']=='train' and expected['source_group'] in spec['train_sources']
        expected_scenes=[spec['records'][spec['arms'][arm][i][0]]['scene'] for i in spec['schedule']]
        assert [r['scene'] for r in history]==expected_scenes
        assert sum(len(r['views']) for r in history)==result['arms'][arm]['images_seen']
        assert all(math.isfinite(r[k]) for r in history for k in ('loss','pose_loss','depth_loss','gradient_norm'))
        assert result['arms'][arm]['encoder_gradients_absent']
        checks[arm]=dict(saved_tensor_hash_verified=True,locked_arm_specific_schedule_verified=True,
            all_training_losses_finite=True,encoder_gradients_absent_runtime_assertion=True,
            final_checkpoint_sha256=fingerprint(OUTPUT/arm/'final_heads.pt'),
            first100_pose_loss=sum(r['pose_loss'] for r in history[:100])/100,
            last100_pose_loss=sum(r['pose_loss'] for r in history[-100:])/100)
        del checkpoint
    assert len(initial_hashes)==1
    report=dict(passed=True,verified_source_scenes=len(sources),original_checkpoint_unchanged=True,
        encoder_checkpoint_unchanged=True,locked_training_code_unchanged=True,run_lock_released=True,
        arms=checks,limitation='Loss windows are descriptive; differing view composition does not establish convergence.')
    write_json(OUTPUT/'delivery_verification.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()






