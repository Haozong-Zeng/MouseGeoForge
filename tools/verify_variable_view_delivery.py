"""Read-only training/source checks; writes only a derived verification report."""
import hashlib
import json
import math
import fcntl

from train_variable_view_adaptation import ROOT, OUTPUT, torch, fingerprint, state_hash, write_json


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
    checks={}; initial_hashes=set()
    for arm in spec['config']['arms']:
        checkpoint=torch.load(OUTPUT/arm/'final_heads.pt',weights_only=True,map_location='cpu')
        assert state_hash(checkpoint['model'])==checkpoint['trained_hash']
        assert checkpoint['trained_hash']!=checkpoint['initial_hash']
        assert not any(k.startswith('aggregator.') for k in checkpoint['model'])
        initial_hashes.add(checkpoint['initial_hash'])
        history=json.loads((OUTPUT/arm/'history.json').read_text())
        assert len(history)==spec['config']['steps']
        expected_scenes=[spec['records'][spec['arms'][arm][i][0]]['scene'] for i in spec['schedule']]
        assert [r['scene'] for r in history]==expected_scenes
        assert sum(len(r['views']) for r in history)==result['arms'][arm]['images_seen']
        assert all(math.isfinite(r[k]) for r in history for k in ('loss','pose_loss','depth_loss','gradient_norm'))
        assert result['arms'][arm]['encoder_gradients_absent']
        checks[arm]=dict(saved_tensor_hash_verified=True,identical_pose_schedule_verified=True,
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
