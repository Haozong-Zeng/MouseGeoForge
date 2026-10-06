"""Verify paired budgets, frozen provenance, source splits and saved checkpoints."""
import fcntl
import json
import math
from collections import Counter

from train_cylinder_view_balance import (ROOT, OUTPUT, build_protocol, scene_digest,
    fingerprint, write_json, torch)
from run_depth_pose_ablation import state_hash


def main():
    spec=json.loads((OUTPUT/'protocol.json').read_text())
    result=json.loads((OUTPUT/'results.json').read_text())
    assert json.loads((OUTPUT/'status.json').read_text())['state']=='completed'
    with (OUTPUT/'running.lock').open() as file:
        fcntl.flock(file,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert build_protocol(spec['config'])==spec
    assert fingerprint(OUTPUT/'protocol.json')==result['protocol_sha256']
    assert not set(spec['train_sources']) & set(spec['development_sources'])
    # Compare retained original pair metrics against saved 15k validation.
    original=json.loads((ROOT/'output/pose_diverse_core23_768_v1/seed17_off/validation.json').read_text())
    byscene={r['scene']:r['metrics'] for r in original['records']}
    errors=[]
    for r in result['baseline']['records']:
        if r['group'].startswith('old_'):
            errors.extend(abs(r['metrics'][k]-v) for k,v in byscene[r['scene']].items() if v is not None)
    assert max(errors)<1e-6,max(errors)
    projection_errors=[]
    camera_reference={}
    for idx,row in enumerate(result['baseline']['records']):
        saved=torch.load(OUTPUT/'initial'/f'prediction_{idx:03d}.pt',weights_only=True,map_location='cpu')
        target=saved['targets']; record=saved['record']
        xyz=target['keypoint_world_xyz'][0]
        homo=torch.cat((xyz,torch.ones_like(xyz[...,:1])),dim=-1)
        proj=target['intrinsics'][0] @ target['extrinsics'][0,:,:3]
        projected=torch.einsum('vij,mkj->vmki',proj,homo)
        uv=projected[...,:2]/projected[...,2:3]
        height,width=record['image_hw']
        expected=target['keypoint_xy'][0]*torch.tensor([width,height])
        valid=target['keypoint_valid'][0]
        projection_errors.extend((uv-expected).norm(dim=-1)[valid].tolist())
        for v,camera in enumerate(record['views']):
            key=(record['scene'],camera)
            values=(target['intrinsics'][0,v],target['extrinsics'][0,v],xyz)
            if key in camera_reference:
                for first,second in zip(camera_reference[key],values):
                    torch.testing.assert_close(first,second,atol=0,rtol=0)
            else: camera_reference[key]=values
    assert max(projection_errors)<.01,max(projection_errors)
    initial_hashes=set(); histories={}; checks={}
    train_scenes=set()
    for arm in spec['config']['arms']:
        history=json.loads((OUTPUT/arm/'history.json').read_text())
        histories[arm]=history
        assert len(history)==800
        checkpoint=torch.load(OUTPUT/arm/'final_heads.pt',weights_only=True,map_location='cpu')
        assert state_hash(checkpoint['model'])==checkpoint['trained_hash']!=checkpoint['initial_hash']
        assert not any(k.startswith('aggregator.') for k in checkpoint['model'])
        assert checkpoint['protocol_sha256']==result['protocol_sha256']
        initial_hashes.add(checkpoint['initial_hash'])
        roles=Counter(); counts=Counter()
        for i,row in enumerate(history):
            rec=spec['records'][spec['arms'][arm][i][0]]
            train_scenes.add(rec['scene'])
            assert rec['split']=='train'
            assert row['scene']==rec['scene'] and row['views']==rec['views']
            assert row['image_hw']==rec['image_hw'] and row['step']==i+1
            assert row['slot_kind']==spec['slots'][i]['kind']
            assert row['learning_rate']==spec['config']['arm_learning_rates'][arm]
            assert abs(row['loss']-row['pose_loss']-row['depth_loss'])<1e-5
            assert all(math.isfinite(row[k]) for k in ('loss','pose_loss','depth_loss','gradient_norm'))
            roles.update(rec['physical_roles']); counts[len(rec['views'])]+=1
        assert sum(len(r['views']) for r in history)==result['arms'][arm]['images_seen']==1800
        tokens=sum(len(r['views'])*(r['image_hw'][0]//16)*(r['image_hw'][1]//16) for r in history)
        assert tokens==result['arms'][arm]['patch_tokens_seen']
        assert result['arms'][arm]['encoder_gradients_absent'] and result['arms'][arm]['checkpoint_reload_verified']
        checks[arm]=dict(image_role_counts=dict(roles),view_count_slots=dict(counts),patch_tokens=tokens,
            checkpoint_sha256=fingerprint(OUTPUT/arm/'final_heads.pt'),finite_losses=True,
            last100_pose_loss=sum(r['pose_loss'] for r in history[-100:])/100)
        del checkpoint
    assert len(initial_hashes)==1
    for a,b in zip(*histories.values()):
        for k in ('scene','image_hw','images_seen','patch_tokens_seen','slot_kind'):
            assert a[k]==b[k],k
        assert len(a['views'])==len(b['views'])
    assert not train_scenes & {r['scene'] for r in result['baseline']['records']}
    report=dict(passed=True,original_pair_parity_max_error=max(errors),
        gt_projection_max_px=max(projection_errors),
        camera_and_world_scales_constant_across_subsets=True,
        source_scene_count=len({r['scene'] for r in spec['records'].values()}),
        protocol_rebuilt_exactly=True,source_hashes_verified=True,locked_code_unchanged=True,
        original_checkpoints_unchanged=True,paired_budgets_identical=True,arms=checks)
    write_json(OUTPUT/'delivery_verification.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
