"""Verify sampling isolation, coverage, labels, budgets and checkpoint provenance."""
from collections import Counter
import fcntl
import json
import math

from train_cylinder_combinations_v2 import ROOT, OUTPUT, PARENT, ARM, build_protocol
from run_depth_pose_ablation import torch, fingerprint, write_json, state_hash


def main():
    spec=json.loads((OUTPUT/'protocol.json').read_text())
    old=json.loads((PARENT/'protocol.json').read_text())
    result=json.loads((OUTPUT/'results.json').read_text())
    prior=json.loads((PARENT/'results.json').read_text())
    assert json.loads((OUTPUT/'status.json').read_text())['state']=='completed'
    with (OUTPUT/'running.lock').open() as file: fcntl.flock(file,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert build_protocol(spec['config'])==spec
    assert fingerprint(OUTPUT/'protocol.json')==result['protocol_sha256']
    assert fingerprint(PARENT/'results.json')==spec['parent_results_sha256']
    for key in ('evaluation','schedule','slots','checkpoint_sha256','encoder_sha256','manifest_sha256'):
        assert spec[key]==old[key],key
    history=json.loads((OUTPUT/ARM/'history.json').read_text())
    previous=json.loads((PARENT/'balanced_views/history.json').read_text())
    assert len(history)==len(previous)==800
    counts=Counter(); replay=0
    for i,(row,before) in enumerate(zip(history,previous)):
        a=spec['records'][spec['arms'][ARM][i][0]]; b=old['records'][old['arms']['balanced_views'][i][0]]
        for key in b:
            if key not in ('views','diagnostic_view_order','physical_roles'): assert a[key]==b[key],key
        if spec['slots'][i]['kind']!='cylinder': assert a==b; replay+=1
        for key in ('scene','image_hw','images_seen','patch_tokens_seen','slot_kind','step','learning_rate'):
            assert row[key]==before[key],(i,key)
        assert row['views']==a['views'] and len(row['views'])==len(before['views'])
        assert all(math.isfinite(row[k]) for k in ('loss','depth_loss','pose_loss','gradient_norm'))
        counts.update(a['physical_roles'])
    current=result['arms'][ARM]; reference=prior['arms']['balanced_views']
    for key in ('images_seen','patch_tokens_seen','initialization_hash'): assert current[key]==reference[key]
    assert current['encoder_gradients_absent'] and current['checkpoint_reload_verified']
    checkpoint=torch.load(OUTPUT/ARM/'final_heads.pt',weights_only=True,map_location='cpu')
    assert state_hash(checkpoint['model'])==checkpoint['trained_hash']!=checkpoint['initial_hash']
    assert checkpoint['protocol_sha256']==result['protocol_sha256']
    assert not any(k.startswith('aggregator.') for k in checkpoint['model'])
    assert len(result['baseline']['records'])==len(prior['baseline']['records'])==88
    errors=[]; projection=[]; cameras={}
    for idx,(a,b) in enumerate(zip(result['baseline']['records'],prior['baseline']['records'])):
        assert all(a[k]==b[k] for k in ('group','scene','views'))
        errors.extend(abs(a['metrics'][k]-v) for k,v in b['metrics'].items() if v is not None)
        item=torch.load(OUTPUT/'initial'/f'prediction_{idx:03d}.pt',weights_only=True,map_location='cpu')
        t=item['targets']; r=item['record']; xyz=t['keypoint_world_xyz'][0]
        homo=torch.cat((xyz,torch.ones_like(xyz[...,:1])),dim=-1)
        p=t['intrinsics'][0] @ t['extrinsics'][0,:,:3]
        uvw=torch.einsum('vij,mkj->vmki',p,homo)
        expected=t['keypoint_xy'][0]*torch.tensor(r['image_hw'][::-1])
        projection.extend((uvw[...,:2]/uvw[...,2:3]-expected).norm(dim=-1)[t['keypoint_valid'][0]].tolist())
        for v,cam in enumerate(r['views']):
            key=(r['scene'],cam); values=(t['intrinsics'][0,v],t['extrinsics'][0,v],xyz)
            if key in cameras:
                for x,y in zip(cameras[key],values): torch.testing.assert_close(x,y,atol=0,rtol=0)
            else: cameras[key]=values
    assert max(errors)<1e-6 and max(projection)<.01
    train={spec['records'][x[0]]['scene'] for x in spec['arms'][ARM]}
    assert not train & {r['scene'] for r in result['baseline']['records']}
    assert not set(spec['train_sources']) & set(spec['development_sources'])
    report=dict(passed=True,protocol_rebuilt_exactly=True,source_code_weights_verified=True,
        baseline_metric_max_difference=max(errors),gt_projection_max_px=max(projection),
        fixed_camera_and_world_scale=True,slot_by_slot_source_shape_budget_identical=True,
        unchanged_replay_slots=replay,images=current['images_seen'],patch_tokens=current['patch_tokens_seen'],
        all_training_physical_image_counts=dict(counts),coverage=spec['sampling_coverage']['gates'],
        final_checkpoint_sha256=fingerprint(OUTPUT/ARM/'final_heads.pt'),
        checkpoint_reload_verified=current['checkpoint_reload_verified'],
        first100_pose_loss=sum(r['pose_loss'] for r in history[:100])/100,
        last100_pose_loss=sum(r['pose_loss'] for r in history[-100:])/100)
    write_json(OUTPUT/'delivery_verification.json',report); print(json.dumps(report,indent=2))


if __name__=='__main__': main()
