"""Resolution-only 512/768 audit and comparison on the fixed diverse400 protocol."""
import argparse
import json
from run_depth_pose_ablation import ROOT, torch, build_manifest, load_batch, write_json, depth_alignment_audit

LOW='pose_diverse_core23_512_v1'
HIGH='pose_diverse_core23_768_v1'
PREFLIGHT=ROOT/'output/pose_diverse_core23_768_preflight_v1'


def assert_pair(low, high):
    ignored={'purpose','notes','long_side'}
    assert [low['config']['long_side'],high['config']['long_side']]==[512,768]
    assert {k:v for k,v in low['config'].items() if k not in ignored}=={
        k:v for k,v in high['config'].items() if k not in ignored}
    for key in ('source_plan_sha256','grouping_sha256','reference_manifest_sha256','experimental_source_groups'):
        assert low[key]==high[key],key
    assert len(low['records'])==len(high['records'])==432
    for a,b in zip(low['records'],high['records'],strict=True):
        assert {k:v for k,v in a.items() if k!='image_hw'}=={k:v for k,v in b.items() if k!='image_hw'}
        assert b['image_hw']==[int(x*1.5) for x in a['image_hw']]


def audit():
    torch.set_num_threads(2)
    low=json.loads((ROOT/'output'/LOW/'manifest.json').read_text())
    assert build_manifest(low['config'])==low, '512 manifest regression'
    high=build_manifest(json.loads((ROOT/'configs/experiments'/f'{HIGH}.json').read_text()))
    assert_pair(low,high)
    write_json(PREFLIGHT/'manifest.json',high)
    rows=[]
    for a,b in zip(low['records'],high['records'],strict=True):
        x,y=load_batch(a),load_batch(b)
        assert x['view_ids']==y['view_ids']
        assert y['images'].shape[-2:]==tuple(b['image_hw'])
        assert y['depths'].shape[-2:]==tuple(b['image_hw'])
        for key in ('extrinsics','scale','instance_keypoint_xy','instance_keypoint_world_xyz',
                    'instance_keypoint_valid','instance_ids'):
            torch.testing.assert_close(x[key],y[key],atol=1e-6,rtol=0)
        k=x['intrinsics'].clone(); k[...,:2,:]*=1.5
        torch.testing.assert_close(k,y['intrinsics'],atol=1e-4,rtol=0)
        h,w=b['image_hw']; errors=[]
        for v in range(2):
            world=y['instance_keypoint_world_xyz'][0]
            e=y['extrinsics'][0,v]; k=y['intrinsics'][0,v]
            p=(world@e[:3,:3].T+e[:3,3])@k.T
            uv=p[...,:2]/p[...,2:]
            gt=y['instance_keypoint_xy'][0,v]*torch.tensor([w,h])
            valid=y['instance_keypoint_valid'][0,v]
            errors.extend((uv-gt).norm(dim=-1)[valid].tolist())
        alignment=depth_alignment_audit(y)
        assert errors and max(errors)<.01
        assert alignment['surface_anchor_absrel_median']<.15
        rows.append(dict(scene=b['scene'],image_hw=b['image_hw'],actual_view_order=y['view_ids'],
                         max_projection_px=max(errors),**alignment))
        if len(rows)%32==0: print(f'Audited {len(rows)}/432 pairs',flush=True)
    write_json(PREFLIGHT/'geometry_audit.json',dict(passed=True,compared_scenes=len(rows),
        unchanged_world_scale=True,unchanged_normalized_uv=True,intrinsics_scale=1.5,records=rows))
    print('PAIRED 768 GEOMETRY PASS',flush=True)


def compare():
    folders=[ROOT/'output'/n for n in (LOW,HIGH)]
    manifests=[json.loads((p/'manifest.json').read_text()) for p in folders]
    assert_pair(*manifests)
    reports=[json.loads((p/'results.json').read_text()) for p in folders]
    checks=[json.loads((p/'checkpoint_verification.json').read_text()) for p in folders]
    assert all(c['passed'] and len(c['runs'])==1 and c['runs'][0]['passed']
               and c['runs'][0]['validated_scenes']==32 and c['runs'][0]['max_metric_difference']<1e-6
               for c in checks)
    assert all(len(r['runs'])==1 and r['passed'] for r in reports)
    runs=[r['runs'][0] for r in reports]
    assert all(r['seed']==17 and r['mode']=='off' and r['steps']==15000 for r in runs)
    for key in ('initial_state_sha256','lr_schedule_sha256','train_order_sha256','train_order_first5000_sha256'):
        assert runs[0][key]==runs[1][key],key
    assert reports[0]['provenance']['checkpoint_sha256']==reports[1]['provenance']['checkpoint_sha256']
    # Initial predictions may differ because inputs/tokens differ; do NOT require initial metric equality.
    delta={}
    for split in ('training','reference_training','validation'):
        assert runs[0][split]['scenes']==runs[1][split]['scenes']
        delta[split]={source:{key:metrics[key]-runs[0][split]['per_source'][source][key]
            for key in metrics if metrics[key] is not None and runs[0][split]['per_source'][source][key] is not None}
            for source,metrics in runs[1][split]['per_source'].items()}
    geometry=json.loads((PREFLIGHT/'geometry_audit.json').read_text())
    assert geometry['passed'] and geometry['compared_scenes']==432
    write_json(ROOT/'docs/experiments/pose_diverse_resolution_768_results.json',dict(
        execution_verified=True,identical_initialization_lr_and_training_order=True,
        same_scenes_and_views=True,equal_updates_not_equal_compute=True,
        low=reports[0],high=reports[1],checkpoint_verification=checks,
        geometry_audit=geometry,per_source_high_minus_low=delta,
        training_curves=[json.loads((p/'seed17_off/training_curve.json').read_text()) for p in folders],
        limitations=['one seed','synthetic only','one dual-mouse training source',
                     'initial predictions need not agree across resolutions','fixed final step; no validation selection']))
    print(json.dumps(delta,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['audit','compare'])
    args=p.parse_args()
    audit() if args.action=='audit' else compare()
