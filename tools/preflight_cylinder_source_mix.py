"""Read-only train-source mixture preflight; no optimizer or split mutation."""
from collections import Counter
import copy
import json
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

from run_depth_pose_ablation import ROOT, torch, load_scene, load_batch, fingerprint, write_json, instance_targets_from_batch
from train_cylinder_view_balance import cameras, spaced, scene_digest, median_camera_baseline

OUT = ROOT/'output/cylinder_source_mix_preflight_v1'
PARENT = ROOT/'output/cylinder_combinations_v2'
NAME = 'PyMouseStudio-BrownFur-KeypointMoseq-Core23-Sample10-First200'


def role_mapping(source, destination):
    """Bijective physical-role mapping minimizing world-forward angular distance."""
    mapping, angles = {}, {}
    for role in ('side', 'top', 'bottom'):
        a = sorted((r for r in source if r['orientation'] == role), key=lambda r:r['view_id'])
        b = sorted((r for r in destination if r['orientation'] == role), key=lambda r:r['view_id'])
        if len(a) != len(b) or not a:
            raise ValueError('Incompatible physical camera roles')
        va, vb = np.array([r['forward'] for r in a]), np.array([r['forward'] for r in b])
        cost = np.arccos(np.clip(va @ vb.T, -1, 1)) * 180/np.pi
        ii, jj = linear_sum_assignment(cost)
        for i,j in zip(ii,jj):
            mapping[a[i]['view_id']] = b[j]['view_id']
            angles[a[i]['view_id']] = float(cost[i,j])
    if len(mapping) != 8 or len(set(mapping.values())) != 8:
        raise ValueError('Expected exactly eight bijectively mapped cameras')
    return mapping, angles


def main():
    if OUT.exists():
        raise RuntimeError('Preflight output already exists; do not overwrite')
    torch.set_num_threads(2)
    parent = json.loads((PARENT/'protocol.json').read_text())
    assert json.loads((PARENT/'delivery_verification.json').read_text())['passed']
    assert all(fingerprint(ROOT/p)==h for p,h in parent['code_sha256'].items())
    manifest_path = ROOT/parent['config']['manifest']
    assert fingerprint(manifest_path)==parent['manifest_sha256']
    records = json.loads(manifest_path.read_text())['records']
    seq = [parent['records'][x[0]] for x in parent['arms']['combination_coverage']]
    used = {r['scene'] for r in seq}
    heldout = {r['scene'] for r in records if r['split'] != 'train'}
    selected = spaced([r for r in records if r['dataset']==NAME and r['split']=='train' and r['scene'] not in used],8)
    pool = {r['scene']:r for r,s in zip(seq,parent['slots']) if s['kind']=='cylinder'}
    replaced = sorted(pool.values(),key=lambda r:r['frame'])[1::2]
    assert len(replaced)==len(selected)==8
    assert all(r['scene'] not in heldout for r in selected)
    revisions = {r['scene']:r['source_revision'] for r in parent['records'].values()}
    assert all(scene_digest(ROOT/p)==h for p,h in revisions.items())
    replacements, diagnostics = {}, []
    for old, new in zip(replaced,selected):
        scene = ROOT/new['scene']; loaded=load_scene(scene)
        roles = cameras(scene)
        mapping, angles = role_mapping(cameras(ROOT/old['scene']),roles)
        role_by_id = {r['view_id']:r['orientation'] for r in roles}
        scale = float(median_camera_baseline(np.stack([loaded.frames[v].extrinsic for v in new['views']])))
        revision = scene_digest(scene); revisions[new['scene']]=revision
        enriched = dict(new, normalization_scale=scale,source_revision=revision)
        replacements[old['scene']] = dict(record=enriched, mapping=mapping)
        # All eight cameras are inspected once, split into actual V=1..4 orders.
        old_orders = sorted({tuple(r['views']) for r in seq if r['scene']==old['scene']},key=lambda x:(len(x),x))
        orders = [[mapping[v] for v in next(o for o in old_orders if len(o)==n)] for n in (1,2,3,4)]
        checked = set(v for order in orders for v in order)
        orders += [[v] for v in sorted(set(mapping.values())-checked)]
        checks=[]
        for order in orders:
            rec=dict(enriched,views=order,diagnostic_view_order=order,physical_roles=[role_by_id[v] for v in order])
            batch=load_batch(rec,num_views=len(order)); target=instance_targets_from_batch(batch)
            assert batch['images'].shape == (1,len(order),3,768,768)
            assert np.isclose(float(batch['scale'][0]),scale)
            xyz=target['keypoint_world_xyz'][0]
            homo=torch.cat((xyz,torch.ones_like(xyz[...,:1])),dim=-1)
            uvw=torch.einsum('vij,mkj->vmki',target['intrinsics'][0]@target['extrinsics'][0,:,:3],homo)
            uv=target['keypoint_xy'][0]*torch.tensor([768,768])
            projection=(uvw[...,:2]/uvw[...,2:3]-uv).norm(dim=-1)[target['keypoint_valid'][0]]
            assert projection.numel() and float(projection.max())<.01
            depths=batch['depths'][batch['depth_masks']]
            assert torch.isfinite(depths).all() and (depths>0).all()
            # Surface anchors are used only for metric-unit/depth alignment QA.
            # Interior skeletal joints must not be equated to rendered surface depth.
            world=batch['surface_world_xyz_instances'][0]
            valid=batch['instance_keypoint_valid'][0] & batch['instance_keypoint_visible'][0]
            residuals=[]
            for i,v in enumerate(order):
                e=batch['extrinsics'][0,i]; z=(world @ e[:3,:3].T+e[:3,3])[...,2]
                xy=(batch['instance_keypoint_xy'][0,i]*768).round().long().clamp(0,767)
                d=batch['depths'][0,i,xy[...,1],xy[...,0]]
                ok=valid[i] & torch.isfinite(z) & (z>0) & torch.isfinite(d) & (d>0)
                residuals.extend(((d-z).abs()/z.clamp_min(1e-6))[ok].tolist())
            med=float(np.median(residuals)) if residuals else None
            assert med is not None and med<.05, ('Surface/depth unit mismatch',new['frame'],order,med)
            checks.append(dict(views=order,roles=rec['physical_roles'],gt_projection_max_px=float(projection.max()),
                normalized_depth_median=float(depths.median()),surface_depth_absrel_median=med,surface_anchor_count=len(residuals)))
        diagnostics.append(dict(old_frame=old['frame'],new_frame=new['frame'],scene=new['scene'],
            native_image_wh=list(Image.open(scene/'blended_images/00000000.jpg').size),
            depth_units=new['depth_units'],normalization_scale=scale,camera_mapping=mapping,
            forward_angle_difference_deg=angles,checks=checks))
        print('Checked source frame',new['frame'],flush=True)
    candidate=[]
    for old,slot in zip(seq,parent['slots']):
        if slot['kind']=='cylinder' and old['scene'] in replacements:
            entry=replacements[old['scene']]; order=[entry['mapping'][v] for v in old['views']]
            new=dict(entry['record'],views=order,diagnostic_view_order=order,physical_roles=old['physical_roles'])
        else:new=copy.deepcopy(old)
        assert new['split']=='train' and new['image_hw']==old['image_hw'] and len(new['views'])==len(old['views'])
        assert new['physical_roles']==old['physical_roles']
        if slot['kind']!='cylinder':assert new==old
        candidate.append(new)
    assert len({r['scene'] for r in candidate})==40
    assert len(candidate)==800 and sum(len(r['views']) for r in candidate)==1800
    assert not {r['scene'] for r in candidate}&heldout
    assert all(scene_digest(ROOT/p)==h for p,h in revisions.items())
    protocol=dict(parent_protocol_sha256=fingerprint(PARENT/'protocol.json'),manifest_sha256=fingerprint(manifest_path),
        code_sha256={str(p.relative_to(ROOT)):fingerprint(p) for p in [ROOT/'tools/preflight_cylinder_source_mix.py']},
        source_sha256=revisions,training_sequence=candidate,evaluation=parent['evaluation'],
        selection='Eight evenly spaced locked-train Sample10 poses excluding replay; replace every second frame-sorted ClimbWall pose. No metric-based selection.',
        scope='Training-source mixture, not isolated action/pose diversity: rig, asset and native resolution also differ. Frozen original initialization; no auxiliary loss. Synthetic reused development, not real evaluation.')
    write_json(OUT/'protocol.json',protocol)
    write_json(OUT/'report.json',dict(passed=True,training_started=False,weights_updated=False,datasets_modified=False,
        training_steps=len(candidate),unique_train_poses=40,images=1800,evaluation_records=len(parent['evaluation']),
        replay_unchanged=True,slot_hw_views_roles_unchanged=True,selected_frames=[r['frame'] for r in selected],
        cylinder_slot_dataset_counts=dict(Counter(r['dataset'] for r,s in zip(candidate,parent['slots']) if s['kind']=='cylinder')),
        diagnostics=diagnostics,excluded_test_source='mouse_walk_around_91; never added to training'))
    print('Preflight passed; no training launched',flush=True)


if __name__=='__main__':main()
