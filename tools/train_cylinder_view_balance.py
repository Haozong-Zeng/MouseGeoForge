"""Physical-view sampling control using the unchanged bounded replay engine.

Only this experiment installs wrappers in the imported engine module. Production
loader/model and earlier locked trainers remain unchanged. K/E are used for
diagnostic triangulation, not injected into VGGT attention or predicted by it.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys

from scipy.stats import spearmanr
import train_variable_view_lr_control as engine
from run_depth_pose_ablation import (ROOT, torch, np, load_batch as base_load_batch,
    fingerprint, write_json, load_scene)
from training.finetune_blendedmvs import median_camera_baseline
from mousegeoforge.pilot_metrics import evaluate_scene as base_metrics
from vggt_omega.utils.robust_pose_geometry import triangulate_robust
from audit_bottom_view_coverage import cameras

CONFIG = ROOT / 'configs/experiments/cylinder_view_balance_v1.json'
OUTPUT = ROOT / 'output/cylinder_view_balance_v1'


def spaced(rows, n):
    rows = sorted(rows, key=lambda r: r['frame'])
    if n < 2 or len(rows) < n:
        raise ValueError('Insufficient independent source poses')
    return [rows[round(i * (len(rows)-1)/(n-1))] for i in range(n)]


def cylinder_order(role_map, n, variant, balanced):
    """Paired V=1..4 schedules; IDs derived from physical roles, never assumed."""
    side = sorted(k for k,v in role_map.items() if v == 'side')
    top = [k for k,v in role_map.items() if v == 'top']
    bottom = [k for k,v in role_map.items() if v == 'bottom']
    if len(side) < 4 or len(top) != 1 or len(bottom) != 1 or n not in (1,2,3,4):
        raise ValueError('Need a verified Cylinder rig with >=4 side, one top/bottom')
    shifted = side[variant % len(side):] + side[:variant % len(side)]
    if not balanced:
        return shifted[:n]
    if n == 1:
        return [bottom[0] if variant % 2 == 0 else top[0]]
    if n == 2:
        return [shifted[0], top[0] if variant % 2 == 0 else bottom[0]]
    order = [shifted[0], bottom[0], top[0]]
    if n == 4:
        order.insert(1, shifted[1])
    # Vary the reference camera deterministically across source poses.
    shift = variant % n
    return order[shift:] + order[:shift]


def scene_digest(scene):
    digest = hashlib.sha256()
    for path in sorted(scene.rglob('*')):
        if path.is_file():
            digest.update(str(path.relative_to(scene)).encode())
            digest.update(fingerprint(path).encode())
    return digest.hexdigest()


def build_protocol(cfg):
    manifest = json.loads((ROOT/cfg['manifest']).read_text())
    original = manifest['records']
    train = [r for r in original if r['split'] == 'train']
    dev = [r for r in original if r['split'] == 'val']
    cylinder = spaced([r for r in train if r['source_group'] == 'mouse_climb_wall_91'], cfg['cylinder_train_scenes'])
    cylinder_dev = spaced([r for r in dev if r['source_group'] == 'mouse_wash_face_91'], cfg['cylinder_development_scenes'])
    dual = spaced([r for r in train if r['expected_instances'] == 2], cfg['replay_scenes_per_dataset'])
    singles = []
    names = sorted({r['dataset'] for r in train if r['source_group'] == 'moseq_sample10'})
    for name in names:
        singles.extend(spaced([r for r in train if r['dataset'] == name], cfg['replay_scenes_per_dataset']))
    poses = {r['scene']: r for r in cylinder + dual + singles + dev}
    rigs, revisions = {}, {}
    for scene, rec in poses.items():
        rows = cameras(ROOT/scene)
        roles = {r['view_id']: r['orientation'] for r in rows}
        loaded = load_scene(ROOT/scene)
        scale = median_camera_baseline(np.stack([loaded.frames[v].extrinsic for v in rec['views']]))
        actual = base_load_batch(rec)['view_ids'][0]
        rigs[scene] = dict(roles=roles, scale=float(scale), original_order=list(map(int,actual.split(','))))
        revisions[scene] = scene_digest(ROOT/scene)
    records = {}
    def register(rec, order):
        rig = rigs[rec['scene']]
        r = dict(rec, views=order, diagnostic_view_order=order,
                 normalization_scale=rig['scale'], source_revision=revisions[rec['scene']],
                 physical_roles=[rig['roles'][v] for v in order])
        key = hashlib.sha256(json.dumps(r,sort_keys=True).encode()).hexdigest()
        records[key] = r
        return key
    schedules = {a: [] for a in cfg['arms']}
    slots = []
    rng = random.Random(cfg['seed'])
    # Every 16-step block: 8 Cylinder slots (each V twice), 4 dual and 4 single.
    counts = Counter()
    queues = {'cylinder': [], 'dual': [], 'single': []}
    pools = {'cylinder': cylinder, 'dual': dual, 'single': singles}
    for _ in range(cfg['steps']//16):
        block = [('cylinder',v) for v in (1,2,3,4) for _ in range(2)] + [('dual',2)]*4 + [('single',2)]*4
        rng.shuffle(block)
        for kind,n in block:
            if not queues[kind]:
                queues[kind] = list(range(len(pools[kind])))
                rng.shuffle(queues[kind])
            index = queues[kind].pop()
            rec = pools[kind][index]
            rig = rigs[rec['scene']]
            for arm in cfg['arms']:
                order = (cylinder_order(rig['roles'],n,index,arm=='balanced_views') if kind=='cylinder'
                         else rig['original_order'])
                schedules[arm].append([register(rec,order)])
            counts[kind] += 1
            slots.append(dict(kind=kind, views=n))
    assert len(slots) == cfg['steps'] and counts == {'cylinder':400,'dual':200,'single':200}
    evaluation = []
    for rec in dev:
        group = 'old_dual_pair' if rec['expected_instances']==2 else 'old_cylinder_side_pair'
        evaluation.append(dict(key=register(rec,rigs[rec['scene']]['original_order']),group=group))
    for rec in cylinder_dev:
        roles = rigs[rec['scene']]['roles']
        bottom = next(k for k,v in roles.items() if v=='bottom')
        top = next(k for k,v in roles.items() if v=='top')
        side = rigs[rec['scene']]['original_order']
        for group,order in (('bottom_only',[bottom]),('top_only',[top]),('side_only',[side[0]]),
                            ('side_bottom',[side[0],bottom]),('sides_bottom',side+[bottom]),
                            ('sides_top',side+[top]),('sides_bottom_top',side+[bottom,top])):
            evaluation.append(dict(key=register(rec,order),group=group))
    train_sources = sorted({r['source_group'] for r in cylinder+dual+singles})
    dev_sources = sorted({r['source_group'] for r in dev})
    assert not set(train_sources)&set(dev_sources)
    for i in range(cfg['steps']):
        a,b = [records[schedules[arm][i][0]] for arm in cfg['arms']]
        assert a['scene']==b['scene'] and a['image_hw']==b['image_hw'] and len(a['views'])==len(b['views'])
    files = ['tools/train_cylinder_view_balance.py','tools/audit_bottom_view_coverage.py',
             'tools/train_variable_view_lr_control.py','tools/run_depth_pose_ablation.py',
             'mousegeoforge/models/depth_pose.py','mousegeoforge/pilot_metrics.py',
             'vggt-omega/training/finetune_blendedmvs.py','vggt-omega/training/instance_pose_losses.py',
             'vggt-omega/vggt_omega/utils/robust_pose_geometry.py']
    return dict(config=cfg,records=records,arms=schedules,evaluation=evaluation,
                schedule=list(range(cfg['steps'])),slots=slots,train_sources=train_sources,
                development_sources=dev_sources,checkpoint_sha256=fingerprint(ROOT/cfg['checkpoint']),
                encoder_sha256=fingerprint(ROOT/'vggt-omega/vggt_omega_1b_512.pt'),
                manifest_sha256=fingerprint(ROOT/cfg['manifest']),
                code_sha256={p:fingerprint(ROOT/p) for p in files},
                scope='Synthetic same-asset/rig development, no real Cylinder validation; physical-view sampling only',
                camera_policy='Known dataset K/E for diagnostic triangulation; model camera head disabled; attention remains unconditioned',
                triangulation_policy='GT box-assisted instance matching, predicted objectness>=0.5, known labeled in-frame point validity; uniform robust rays; not end-to-end association')


def oriented_batch(record, *, num_views=2):
    batch = base_load_batch(record,num_views=num_views)
    batch['_physical_roles'] = record['physical_roles']
    return batch


def physical_metrics(pred, batch, target):
    metric, assignment = base_metrics(pred,batch,target)
    vcount = batch['images'].shape[1]
    # CPU post-processing avoids thousands of tiny GPU triangulation operations.
    pred = {k:v.detach().cpu() for k,v in pred.items()}
    target = {k:v.detach().cpu() for k,v in target.items()}
    batch = {k:v.detach().cpu() if torch.is_tensor(v) else v for k,v in batch.items()}
    def one(mapping,i):
        return {k:v[:,i:i+1] if torch.is_tensor(v) and v.ndim>=2 and v.shape[1]==vcount else v
                for k,v in mapping.items()}
    parts = {role:[] for role in ('top','bottom','side')}
    for i,role in enumerate(batch['_physical_roles']):
        row,_ = base_metrics(one(pred,i),one(batch,i),one(target,i))
        valid = batch['depth_masks'][0,i] & batch['body_masks'][0,i]
        gt = batch['depths'][0,i][valid].numpy()
        pd = pred['depth'][0,i,...,0][valid].numpy()
        corr = float(spearmanr(gt,pd).statistic) if len(gt)>1 else float('nan')
        row['body_depth_spearman'] = corr if np.isfinite(corr) else None
        parts[role].append(row)
    keys = ('body_keypoint_nme','box_matched_nme','detection_aware_pck_0.1',
            'detection_recall','body_depth_absrel','body_depth_spearman')
    for role,rows in parts.items():
        for key in keys:
            values = [r[key] for r in rows if r[key] is not None]
            metric[f'{role}_{key}'] = float(np.mean(values)) if values else None
    metric['calibrated_body3d_coverage'] = None
    metric['calibrated_body3d_error_native'] = None
    if int(batch['actual_instance_count'][0])==1 and vcount>=2:
        proj = target['intrinsics'][0] @ target['extrinsics'][0,:,:3]
        height,width = batch['images'].shape[-2:]
        qs = [a[0][a[1].index(0)] for a in assignment]
        uv = torch.stack([pred['instance_keypoint_xy'][0,i,q] for i,q in enumerate(qs)]) * torch.tensor([width,height])
        scores = torch.stack([pred['instance_objectness_logits'][0,i,q].sigmoid() for i,q in enumerate(qs)])
        errors = []
        for k in list(range(7))+list(range(14,23)):
            valid = target['keypoint_valid'][0,:,0,k] & (scores>=.5)
            result = triangulate_robust(uv[:,k],proj,valid,huber_delta_px=3.)
            if result.supported:
                errors.append(float((result.xyz-target['keypoint_world_xyz'][0,0,k]).norm())*float(batch['scale'][0]))
        metric['calibrated_body3d_coverage'] = len(errors)/16
        metric['calibrated_body3d_error_native'] = float(np.mean(errors)) if errors else None
    return metric, assignment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--launch',action='store_true')
    args = parser.parse_args()
    if args.launch:
        OUTPUT.mkdir(parents=True,exist_ok=True)
        with (OUTPUT/'run.log').open('a') as log:
            proc = subprocess.Popen([sys.executable,'-u',__file__],cwd=ROOT,stdout=log,
                                    stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(pid=proc.pid,log=str(OUTPUT/'run.log'))))
        return
    engine.build_protocol = build_protocol
    engine.load_batch = oriented_batch
    engine.evaluate_scene = physical_metrics
    engine.run(OUTPUT,json.loads(CONFIG.read_text()),args.prepare_only)


if __name__=='__main__': main()
