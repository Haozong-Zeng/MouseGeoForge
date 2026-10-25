"""Pinned, frozen-feature calibrated depth pilots. Never score sealed scenes."""
from __future__ import annotations
import argparse
import copy
import datetime as dt
from functools import lru_cache
import json
import math
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from scipy.spatial import cKDTree
import torch
from mousegeoforge.models.calibrated_mouse_depth_roi import CalibratedRayDepthROI as CalibratedRayDepth, supervised_loss
from mousegeoforge.semseg_training import digest, rng_state, seed_all, write_json
from calibrated_mouse_data_20261025 import OUT, gpu_lock, points


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['smoke', 'train', 'evaluate'])
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--gradient-weight', type=float, default=0.)
    p.add_argument('--fit-only', action='store_true')
    p.add_argument('--no-cost', action='store_true', help='Matched reference-feature and calibrated volume prior control')
    p.add_argument('--holdout-frame', type=int, choices=[49,97,145])
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--rays', type=int, default=256)
    p.add_argument('--planes', type=int, default=96)
    p.add_argument('--eval-max-scenes', type=int)
    p.add_argument('--feature-mode', choices=['full','roi'], required=True)
    p.add_argument('--roi-cache', type=Path, default=ROOT/'output/calibrated_mouse_resume8h_20261025/roi_cache_v1')
    return p


ACTIVE_MODE='full'
ACTIVE_CACHE=None

@lru_cache(maxsize=64)
def load_scene(index):
    path=(OUT/'depth_features' if ACTIVE_MODE=='full' else ACTIVE_CACHE)/f'{index:06d}.pt'
    item=torch.load(path,map_location='cpu',weights_only=False)
    if 'feature_affines' not in item:item['feature_affines']=torch.eye(3).repeat(3,1,1)
    return item


def tensors(index):
    item = load_scene(index)
    return item, item['features'].cuda().float(), item['K'].cuda().float(), item['E'].cuda().float()


def sample_rays(item, view, rng, rays):
    body, z = item['body'][view].numpy(), item['depth'][view].numpy()
    ys, xs = np.where(body & np.isfinite(z) & (z > 0))
    selection = rng.integers(len(xs), size=rays)
    x, y = xs[selection], ys[selection]
    # Adjacent pairs test actual local depth gradients; no cross-silhouette pairs.
    offset = rng.integers(2, size=rays)
    nx, ny = np.minimum(x + (offset == 0), body.shape[1]-1), np.minimum(y + (offset == 1), body.shape[0]-1)
    pair_valid = body[ny, nx] & (z[ny, nx] > 0) & (np.abs(z[ny, nx]-z[y, x]) <= 20.)
    xy = torch.from_numpy(np.concatenate([np.stack([x, y], -1), np.stack([nx, ny], -1)])).cuda().float()
    target = torch.from_numpy(np.concatenate([z[y, x], z[ny, nx]])).cuda().float()
    return xy, target, torch.from_numpy(pair_valid).cuda()


def save_checkpoint(path, model, ema, optimizer, step, protocol, records):
    temporary = path.with_suffix('.tmp')
    torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), optimizer=optimizer.state_dict(),
                    step=step, protocol=protocol, rng=rng_state(), records=records), temporary)
    temporary.replace(path)


def train(args, manifest, receipt):
    if args.run_dir.exists():
        raise FileExistsError(args.run_dir)
    args.run_dir.mkdir(parents=True)
    seed_all(args.seed)
    train_indices = [i for i, row in enumerate(manifest['scenes']) if row['split'] == 'train' and row['pose_frame'] != args.holdout_frame]
    if args.fit_only or args.command == 'smoke':
        selected = {}
        for index in train_indices:
            row = manifest['scenes'][index]
            key = row['source'], row['pose_frame']
            selected.setdefault(key, []).append(index)
        train_indices = [i for key in sorted(selected) for i in selected[key][:2]]
    assert len(train_indices) >= 8
    model = CalibratedRayDepth(planes=args.planes).cuda()
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.01, foreach=False)
    aabb = manifest['capture_aabb_mm']
    if args.holdout_frame is not None:
        minimum, maximum = np.full(3,np.inf), np.full(3,-np.inf)
        for index in train_indices:
            item=load_scene(index)
            for view in range(3):
                world=points(item['depth'][view].numpy(),item['body'][view].numpy(),item['K'][view].numpy(),item['E'][view].numpy(),stride=64)
                minimum=np.minimum(minimum,world.min(0));maximum=np.maximum(maximum,world.max(0))
        aabb=[np.floor(minimum-50).tolist(),np.ceil(maximum+50).tolist()]
    bounds = torch.tensor(aabb, device='cuda', dtype=torch.float32)
    source = ['mousegeoforge/models/calibrated_mouse_depth_roi.py', Path(__file__).relative_to(ROOT).as_posix()]
    protocol = dict(schema='calibrated-mouse-ray-depth-v1', command=args.command, steps=args.steps,
        seed=args.seed, planes=args.planes, rays=args.rays, fit_only=args.fit_only, feature_mode=args.feature_mode,
        roi_cache_receipt_sha256=digest(args.roi_cache/'cache_receipt.json'),
        cost_enabled=not args.no_cost, holdout_frame=args.holdout_frame,
        gradient_weight=args.gradient_weight, peak_lr=.001, warmup=100, cosine_floor=.1,
        ema_decay=.995, cameras='fixed K/E world-to-camera, mm camera-Z',
        encoder='frozen released VGGT patch_embed block23 cache', train_scene_indices=train_indices,
        capture_aabb_mm=aabb, forward_has_no_gt_inputs=True,
        cache_receipt_sha256=digest(OUT / 'cache_receipt.json'),
        manifest_sha256=digest(OUT / 'data_manifest.json'),
        source_sha256={p: digest(ROOT / p) for p in source},
        scope='new fixed6000 full-vs-predicted-ROI feature pair; all576 TRAIN scenes, no unseen-pose/real-mm/convergence claim',
        began=dt.datetime.now(dt.timezone.utc).isoformat())
    write_json(args.run_dir / 'protocol.json', protocol)
    records, began = [], time.monotonic()
    first_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    optimizer.zero_grad(set_to_none=True)
    for step in range(args.steps):
        rng = np.random.default_rng(args.seed * 1000003 + step)
        index = train_indices[int(rng.integers(len(train_indices)))]; view = int(rng.integers(3))
        item, features, k, e = tensors(index)
        xy, target, pair_valid = sample_rays(item, view, rng, args.rays)
        optimizer.zero_grad(set_to_none=True)
        encoded = model.encode(features)
        prediction, confidence, valid = model(encoded, xy, view, k, e, bounds, use_cost=not args.no_cost,feature_affines=item['feature_affines'].cuda().float())
        loss_z = supervised_loss(prediction[:args.rays], target[:args.rays], valid[:args.rays])
        pairs = pair_valid & valid[:args.rays] & valid[args.rays:]
        gradient = prediction.sum() * 0
        if pairs.any():
            pred_grad = prediction[args.rays:].clamp_min(1).log()-prediction[:args.rays].clamp_min(1).log()
            true_grad = target[args.rays:].clamp_min(1).log()-target[:args.rays].clamp_min(1).log()
            gradient = (pred_grad[pairs]-true_grad[pairs]).abs().mean()
        loss = loss_z + args.gradient_weight * gradient
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite calibrated depth loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
        if step == 0:
            gradients = {n: float(p.grad.norm()) for n, p in model.named_parameters() if p.grad is not None}
            assert gradients['project.0.weight'] > 0 and gradients['score.4.weight'] > 0
            write_json(args.run_dir / 'gradient_receipt.json', gradients)
        lr = .001 * min(1, (step+1)/100) * (.1+.9*.5*(1+math.cos(math.pi*step/args.steps)))
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.step()
        with torch.no_grad():
            for source_param, target_param in zip(model.parameters(), ema.parameters()):
                target_param.lerp_(source_param, .005)
        record = dict(step=step+1, loss=float(loss.detach()), logdepth=float(loss_z.detach()),
            gradient=float(gradient.detach()), rmse_mm=float(((prediction[:args.rays]-target[:args.rays])**2).mean().sqrt().detach()),
            ray_valid_fraction=float(valid.float().mean()), scene=index, reference=view,
            lr=lr, gradient_norm=float(norm), seconds=time.monotonic()-began)
        records.append(record)
        if step == 0 or (step+1) % 25 == 0:
            write_json(args.run_dir / 'progress.json', record)
            print('DEPTH', json.dumps(record), flush=True)
        if (step+1) % 250 == 0 or step+1 == args.steps:
            save_checkpoint(args.run_dir / f'step_{step+1:06d}.pt', model, ema, optimizer, step+1, protocol, records)
    changed = max(float((p-first_state[k]).abs().max()) for k, p in model.state_dict().items())
    assert changed > 0
    # Actual serialization round trip for model/Adam/EMA on the active cached path.
    saved = torch.load(args.run_dir / f'step_{args.steps:06d}.pt', map_location='cuda', weights_only=False)
    restored = CalibratedRayDepth(planes=args.planes).cuda(); restored.load_state_dict(saved['model'])
    restored_optim = torch.optim.AdamW(restored.parameters(), foreach=False); restored_optim.load_state_dict(saved['optimizer'])
    with torch.no_grad():
        a, _, _ = model(model.encode(features), xy, view, k, e, bounds, use_cost=not args.no_cost,feature_affines=item['feature_affines'].cuda().float())
        b, _, _ = restored(restored.encode(features), xy, view, k, e, bounds, use_cost=not args.no_cost,feature_affines=item['feature_affines'].cuda().float())
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    # Discard these verification updates. Compare restored optimizer moments,
    # parameter groups, gradients and the next update on identical real rays.
    next_losses = []
    for candidate, candidate_optimizer in [(model, optimizer), (restored, restored_optim)]:
        candidate_optimizer.zero_grad(set_to_none=True)
        prediction, _, valid = candidate(candidate.encode(features), xy, view, k, e, bounds, use_cost=not args.no_cost,feature_affines=item['feature_affines'].cuda().float())
        next_loss = supervised_loss(prediction[:args.rays], target[:args.rays], valid[:args.rays])
        keep = pair_valid & valid[:args.rays] & valid[args.rays:]
        if keep.any():
            pg = prediction[args.rays:].clamp_min(1).log()-prediction[:args.rays].clamp_min(1).log()
            tg = target[args.rays:].clamp_min(1).log()-target[:args.rays].clamp_min(1).log()
            next_loss = next_loss + args.gradient_weight * (pg[keep]-tg[keep]).abs().mean()
        next_loss.backward()
        torch.nn.utils.clip_grad_norm_(candidate.parameters(), 10., error_if_nonfinite=True)
        candidate_optimizer.step()
        next_losses.append(float(next_loss.detach()))
    difference = max(float((p-restored.state_dict()[name]).abs().max()) for name, p in model.state_dict().items())
    assert abs(next_losses[0]-next_losses[1]) < 1e-7 and difference < 1e-6
    write_json(args.run_dir / 'completion.json', dict(passed=True, numerical_training_only=True,
        steps=args.steps, seconds=time.monotonic()-began, frozen_encoder_online=False,
        max_parameter_change=changed, checkpoint_forward_roundtrip_exact=True,
        full_next_update_resume_tested=True, next_update_parameter_max_difference=difference,
        train_fit_only=args.fit_only,
        initial_25_mean_logdepth=float(np.mean([x['logdepth'] for x in records[:25]])),
        final_25_mean_logdepth=float(np.mean([x['logdepth'] for x in records[-25:]])),
        peak_gpu_memory_bytes=torch.cuda.max_memory_allocated()))


def cloud_metrics(predicted, target):
    if not len(target):
        return dict(failed=True, reason='empty truth')
    if not len(predicted):
        return dict(accuracy_mm=None, completeness_mm=None, fscore={str(t): 0. for t in (1,2,5)}, failed=True)
    d_pred = cKDTree(target).query(predicted, workers=2)[0]
    d_true = cKDTree(predicted).query(target, workers=2)[0]
    result = dict(accuracy_mm=float(d_pred.mean()), completeness_mm=float(d_true.mean()), fscore={}, failed=False)
    for t in (1,2,5):
        precision, recall = float((d_pred <= t).mean()), float((d_true <= t).mean())
        result['fscore'][str(t)] = 2*precision*recall/max(precision+recall, 1e-12)
    return result


def write_ply(path, world):
    np.savetxt(path, world, fmt='%.6f', header='ply\nformat ascii 1.0\nelement vertex '+str(len(world))+
               '\nproperty float x\nproperty float y\nproperty float z\nend_header', comments='')


def evaluate(args, manifest):
    if not args.checkpoint:
        raise ValueError('--checkpoint required')
    if args.run_dir.exists():
        raise FileExistsError(args.run_dir)
    args.run_dir.mkdir(parents=True)
    saved = torch.load(args.checkpoint, map_location='cuda', weights_only=False)
    model = CalibratedRayDepth(planes=saved['protocol']['planes']).cuda().eval()
    assert saved['protocol']['feature_mode']==args.feature_mode
    model.load_state_dict(saved['ema'])
    bounds = torch.tensor(saved['protocol']['capture_aabb_mm'], device='cuda', dtype=torch.float32)
    indices = [i for i, s in enumerate(manifest['scenes']) if s['split'] == 'dev']
    if args.fit_only:
        indices = saved['protocol']['train_scene_indices']
    if args.eval_max_scenes:
        # Deterministic stratified subset, not only one motion/rig prefix.
        indices = [indices[i] for i in np.linspace(0, len(indices)-1, min(args.eval_max_scenes, len(indices))).astype(int)]
    results, began = [], time.monotonic()
    with torch.inference_mode():
        for order, index in enumerate(indices):
            item, features, k, e = tensors(index)
            encoded = model.encode(features)
            per_view, cloud_pred, cloud_gt = [], [], []
            for view in range(3):
                target = item['depth'][view].numpy(); body = item['body'][view].numpy()
                # Oracle masks select evaluation support only, never model inputs.
                y, x = np.where(body)
                xy = torch.from_numpy(np.stack([x,y], -1)).cuda().float()
                outputs, confidences, validity = [], [], []
                for chunk in xy.split(512):
                    z, conf, valid = model(encoded, chunk, view, k, e, bounds, use_cost=saved['protocol']['cost_enabled'],feature_affines=item['feature_affines'].cuda().float())
                    outputs.append(z.cpu()); confidences.append(conf.cpu()); validity.append(valid.cpu())
                z = torch.cat(outputs).numpy(); valid = torch.cat(validity).numpy()
                assert np.isfinite(z).all()
                prediction = np.zeros_like(target); prediction[y[valid], x[valid]] = z[valid]
                residual = prediction[body]-target[body]
                per_view.append(dict(view=view, bottom=view==0, pixels=int(body.sum()),
                    valid_fraction=float(valid.mean()), rmse_mm=float(np.sqrt(np.mean(residual**2))),
                    mae_mm=float(np.abs(residual).mean()), absrel=float((np.abs(residual)/target[body]).mean()),
                    within_1mm=float((np.abs(residual)<=1).mean()), within_2mm=float((np.abs(residual)<=2).mean()),
                    within_5mm=float((np.abs(residual)<=5).mean())))
                k_np, e_np = k[view].cpu().numpy(), e[view].cpu().numpy()
                cloud_pred.append(points(prediction, body & (prediction>0), k_np, e_np))
                cloud_gt.append(points(target, body, k_np, e_np))
                if order < 4:
                    np.savez_compressed(args.run_dir / f'scene{index:06d}_view{view}.npz',
                        depth_mm=prediction, oracle_evaluation_mask=body, K=k_np, E=e_np,
                        confidence_on_oracle_support=torch.cat(confidences).numpy(),
                        note='Oracle-support diagnostic, not full-frame automatic segmentation/depth')
                    low, high = np.percentile(target[body], [1,99]); span=max(1., high-low)
                    pred_color = cv2.applyColorMap(np.clip((prediction-low)/span*255,0,255).astype('uint8'),cv2.COLORMAP_TURBO)
                    gt_color = cv2.applyColorMap(np.clip((target-low)/span*255,0,255).astype('uint8'),cv2.COLORMAP_TURBO)
                    pred_color[~body] = 0; gt_color[~body] = 0
                    cv2.imwrite(str(args.run_dir/f'scene{index:06d}_view{view}_gt_prediction.png'),np.concatenate([gt_color,pred_color],axis=1))
            predicted, truth = np.concatenate(cloud_pred), np.concatenate(cloud_gt)
            # Full observed surfaces; no global or per-view GT similarity fitting.
            geometry = cloud_metrics(predicted, truth)
            result = dict(id=item['scene']['id'], rig=item['scene']['rig'], source=item['scene']['source'],
                          condition=item['scene']['condition'], pose_frame=item['scene']['pose_frame'],
                          source_frame_held_out=item['scene']['pose_frame']==saved['protocol']['holdout_frame'],
                          views=per_view, pointcloud=geometry)
            results.append(result)
            if order < 4:
                write_ply(args.run_dir / f'scene{index:06d}_predicted_world_mm.ply', predicted)
                write_ply(args.run_dir / f'scene{index:06d}_truth_world_mm.ply', truth)
            if (order+1) % 6 == 0:
                write_json(args.run_dir/'progress.json',dict(completed=order+1,total=len(indices),seconds=time.monotonic()-began))
                print('EVAL', order+1, len(indices), flush=True)
    metrics = [v for r in results for v in r['views']]
    summary = {key: float(np.mean([m[key] for m in metrics])) for key in ['rmse_mm','mae_mm','absrel','within_1mm','within_2mm','within_5mm','valid_fraction']}
    summary['pointcloud_fscore'] = {str(t): float(np.mean([r['pointcloud']['fscore'][str(t)] for r in results])) for t in (1,2,5)}
    breakdown={}
    for held in [False,True]:
        subset=[r for r in results if r['source_frame_held_out']==held]
        vv=[v for r in subset for v in r['views']]
        if vv:
            breakdown['held_out_source_frame' if held else 'seen_source_frame']=dict(scenes=len(subset),mean_view_rmse_mm=float(np.mean([v['rmse_mm'] for v in vv])),pointcloud_fscore={str(t):float(np.mean([r['pointcloud']['fscore'][str(t)] for r in subset])) for t in (1,2,5)})
    write_json(args.run_dir / 'evaluation.json', dict(checkpoint_sha256=digest(args.checkpoint), weights='EMA',
        breakdown=breakdown, cost_enabled=saved['protocol']['cost_enabled'], holdout_frame=saved['protocol']['holdout_frame'],
        split='TRAIN_fit_diagnostic' if args.fit_only else 'DEV', source_sha256=saved['protocol']['source_sha256'],
        evaluation_source_sha256=digest(Path(__file__)), oracle_mask_support=True, mask_input_to_depth=False,
        inference_uses_fixed_capture_aabb=True, ground_truth_scale_alignment=False,
        real_mm_accuracy_claim=False, unseen_pose_claim=False, failure_denominator=len(indices),
        summary=summary, scenes=results, seconds=time.monotonic()-began))
    print(json.dumps(summary), flush=True)


def main():
    global ACTIVE_MODE,ACTIVE_CACHE
    args = parser().parse_args();ACTIVE_MODE=args.feature_mode;ACTIVE_CACHE=args.roi_cache
    torch.set_num_threads(4); cv2.setNumThreads(1)
    manifest = json.loads((OUT / 'data_manifest.json').read_text())
    receipt = json.loads((OUT / 'cache_receipt.json').read_text())
    roi_receipt=json.loads((args.roi_cache/'cache_receipt.json').read_text());assert roi_receipt['passed'] and len(roi_receipt['files'])==720
    for row in roi_receipt['files']:assert digest(ROOT/row['path'])==row['sha256']
    assert receipt['manifest_sha256'] == digest(OUT / 'data_manifest.json')
    for row in receipt['files']:
        assert digest(ROOT / row['path']) == row['sha256']
    with gpu_lock():
        if args.command == 'evaluate':
            evaluate(args, manifest)
        else:
            if args.command == 'smoke':
                args.steps = 3
            train(args, manifest, receipt)


if __name__ == '__main__':
    main()
