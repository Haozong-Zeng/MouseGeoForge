"""Frozen synthetic Cylinder diagnostic; never a substitute for missing real data."""
import copy
import json
from collections import defaultdict

import numpy as np
import torch
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from run_depth_pose_ablation import (ROOT, VGGTOmega, DepthPoseModel, load_batch,
    restore, fingerprint, write_json, instance_targets_from_batch)
from training.finetune_blendedmvs import load_scene, median_camera_baseline
from mousegeoforge.pilot_metrics import evaluate_scene
from vggt_omega.utils.pose_enc import encoding_to_camera

OUT = ROOT / 'output/bottom_view_audit_v1/cylinder_diagnostic'
CONDITIONS = {'side_pair': [2, 0], 'side_plus_bottom': [2, 0, 6],
              'side_plus_top': [2, 0, 7], 'bottom_only': [6],
              'side_bottom_pair': [2, 6]}


def slice_views(mapping, idx, nviews):
    return {k: v[:, idx:idx+1] if torch.is_tensor(v) and v.ndim >= 2 and v.shape[1] == nviews else v
            for k, v in mapping.items()}


def angle(a, b):
    return float(np.degrees(np.arccos(np.clip((np.trace(a @ b.T) - 1) / 2, -1, 1))))


def main():
    torch.set_num_threads(2)
    torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    OUT.mkdir(parents=True, exist_ok=True)
    run = ROOT / 'output/pose_diverse_core23_768_v1'
    records = [r for r in json.loads((run / 'manifest.json').read_text())['records']
               if r['split'] == 'val' and r['expected_instances'] == 1]
    records = [records[i] for i in (0, 8, 15)]
    reference = {r['scene']: r['metrics'] for r in json.loads(
        (run / 'seed17_off/validation.json').read_text())['records']}
    base = ROOT / 'vggt-omega/vggt_omega_1b_512.pt'
    head = run / 'seed17_off/final_heads.pt'
    protocol = dict(scope='Three held-out-motion SYNTHETIC Cylinder frames, not real Cylinder',
                    frames=[r['frame'] for r in records], conditions=CONDITIONS,
                    resolution=[768, 768], encoder_sha256=fingerprint(base),
                    head_sha256=fingerprint(head), weights_updated=False,
                    normalization='Fixed original side-pair [2,0] baseline for every condition',
                    oracle_depth_note='Pretrained body depth is median scale-aligned separately per view; not metric depth accuracy.',
                    limits=['No real Cylinder data available', 'Three frames, one synthetic asset/rig',
                            'Bottom and top are controlled for view count, not identical visual overlap',
                            'Current heads trained for two-view context; input V=3 is a distribution change'])
    write_json(OUT / 'protocol.json', protocol)
    spatial = VGGTOmega(enable_multi_instance_pose=True, num_keypoints=23)
    state = torch.load(base, map_location='cpu', weights_only=True, mmap=True)
    loaded = spatial.load_state_dict(state, strict=False)
    assert not loaded.unexpected_keys and all(k.startswith('instance_pose_head.') for k in loaded.missing_keys)
    del state
    original_dense = copy.deepcopy(spatial.dense_head).cuda().eval()
    camera_head = spatial.camera_head.cuda().eval()
    model = DepthPoseModel.from_spatial_model(spatial).cuda().eval()
    checkpoint = torch.load(head, map_location='cpu', weights_only=True)
    assert checkpoint['checkpoint_sha256'] == protocol['encoder_sha256']
    restore(model, checkpoint['model'])
    del checkpoint
    allrows = []
    for ri, source in enumerate(records):
        scene = load_scene(ROOT / source['scene'])
        # Keep the original checkpoint's side-pair depth target scale unchanged.
        # Never renormalize when adding top/bottom or removing a camera.
        scale = median_camera_baseline(np.stack([scene.frames[v].extrinsic for v in (2, 0)]))
        for condition, views in CONDITIONS.items():
            record = dict(source, split='synthetic_bottom_diagnostic', views=views,
                          diagnostic_view_order=views, normalization_scale=float(scale))
            batch = load_batch(record, num_views=len(views))
            targets = instance_targets_from_batch(batch)
            images = batch['images'].cuda()
            with torch.inference_mode():
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    cached, start = model.aggregator(images)
                # Both heads consume exactly the same unadapted frozen RGB features.
                original_depth, conf = original_dense(cached, images, start)
                pose_enc = camera_head(cached, patch_token_start=start)
                ep, _ = encoding_to_camera(pose_enc, images.shape[-2:])
                output = {k: v.cpu() for k, v in model.forward_cached(images, cached, start).items()}
            original_depth = original_depth[0, ..., 0].cpu().numpy()
            ep = ep[0].cpu().numpy()
            gt_e = batch['extrinsics'][0].numpy()
            metrics, assignments = evaluate_scene(output, batch, targets)
            if condition == 'side_pair':
                parity = max(abs(v-reference[source['scene']][k]) for k, v in metrics.items() if v is not None)
                assert parity < 1e-6, ('Original two-view parity failed', parity)
            rows = []
            for j, camera in enumerate(views):
                metric, _ = evaluate_scene(slice_views(output, j, len(views)),
                                          slice_views(batch, j, len(views)),
                                          slice_views(targets, j, len(views)))
                gt = batch['depths'][0, j].numpy()
                body = (batch['depth_masks'][0, j] & batch['body_masks'][0, j]).numpy()
                valid = body & np.isfinite(gt) & (gt > 0) & np.isfinite(original_depth[j]) & (original_depth[j] > 0)
                assert valid.sum() > 20
                scalar = float(np.median(gt[valid] / original_depth[j][valid]))
                aligned = original_depth[j] * scalar
                corr = float(spearmanr(gt[valid], original_depth[j][valid]).statistic)
                current_d = output['depth'][0,j,...,0].numpy()
                current_corr = float(spearmanr(gt[valid], current_d[valid]).statistic)
                world = targets['keypoint_world_xyz'][0,0]
                homo = torch.cat((world, torch.ones_like(world[:,:1])), dim=-1)
                projected = homo @ (batch['intrinsics'][0,j] @ batch['extrinsics'][0,j,:3]).T
                uv = projected[:,:2] / projected[:,2:3]
                kp_valid = targets['keypoint_valid'][0,j,0]
                projection_error = float((uv-targets['keypoint_xy'][0,j,0]*768).norm(dim=-1)[kp_valid].max())
                assert projection_error < .01, ('Camera/label mismatch', projection_error)
                gt_rel = gt_e[j, :3, :3] @ gt_e[0, :3, :3].T
                pred_rel = ep[j, :3, :3] @ ep[0, :3, :3].T
                row = dict(frame=record['frame'], condition=condition, camera=camera,
                           orientation='bottom' if camera == 6 else 'top' if camera == 7 else 'side',
                           current=metric, original_body_oracle_scale=scalar,
                           original_body_scale_aligned_absrel=float(np.mean(np.abs(aligned[valid]-gt[valid])/gt[valid])),
                           original_body_depth_spearman=corr if np.isfinite(corr) else None,
                           current_body_depth_spearman=current_corr if np.isfinite(current_corr) else None,
                           constant_gt_median_body_absrel=float(np.mean(np.abs(np.median(gt[valid])-gt[valid])/gt[valid])),
                           gt_keypoint_projection_max_px=projection_error,
                           original_relative_rotation_error_deg=angle(pred_rel, gt_rel) if j else None,
                           body_pixels=int(valid.sum()), gt_body_depth_p05_p95=np.percentile(gt[valid], [5,95]).tolist())
                rows.append(row)
            allrows.extend(rows)
            write_json(OUT / 'per_view.json', allrows)
            if ri == 1:
                fig, axes = plt.subplots(len(views), 4, squeeze=False,
                                         figsize=(12, 3.5 * len(views)), layout='constrained')
                for j, camera in enumerate(views):
                    gt = batch['depths'][0, j].numpy()
                    body = (batch['depth_masks'][0, j] & batch['body_masks'][0, j]).numpy()
                    lo, hi = np.percentile(gt[body], [1, 99])
                    rgb = batch['images'][0, j].permute(1, 2, 0).numpy()
                    axes[j, 0].imshow(rgb)
                    qs, ms = assignments[j]
                    for q, m in zip(qs, ms):
                        valid = targets['keypoint_valid'][0, j, m].numpy()
                        p = output['instance_keypoint_xy'][0, j, q].numpy() * 768
                        t = targets['keypoint_xy'][0, j, m].numpy() * 768
                        axes[j, 0].scatter(t[valid,0], t[valid,1], s=8, c='lime', label='GT')
                        axes[j, 0].scatter(p[valid,0], p[valid,1], s=8, c='magenta', marker='x', label='pred')
                    axes[j,0].set_title(f'camera {camera} RGB + pose\nGT green / prediction magenta')
                    for k, (d, title) in enumerate(((gt, 'GT body depth'),
                            (original_depth[j]*rows[j]['original_body_oracle_scale'], 'Pretrained (oracle body scale)'),
                            (output['depth'][0,j,...,0].numpy(), 'Current trained head')), 1):
                        im = axes[j,k].imshow(np.where(body, d, np.nan), cmap='viridis', vmin=lo, vmax=hi)
                        axes[j,k].set_title(title)
                        fig.colorbar(im, ax=axes[j,k], shrink=.6)
                    for ax in axes[j]:
                        ax.set_xlim(0,768); ax.set_ylim(768,0); ax.axis('off')
                fig.suptitle(f'SYNTHETIC Cylinder frame {record["frame"]}: {condition}\nDepth shown on GT body mask, fixed reference-pair baseline units')
                fig.savefig(OUT / f'{condition}.png', dpi=120)
                plt.close(fig)
            print(f'Completed {ri+1}/3 frame={record["frame"]} condition={condition}', flush=True)
            del cached, output, images, original_depth, pose_enc
    grouped = defaultdict(list)
    for row in allrows:
        grouped[(row['condition'], row['orientation'])].append(row)
    summary = []
    for (condition, orientation), rows in grouped.items():
        summary.append(dict(condition=condition, orientation=orientation, images=len(rows),
            body_nme=float(np.mean([r['current']['body_keypoint_nme'] for r in rows])),
            pck=float(np.mean([r['current']['detection_aware_pck_0.1'] for r in rows])),
            detection_recall=float(np.mean([r['current']['detection_recall'] for r in rows])),
            body_depth_absrel=float(np.mean([r['current']['body_depth_absrel'] for r in rows])),
            original_body_oracle_absrel=float(np.mean([r['original_body_scale_aligned_absrel'] for r in rows]))))
    assert len(allrows) == 33
    write_json(OUT / 'summary.json', dict(protocol=protocol, summary=summary,
        checks=dict(original_side_pair_metric_parity=True,
                    gt_keypoint_projection_max_px=max(r['gt_keypoint_projection_max_px'] for r in allrows),
                    completed_forwards=15, per_view_records=len(allrows))))
    assert fingerprint(head) == protocol['head_sha256']
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
