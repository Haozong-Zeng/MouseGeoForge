"""Bounded GPU integration smoke on explicitly selected, fully annotated scenes.

No dataset writes or production checkpoint overwrite. Not an ablation trainer:
no split selection, temporal windows, pseudo-depth or partial detection labels.
Writes a small diagnostic report, not learned weights or accuracy claims.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'vggt-omega')]
import torch
from torch.utils.data import default_collate
from vggt_omega.models import VGGTOmega
from training.finetune_blendedmvs import BlendedMVSDataset, load_scene, depth_loss
from training.instance_pose_losses import instance_targets_from_batch, instance_pose_loss, InstancePoseLossWeights
from mousegeoforge.models import DepthPoseModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--scene', action='append', type=Path, required=True)
    parser.add_argument('--steps-per-scene', type=int, default=2)
    parser.add_argument('--coupling', choices=['off', 'local_depth_features'], default='local_depth_features')
    parser.add_argument('--pfm-flip-y', action=argparse.BooleanOptionalAction, default=True,
                        help='Audited Blender exports use bottom-up PFM; do not assume for real datasets')
    parser.add_argument('--report', type=Path, default=ROOT/'output/depth_pose_joint_smoke/report.json')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('This integration smoke requires CUDA')
    if not 1 <= args.steps_per_scene <= 5:
        raise ValueError('Smoke is limited to 1..5 updates per scene; use a trainer for experiments')
    torch.manual_seed(17)
    torch.set_num_threads(2)
    scenes = [load_scene(p.resolve()) for p in args.scene]
    views = min(2, min(len(s.frames) for s in scenes))
    ds = BlendedMVSDataset(scenes, views, 128, 128, 1, 17, False, args.pfm_flip_y,
                          False, 'median_baseline', views, True)
    if len(ds.keypoint_names) != 23:
        raise ValueError(f'Smoke requires Core23 only, found {len(ds.keypoint_names)} channels')
    ds.max_instances = max(2, ds.max_instances)
    spatial = VGGTOmega(enable_camera=False, enable_depth=True,
                        enable_multi_instance_pose=True, num_keypoints=23)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True, mmap=True)
    if not isinstance(checkpoint, dict) or not all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        raise ValueError('Expected original raw tensor state dict')
    prefixes = ('aggregator.', 'dense_head.')
    expected = {k for k in spatial.state_dict() if k.startswith(prefixes)}
    selected = {k: v for k, v in checkpoint.items() if k.startswith(prefixes)}
    if set(selected) != expected:
        raise ValueError(f'Upstream key mismatch: missing={len(expected-set(selected))}, extra={len(set(selected)-expected)}')
    loaded = spatial.load_state_dict(selected, strict=False)
    assert not loaded.unexpected_keys
    assert all(k.startswith('instance_pose_head.') for k in loaded.missing_keys)
    print(f'Loaded all {len(expected)} encoder/depth tensors; instance head is newly initialized.', flush=True)
    del selected, checkpoint
    model = DepthPoseModel.from_spatial_model(spatial, coupling=args.coupling).cuda().train()
    del spatial
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    depth_args = SimpleNamespace(semantic_label_weights='', body_depth_balance=.5,
                                 body_depth_fraction=.5, depth_gradient_weight=0.)
    pose_weights = InstancePoseLossWeights(keypoint_3d=0., reprojection=0.)
    records = []
    for index, scene in enumerate(scenes):
        batch = default_collate([ds[index]])
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        if not batch['depth_available'].all() or not batch['instance_valid'].any():
            raise ValueError('Selected smoke scenes need complete depth and annotated instances')
        targets = instance_targets_from_batch(batch)
        for step in range(args.steps_per_scene):
            optimizer.zero_grad(set_to_none=True)
            output = model(batch['images'])
            d = depth_loss(output['depth'], batch['depths'], batch['depth_masks'],
                           batch['body_masks'], batch['semantic_masks'], depth_args)['loss_depth']
            p, _ = instance_pose_loss(output, targets, pose_weights, image_size=(128, 128))
            total = d + p
            if not torch.isfinite(total):
                raise RuntimeError('Nonfinite joint loss')
            total.backward()
            gradients = {}
            for name in ('shared_adapter', 'dense_head', 'instance_pose_head', 'depth_pose_bridge'):
                module = getattr(model, name)
                grads = [v.grad for v in module.parameters() if v.grad is not None]
                assert all(torch.isfinite(g).all() for g in grads)
                gradients[name] = sum(float(g.abs().sum()) for g in grads)
            assert all(v.grad is None for v in model.aggregator.parameters())
            assert all(gradients[k] > 0 for k in ('shared_adapter', 'dense_head', 'instance_pose_head'))
            if args.coupling == 'local_depth_features':
                assert gradients['depth_pose_bridge'] > 0
            else:
                assert gradients['depth_pose_bridge'] == 0
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            record = {'scene': str(scene.root.relative_to(ROOT)), 'step': step,
                      'instances': int(batch['actual_instance_count'][0]), 'views': views,
                      'image_hw': [128, 128], 'depth_loss': float(d.detach()),
                      'pose_loss': float(p.detach()), 'gradient_l1': gradients}
            records.append(record)
            print(json.dumps(record), flush=True)
    report = {'passed': True, 'purpose': 'integration_smoke_not_accuracy_evaluation',
              'coupling': args.coupling, 'checkpoint': str(args.checkpoint),
              'loaded_encoder_depth_tensors': len(expected), 'encoder_frozen': True,
              'detach_depth_features': True, 'new_instance_head': True,
              'pfm_flip_y': args.pfm_flip_y,
              'camera_depth_scale': 'same_static_rig_median_baseline_per_scene',
              'trained_weights_saved': False, 'records': records,
              'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated()}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(f'PASS report={args.report}', flush=True)


if __name__ == '__main__':
    main()
