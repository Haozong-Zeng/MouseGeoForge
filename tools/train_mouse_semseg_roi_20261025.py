"""Predicted-ROI vs full native-center paired engine; torchrun DDP on one/multiple GPUs.

smoke is engineering-only. This engine accepts explicit pinned, fixed-update
arms; it does not automatically register/launch the route's complete paired
convergence/selection campaign. A Slurm template is provided separately.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from mousegeoforge.semseg_pixel_center_20261025 import label_letterbox,native_label
from mousegeoforge.semseg_roi_20261025 import read_roi_sample,roi_sample,native_probability
from mousegeoforge.models.mouse_semseg import hungarian_loss, semantic_scores
from mousegeoforge.semseg_training import (BalancedSchedule, EMA, MouseSemseg, ROOT, configure_stage,
    digest, letterbox, load_checkpoint, load_items, make_encoder, mask_metrics, native_mask,
    optimizer_for, read_sample, rng_state, save_checkpoint, seed_all, set_learning_rates, write_json)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare', 'train', 'smoke', 'evaluate'])
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--items', type=Path)
    p.add_argument('--output-manifest', type=Path)
    p.add_argument('--run-dir', type=Path)
    p.add_argument('--checkpoint', type=Path, default=ROOT / 'vggt-omega/vggt_omega_1b_512.pt')
    p.add_argument('--resume', type=Path)
    p.add_argument('--parent', type=Path)
    p.add_argument('--arm', choices=['M'], default='M')
    p.add_argument('--crop-mode',choices=['full','roi'],required=True)
    p.add_argument('--label-policy', choices=['legacy','center_exact'], required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--precision', choices=['fp32', 'bf16'], default='bf16')
    p.add_argument('--fixture', action='store_true', help='Tiny random encoder/data; never release evidence')
    p.add_argument('--side', type=int, default=512)
    p.add_argument('--steps', type=int)
    p.add_argument('--stop-after', type=int, help='Pause at this update without changing the full registered LR schedule')
    p.add_argument('--global-batch', type=int, default=8)
    p.add_argument('--micro-batch', type=int, default=1)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--save-every', type=int, default=500)
    p.add_argument('--eval-every', type=int, default=500)
    p.add_argument('--allow-unregistered-fixed-budget', action='store_true',
                   help='Explicit execution of a fixed-budget arm, not the entire design-only campaign')
    return p


def fixture_items():
    return [dict(id=f'fixture{i}', split='train', domain=domain, bottom=bottom,
                 group='fixture', cohort='fixture', rig='fixture')
            for i, (domain, bottom) in enumerate([('real', True), ('real', False),
                                                 ('synthetic', True), ('synthetic', False)])]


def fixture_sample(row, side, mode):
    rng = np.random.default_rng(int(row['id'][-1]) + 123)
    image = rng.integers(0, 80, (side // 2 + 3, side - 7, 3), dtype=np.uint8)
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.ellipse(mask, (side // 2, side // 4), (side // 5, side // 8), 15, 0, 360, 1, -1)
    cv2.line(mask, (side // 2, side // 4), (side - 4, side // 3), 1, 1)
    image[mask > 0] += 140
    h,w=image.shape[:2]
    result=roi_sample(image,mask,side,mode,[2,2,w-2,h-2])
    result[2]['fixture_original_label']=mask
    return result


def prepare(args):
    if not args.items or not args.output_manifest:
        raise ValueError('prepare requires --items and --output-manifest')
    if args.output_manifest.exists():
        raise FileExistsError(args.output_manifest)
    rows = load_items(args.items, args.root, verify=True)
    for row in rows:
        row['label_policy'] = ('manual_visible_mouse' if row['domain'] == 'real' else 'synthetic_skin_surrogate')
        row.setdefault('label_encoding', 'binary_0_255')
    write_json(args.output_manifest, dict(schema='mouse-semseg-v1', items=rows,
        source_manifest_sha256=digest(args.items), classes=dict(background=0, mouse=1, ignore=255),
        no_new_source_admission=True))
    print('PREPARED', len(rows), args.output_manifest, flush=True)


def setup_distributed(args):
    rank, world, local = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1)), int(os.environ.get('LOCAL_RANK', 0))
    if args.device == 'cuda':
        torch.cuda.set_device(local)
        device = torch.device('cuda', local)
    else:
        device = torch.device('cpu')
    if world > 1:
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo',
                                timeout=datetime.timedelta(minutes=5))
    return rank, world, device


def gather_rng(world):
    state = rng_state()
    if world == 1:
        return [state]
    output = [None] * world
    dist.all_gather_object(output, state)
    return output


def get_rows(args, rank):
    if args.fixture:
        return fixture_items(), 'fixture-v1'
    if args.items is None:
        raise ValueError('--items required for real data')
    rows = load_items(args.items, args.root, verify=False)
    if args.command == 'smoke':
        rows = [next(x for x in rows if x['split'] == 'train' and (x['domain'], x['bottom']) == stratum)
                for stratum in [('real', True), ('real', False), ('synthetic', True), ('synthetic', False)]]
    # Rank zero checks immutable bytes once on the shared filesystem.
    if rank == 0:
        for row in rows:
            for key, pin in [('image', 'image_sha256'), ('label', 'label_sha256')]:
                if digest(args.root / row[key]) != row[pin]:
                    raise ValueError('Input pin mismatch: ' + row[key])
    if dist.is_initialized():
        dist.barrier()
    identity = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    return rows, identity


def evaluate_rows(model, rows, args, device, arm, pool, save_dir=None):
    metrics = []
    for row in rows:
        image, label, meta = (fixture_sample(row, args.side,args.crop_mode) if args.fixture else read_roi_sample(row, args.root, args.side,args.crop_mode))
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16,
                                             enabled=device.type == 'cuda' and args.precision == 'bf16'):
            predictions = model(image[None].to(device), arm)
            loss = hungarian_loss(predictions, label[None].to(device))
            scores = semantic_scores(predictions[-1], (args.side, args.side))
            mask = scores.argmax(1)[0].cpu().numpy()
            probability = (scores[:, 1] / scores.sum(1).clamp_min(1e-12))[0].cpu().numpy()
        y, x, h, w = meta['roi']
        roi = np.zeros_like(mask, bool); roi[y:y + h, x:x + w] = True
        target = label.numpy()
        # Letterboxed ROI has native aspect at max-side512; native metric separately.
        ref_pred, ref_gt = mask[y:y + h, x:x + w], target[y:y + h, x:x + w]
        ref_scale = 512 / max(h, w)
        hw = (max(1, round(w * ref_scale)), max(1, round(h * ref_scale)))
        ref_pred = cv2.resize(ref_pred.astype(np.uint8), hw, interpolation=cv2.INTER_NEAREST)
        ref_gt = cv2.resize(ref_gt.astype(np.uint8), hw, interpolation=cv2.INTER_NEAREST)
        native_prob,native_pred=native_probability(probability,meta)
        native_gt=meta['fixture_original_label'] if args.fixture else native_label(row,args.root)
        row_metrics = dict(id=row['id'], domain=row['domain'], bottom=row['bottom'], group=row['group'],
            loss=float(loss), native=mask_metrics(native_pred, native_gt, 1),
            ref512_tol1=mask_metrics(ref_pred, ref_gt, 1), ref512_tol2=mask_metrics(ref_pred, ref_gt, 2),
            padding_foreground_pixels=int((mask[~roi] == 1).sum()))
        metrics.append(row_metrics)
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_dir / (row['id'] + '.png')), native_pred.astype(np.uint8) * 255)
            native_probability_export=native_prob
            np.savez_compressed(save_dir / (row['id'] + '_probability.npz'),
                                foreground=native_probability_export.astype(np.float32),
                                pixel_center_affine=np.array(meta['pixel_center_affine']))
            if len(metrics) <= 4:
                rgb = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                truth_rgb = np.repeat((target == 1)[..., None] * 255, 3, axis=2).astype(np.uint8)
                pred_rgb = np.repeat((mask == 1)[..., None] * 255, 3, axis=2).astype(np.uint8)
                panel = np.concatenate([rgb, truth_rgb, pred_rgb], axis=1)
                cv2.imwrite(str(save_dir / (row['id'] + '_rgb_gt_prediction.jpg')), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
    return metrics


@contextlib.contextmanager
def gpu_guard(args, rank):
    handle = None
    if rank == 0 and args.device == 'cuda':
        import fcntl
        if os.environ.get('SLURM_JOB_ID'):
            name = f'semseg_{socket.gethostname()}_{os.environ["SLURM_JOB_ID"]}.lock'
            path = args.root / 'output/gpu_locks' / name
        else:
            path = args.root / 'output/teacher_temporal_cycle_20261001/gpu_job.lock'
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open('a+')
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        yield
    finally:
        if handle:
            handle.close()


def run(args):
    smoke = args.command == 'smoke'
    if args.label_policy!='center_exact' or args.precision!='fp32':
        raise ValueError('This ROI version requires center_exact labels and FP32')
    if args.run_dir is None:
        raise ValueError('--run-dir required; historical output directories must not be reused')
    if args.side % 16 or args.side < 64:
        raise ValueError('--side must be a multiple of16 and at least64')
    if args.fixture and not smoke:
        raise ValueError('Fixtures are smoke-only')
    if not smoke and args.command == 'train' and not args.allow_unregistered_fixed_budget:
        raise ValueError('Supply --allow-unregistered-fixed-budget for an explicit arm; design_only is not a trainer config')
    if args.arm in ('F', 'U') and not smoke and not args.parent and not args.resume:
        raise ValueError('F/U require a matched head parent checkpoint')
    if not args.fixture and (not args.parent or digest(args.parent)!='30a50a5cfb67b7639b1d2e24988789a11460b8787ed9eea46963b2e9e148b8dc'):
        raise ValueError('This version accepts only the fixed original M18000 parent')
    rank, world, device = setup_distributed(args)
    torch.set_num_threads(max(1, int(os.environ.get('OMP_NUM_THREADS', '4')))); cv2.setNumThreads(1)
    seed_all(args.seed)
    if args.global_batch % (world * args.micro_batch):
        raise ValueError('global batch must be divisible by WORLD_SIZE * micro batch')
    steps = args.steps if args.steps is not None else (4 if smoke else 6000)
    if steps < 1 or (smoke and steps > 8):
        raise ValueError('smoke capped at8 engineering updates')
    phase_steps = 1 if smoke else 2000
    if rank == 0:
        if args.run_dir.exists() and not args.resume:
            raise FileExistsError(args.run_dir)
        if args.run_dir.exists() and (args.run_dir / 'completion.json').exists() and args.command != 'evaluate':
            raise FileExistsError('Completed run preserved; resume into a new --run-dir')
        args.run_dir.mkdir(parents=True, exist_ok=bool(args.resume))
    if world > 1:
        dist.barrier()
    with gpu_guard(args, rank), ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows, data_hash = get_rows(args, rank)
        train_rows = [r for r in rows if r['split'] == 'train']
        schedule = BalancedSchedule(rows, args.global_batch, args.seed)
        encoder = make_encoder(args.checkpoint, fixture=args.fixture)
        model = MouseSemseg(encoder, dim=32 if args.fixture else 256, queries=4 if args.fixture else 16,
                            heads=4 if args.fixture else 8, pixel_layers=1 if args.fixture else 3,
                            query_layers=1 if args.fixture else 3).to(device)
        head_lr = 3e-5 if args.arm in ('F', 'U') else 1e-5
        configure_stage(model, args.arm, 0, phase_steps)
        optimizer = optimizer_for(model, args.arm, head_lr)
        ema = EMA()
        ema.update(model)
        if args.parent:
            parent = torch.load(args.parent, map_location='cpu', weights_only=False)
            if parent['protocol']['fixture'] or parent['protocol']['encoder_sha256'] != digest(args.checkpoint):
                raise ValueError('Invalid parent encoder identity')
            with torch.no_grad():
                for name, value in parent['ema'].items():
                    if name.startswith('head.'):
                        dict(model.named_parameters())[name].copy_(value)
            ema = EMA(); ema.update(model)
        source_files = [Path(__file__), ROOT/'mousegeoforge/semseg_roi_20261025.py', ROOT/'mousegeoforge/semseg_pixel_center_20261025.py', ROOT / 'mousegeoforge/models/mouse_semseg.py',
                        ROOT / 'mousegeoforge/semseg_training.py']
        protocol = dict(engine='mouse-semseg-predicted-ROI-pair-v1',crop_mode=args.crop_mode,crop_policy='fixed originalM predictions;1.4padding/min96/emptyfull; fullGT denominator', smoke=smoke, fixture=args.fixture,
            label_policy=args.label_policy,native_truth_policy='original pixels; no GT resize',
            native_inverse_policy='FP32 bilinear probability strictly>.5; tie background',
            parent_policy='same originalM18000EMA; frozen released encoder; fresh Adam and EMA at start',
            arm=args.arm, side=args.side, seed=args.seed, world_size=world, global_batch=args.global_batch,
            micro_batch=args.micro_batch, precision=args.precision, steps=steps, phase_steps=phase_steps,
            data_sha256=data_hash, encoder_sha256='tiny-random-fixture' if args.fixture else digest(args.checkpoint),
            parent_sha256=digest(args.parent) if args.parent else None,
            source_sha256={p.relative_to(ROOT).as_posix(): digest(p) for p in source_files},
            roi_warmup=600,roi_cosine_floor=.1,head_phase_peak_lr=1e-5 if args.arm in ('L','M') else 3e-5,
            head_phase_exposure_offset=18000 if args.arm in ('L','M') else 0,
            head_phase_parent_policy='own original18000 EMA; fresh Adam/EMA at phase start, then exact phase resume',
            implementation='pure-torch sparse deformable pixel attention + masked query decoder; dense-grid Hungarian losses',
            amp_scaler='not used: BF16/FP32 only', formal_convergence_campaign=False,primary_monitor='native-original full-frame metrics; ref512 is crop-local and not a matched full-frame comparison',
            device_type=device.type, torch_version=torch.__version__)
        step, log = 0, []
        if args.resume:
            step, log = load_checkpoint(args.resume, model, optimizer, ema, protocol, rank,
                                         inference=args.command == 'evaluate')
        if rank == 0:
            old_protocol = args.run_dir / 'protocol.json'
            if old_protocol.exists() and json.loads(old_protocol.read_text()) != protocol:
                raise ValueError('Existing run protocol is different; original preserved')
            write_json(old_protocol, protocol)
        if args.command == 'evaluate':
            if world != 1 or not args.resume:
                raise ValueError('evaluate requires single process and --resume')
            evaluated = [r for r in rows if r['split'] == 'dev']
            if not evaluated:
                raise ValueError('No DEV rows to evaluate')
            result = dict(checkpoint_sha256=digest(args.resume), evaluation_world_size=world,
                          raw=evaluate_rows(model, evaluated, args, device, args.arm, pool, args.run_dir / 'masks_raw'))
            with ema.apply(model):
                result['ema'] = evaluate_rows(model, evaluated, args, device, args.arm, pool, args.run_dir / 'masks_ema')
            write_json(args.run_dir / 'evaluation.json', result)
            return
        ddp = None; last_active = None
        began = time.monotonic()
        # Whole frozen encoder hash before and after smoke is an engineering invariant.
        frozen_before = {n: p.detach().clone() for n, p in model.named_parameters()
                         if not p.requires_grad and not (args.arm == 'U' and any(n.startswith(f'encoder.blocks.{i}.') for i in [20,21,22]))}
        gradient_receipts = []
        feature_receipt = None
        segment_stop = min(steps, args.stop_after if args.stop_after is not None else steps)
        if step >= segment_stop:
            raise ValueError('Segment must advance the restored checkpoint')
        while step < segment_stop:
            active = configure_stage(model, args.arm, step, phase_steps)
            if active != last_active:
                # Rebuild reducer whenever requires_grad changes; do not silently omit newly thawed parameters.
                forward = None
                ddp = None
                if world > 1:
                    dist.barrier()
                    ddp = DDP(model, device_ids=[device.index] if device.type == 'cuda' else None,
                              broadcast_buffers=False, find_unused_parameters=False)
                last_active = active
            forward = ddp if ddp is not None else model
            set_learning_rates(optimizer, step, steps,
                               2 if smoke else 200 if args.arm in ('F', 'U') else 100,
                               .1 if args.arm in ('F', 'U') else .5, phase_steps)
            indices = schedule.at(step+(18000 if args.arm in ('L','M') else 0))[rank::world]
            examples = list(pool.map(lambda i: fixture_sample(rows[i], args.side,args.crop_mode) if args.fixture
                                     else read_roi_sample(rows[i], args.root, args.side,args.crop_mode), indices))
            if smoke and feature_receipt is None:
                with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16,
                             enabled=device.type == 'cuda' and args.precision == 'bf16'):
                    image = examples[0][0][None].to(device)
                    normalized = (image - model.image_mean) / model.image_std
                    four = model.encoder.get_intermediate_layers(normalized, n=[20,21,22,23], norm=True, reshape=True)
                    direct = model.encoder(normalized)['x_norm_patchtokens']
                    direct = direct.transpose(1,2).reshape_as(four[-1])
                    torch.testing.assert_close(four[-1], direct, atol=1e-5, rtol=1e-5)
                    feature_receipt = dict(shapes=[list(f.shape) for f in four], last_forward_matches=True,
                                           actual_weights=not args.fixture)
            optimizer.zero_grad(set_to_none=True)
            chunks = len(examples) // args.micro_batch
            loss_total = torch.zeros((), device=device)
            for micro in range(chunks):
                ex = examples[micro * args.micro_batch:(micro + 1) * args.micro_batch]
                images = torch.stack([e[0] for e in ex]).to(device)
                labels = torch.stack([e[1] for e in ex]).to(device)
                sync = ddp.no_sync() if ddp is not None and micro < chunks - 1 else contextlib.nullcontext()
                with sync, torch.autocast(device.type, dtype=torch.bfloat16,
                                         enabled=device.type == 'cuda' and args.precision == 'bf16'):
                    prediction = forward(images, args.arm)
                    loss = hungarian_loss(prediction, labels) / chunks
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite semantic loss')
                loss.backward()
                loss_total += loss.detach()
            gradient = {n: float(p.grad.float().norm()) for n, p in model.named_parameters()
                        if p.requires_grad and p.grad is not None}
            if not gradient or not all(np.isfinite(v) for v in gradient.values()):
                raise FloatingPointError('Missing or non-finite gradients')
            if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
                raise RuntimeError('Frozen parameter received gradient')
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 10., error_if_nonfinite=True)
            optimizer.step(); ema.update(model); step += 1
            if world > 1:
                dist.all_reduce(loss_total); loss_total /= world
            record = dict(step=step, loss=float(loss_total), grad_norm=float(norm), blocks=active,
                          local_ids=[rows[i]['id'] for i in indices],
                          global_ids=[rows[i]['id'] for i in schedule.at(step - 1+18000)],
                          lrs={g['name']: g['lr'] for g in optimizer.param_groups})
            log.append(record)
            if smoke:
                gradient_receipts.append(dict(step=step, blocks=active,
                    head_nonzero=any(v > 0 for n, v in gradient.items() if n.startswith('head.')),
                    encoder_nonzero_by_block={str(i): any(v > 0 for n, v in gradient.items()
                        if n.startswith(f'encoder.blocks.{i}.')) for i in active},
                    norm_nonzero=any(v > 0 for n, v in gradient.items() if n.startswith('encoder.norm.'))))
                check = gradient_receipts[-1]
                if not check['head_nonzero'] or not all(check['encoder_nonzero_by_block'].values()):
                    raise RuntimeError('Active semantic path lacks nonzero gradients')
                if active and not check['norm_nonzero']:
                    raise RuntimeError('Active final normalization lacks gradients')
            if rank == 0:
                print('SEMSEG', step, round(record['loss'], 5), 'active', active, flush=True)
                write_json(args.run_dir / 'progress.json', record)
            if not smoke and step % args.eval_every == 0:
                if rank == 0:
                    probes = []
                    for domain, bottom in [('real', True), ('real', False), ('synthetic', True), ('synthetic', False)]:
                        probes.extend(sorted([r for r in train_rows if (r['domain'], r['bottom']) == (domain, bottom)],
                                             key=lambda r: r['id'])[:64])
                    with ema.apply(model):
                        monitor = evaluate_rows(model, probes, args, device, args.arm, pool)
                    write_json(args.run_dir / f'train_monitor_{step:06d}.json', monitor)
                if world > 1:
                    dist.barrier()
            if step % args.save_every == 0 or step == segment_stop:
                rank_rngs = gather_rng(world)
                if rank == 0:
                    save_checkpoint(args.run_dir / f'step_{step:06d}.pt', model, optimizer, ema, step,
                                    protocol, rank_rngs, log)
                if world > 1:
                    dist.barrier()
        frozen_unchanged = all(torch.equal(p, frozen_before[n]) for n, p in model.named_parameters() if n in frozen_before)
        if not frozen_unchanged:
            raise RuntimeError('Frozen encoder bytes changed')
        if rank == 0:
            topology = {n: p.numel() for n, p in model.named_parameters() if p.requires_grad}
            write_json(args.run_dir / 'parameter_topology.json', topology)
            result = dict(passed=True, registered_total_steps=steps, segment_endpoint=step, full_schedule_completed=step==steps, engineering_only=smoke, fixture=args.fixture, steps=step, world_size=world,
                          backend=dist.get_backend() if world > 1 else 'single', frozen_unchanged=frozen_unchanged,
                          gradient_checks=gradient_receipts, seconds=time.monotonic() - began,
                          feature_receipt=feature_receipt,
                          gpu_peak_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
                          numerical_training_only=not smoke, formal_convergence_claim=False)
            result['train_probe_raw'] = evaluate_rows(model, train_rows[:4], args, device, args.arm, pool,
                                                       args.run_dir / 'smoke_masks_raw' if smoke else None)
            with ema.apply(model):
                result['train_probe'] = evaluate_rows(model, train_rows[:4], args, device, args.arm, pool,
                                                       args.run_dir / 'smoke_masks' if smoke else None)
            write_json(args.run_dir / 'completion.json', result)
            print(json.dumps({k: v for k, v in result.items() if not k.startswith('train_probe')}), flush=True)
        if world > 1:
            dist.barrier()


def main():
    args = parser().parse_args()
    try:
        if args.command == 'prepare':
            prepare(args)
        else:
            run(args)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
