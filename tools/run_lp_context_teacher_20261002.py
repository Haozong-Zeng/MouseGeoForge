"""Bounded native LP MHCRNN pilot; no pseudo-label export or held-out data access."""
import argparse
import datetime as dt
import fcntl
import json
import os
import subprocess
import sys
import time
import traceback
from run_teacher_cycle_dlc_20261001 import ROOT, OUT, write_state

RUN = OUT / 'lp_context_seed17_kl'


def run(smoke=False, train_repeat_center=False):
    guard = (OUT / 'gpu_job.lock').open('a+')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = json.loads((OUT / 'plan.json').read_text())
    deadline = min(dt.datetime.fromisoformat(plan['deadline_utc']).timestamp(), time.time() + 3600)
    assert time.time() < deadline - 300 and not (OUT / 'STOP').exists()
    import numpy as np
    import pandas as pd
    import torch
    import imgaug.augmenters as iaa
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import Callback, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    from lightning_pose.data.datasets import HeatmapDataset
    from lightning_pose.models.heatmap_tracker_mhcrnn import HeatmapTrackerMHCRNN
    from lightning_pose.losses.factory import LossFactory
    from lightning_pose.callbacks import UnfreezeBackbone
    from lightning_pose.utils.io import get_context_img_paths
    from torch.utils.data import DataLoader, Subset, Dataset
    from torchvision.transforms import ToTensor, Normalize
    from PIL import Image
    from audit_teacher_dlc_predictions_20261001 import stats
    torch.set_num_threads(8)
    pl.seed_everything(17, workers=True)
    RUN.mkdir(exist_ok=True)
    m = json.loads((OUT / 'lp_context_dataset_v1/manifest.json').read_text())
    old = np.array([e['old_index'] for e in m['entries']])
    with np.load(OUT / 'dlc_baseline_review/predictions_source_pixels.npz') as z:
        gt = z['gt'][old].copy()
    A = np.array(m['affine_source_to_input'])
    xy = gt * np.diag(A)[:2] + A[:2, 2]
    assert len(old) == 1331 and len(m['train_indices']) == 1043 and len(m['validation_indices']) == 288
    assert m['ids_1based'] == plan['approved_body_ids_1based']
    for role, key, sessions in [('train', 'train_indices', plan['training_sessions']), ('validation', 'validation_indices', plan['development_validation_sessions'])]:
        assert all(m['entries'][i]['role'] == role and m['entries'][i]['session'] in sessions for i in m[key])
    # Native filename-based context loader must not silently repeat missing interior images.
    for e in m['entries']:
        center = ROOT / e['context_center']
        paths = get_context_img_paths(center)
        for path, frame in zip(paths, e['context_frames']):
            p = center.with_name(f'frame_{frame:06d}.png')
            assert p.exists()
            if not e['boundary_clamped']:
                assert str(p) == str(path) and os.path.exists(path)
    columns = pd.MultiIndex.from_product([['manualv3'], m['names'], ['x', 'y']], names=['scorer', 'bodyparts', 'coords'])
    csv = RUN / 'labels.csv'
    pd.DataFrame(xy.reshape(len(xy), -1), columns=columns, index=[e['context_center'] for e in m['entries']]).to_csv(csv)
    common = dict(root_directory=ROOT, csv_path=str(csv), image_resize_height=768, image_resize_width=704,
                  downsample_factor=2, uniform_heatmaps=False, imgaug_hflip=False, do_context=True)
    trainset = HeatmapDataset(**common, imgaug_transform=iaa.Sequential([
        iaa.Sometimes(.5, iaa.Affine(rotate=(-15, 15), scale=(.9, 1.1))), iaa.AdditiveGaussianNoise(scale=(0, 6))]))
    evalset = HeatmapDataset(**common, imgaug_transform=iaa.Sequential([]))
    class RepeatCenter(Dataset):
        # Read/augment the identical five-frame sample before replacement, preserving
        # the base augmentation RNG path. Only the model input is replaced.
        def __init__(self, base):
            self.base = base

        def __len__(self):
            return len(self.base)

        def __getitem__(self, index):
            item = dict(self.base[index])
            item['images'] = item['images'][2:3].expand(5, -1, -1, -1).clone()
            return item

    fit_trainset = RepeatCenter(trainset) if train_repeat_center else trainset
    fit_evalset = RepeatCenter(evalset) if train_repeat_center else evalset
    assert evalset.keypoint_names == m['names']
    assert np.allclose(evalset.keypoints.numpy(), xy, equal_nan=True, atol=.001)
    for i in range(0, len(old), 100):
        item = evalset[i]
        assert tuple(item['images'].shape) == (5, 3, 768, 704)
        expected = Normalize([.485, .456, .406], [.229, .224, .225])(ToTensor()(Image.open(ROOT / m['entries'][i]['derived']).convert('RGB')))
        assert torch.equal(item['images'][2], expected)
    sample = evalset[next(i for i in m['train_indices'] if np.isnan(xy[i]).any())]
    missing = torch.isnan(sample['keypoints'].reshape(7, 2)).any(-1)
    factory = LossFactory({'heatmap_kl': {'log_weight': 0.0}}, data_module=None)
    target = sample['heatmaps'].unsqueeze(0)
    pred = torch.rand_like(target, requires_grad=True)
    loss, _ = factory(heatmaps_targ=target, heatmaps_pred=pred)
    loss.backward()
    assert torch.all(pred.grad[0, missing] == 0)
    settings = dict(framework='lightning-pose 2.4.2', model='HeatmapTrackerMHCRNN', seed=17, backbone='resnet50',
        pretrained='ImageNet', epochs_ceiling=30, wall_minutes=60, evaluation_reserve_minutes=5, batch_size=4,
        frames_per_context=5, fps=52, input_wh=[704, 768], loss='heatmap_kl', learning_rate=.0001,
        milestones=[20, 27], gamma=.3, unfreeze_epoch=1, center_input_exact=True, missing_gradient_zero=True,
        context_missing_interior_forbidden=True, boundary_policy='native clamping/repeat center only at actual video endpoints',
        train_indices=m['train_indices'], validation_indices=m['validation_indices'], old_indices=old.tolist(),
        SEM01_excluded=True, locked_holdout_opened=False, pseudo_labels_used=False,
        training_context='repeated_center' if train_repeat_center else 'real_neighbors',
        checkpoint_selection_context='repeated_center' if train_repeat_center else 'real_neighbors',
        comparison_note='Compare trained-real-neighbor selected/mf with trained-repeated-center selected_repeat/mf_repeat for matched single-seed training ablation. Same-checkpoint input substitution alone is not training ablation.')
    (RUN / 'settings.json').write_text(json.dumps(settings, indent=2))
    model = HeatmapTrackerMHCRNN(num_keypoints=7, loss_factory=factory, backbone='resnet50', pretrained=True,
        downsample_factor=2, torch_seed=17, optimizer_params={'learning_rate': .0001},
        lr_scheduler_params={'milestones': [20, 27], 'gamma': .3})
    if smoke:
        model.cuda().train()
        batch = next(iter(DataLoader(Subset(fit_trainset, m['train_indices'][:4]), batch_size=4)))
        if train_repeat_center:
            assert all(torch.equal(batch['images'][:, j], batch['images'][:, 2]) for j in range(5))
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        torch.cuda.reset_peak_memory_stats()
        inputs = model.get_loss_inputs_labeled(batch)
        loss, _ = factory(**inputs)
        assert torch.isfinite(loss)
        loss.backward()
        result = dict(loss=float(loss.detach()), batch_shape=list(batch['images'].shape),
                      peak_gpu_memory_gb=torch.cuda.max_memory_allocated() / 1e9, repeated_center_verified=train_repeat_center, passed=True)
        (RUN / 'smoke.json').write_text(json.dumps(result, indent=2))
        print('CONTEXT_SMOKE', result, flush=True)
        return
    assert json.loads((RUN / 'smoke.json').read_text())['passed']
    assert not list((RUN / 'checkpoints').glob('*.ckpt')), 'Do not overwrite a training run'
    write_state(stage='lp_context_initializing', pid=os.getpid(), active_run=RUN.name, mode='train',
                project_config=str(RUN / 'settings.json'), epoch=0, global_step=0, last_epoch_record=None, checkpoint=None, result=None,
                job_log=str(OUT / f'{RUN.name}.log'), job_deadline_utc=dt.datetime.fromtimestamp(deadline, dt.timezone.utc).isoformat(),
                next_action='Compare native SF/MF/selected and repeated-center predictions on identical dev images; no pseudo-GT release.')
    class Guard(Callback):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if trainer.global_step % 25 == 0:
                write_state(stage='lp_context_training', epoch=trainer.current_epoch + 1, global_step=trainer.global_step)
            if time.time() >= deadline - 300 or (OUT / 'STOP').exists():
                trainer.save_checkpoint(RUN / 'deadline.ckpt')
                trainer.should_stop = True

        def on_validation_epoch_end(self, trainer, pl_module):
            if trainer.sanity_checking:
                return
            record = dict(epoch=trainer.current_epoch + 1, step=trainer.global_step,
                          metrics={k: float(v.detach().cpu()) for k, v in trainer.callback_metrics.items()})
            with (RUN / 'epochs.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
            write_state(stage='lp_context_training', last_epoch_record=record)
            print('CONTEXT_EPOCH', json.dumps(record), flush=True)
    checkpoint = ModelCheckpoint(dirpath=RUN / 'checkpoints', filename='epoch{epoch:03d}', monitor='val_supervised_rmse', mode='min', save_top_k=2, save_last=True)
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=30, precision='32-true',
        callbacks=[Guard(), UnfreezeBackbone(unfreeze_epoch=1), checkpoint], logger=CSVLogger(str(RUN), name='metrics'),
        enable_progress_bar=False, log_every_n_steps=25, num_sanity_val_steps=2,
        max_time={'seconds': int(deadline - time.time() - 300)}, check_val_every_n_epoch=1)
    trainer.fit(model,
        train_dataloaders=DataLoader(Subset(fit_trainset, m['train_indices']), batch_size=4, shuffle=True, num_workers=4, pin_memory=True),
        val_dataloaders=DataLoader(Subset(fit_evalset, m['validation_indices']), batch_size=4, num_workers=4, pin_memory=True))
    assert checkpoint.best_model_path
    write_state(stage='lp_context_evaluating', checkpoint=checkpoint.best_model_path)
    model.load_state_dict(torch.load(checkpoint.best_model_path, map_location='cpu', weights_only=False)['state_dict'])
    model.cuda().eval()
    predictions = {k: [] for k in ['sf', 'mf', 'selected', 'sf_repeat', 'mf_repeat', 'selected_repeat']}
    valid = np.array(m['validation_indices'])
    with torch.inference_mode():
        for batch in DataLoader(Subset(evalset, valid.tolist()), batch_size=4, num_workers=4):
            assert time.time() < deadline and not (OUT / 'STOP').exists(), 'Checkpoint safe; evaluation interrupted'
            images = batch['images'].cuda()
            for suffix, x in [('', images), ('_repeat', images[:, 2:3].expand(-1, 5, -1, -1, -1).contiguous())]:
                sf, mf = model(x)
                ps, cs = model.head.run_subpixelmaxima(sf)
                pm, cm = model.head.run_subpixelmaxima(mf)
                ps, pm = ps.reshape(-1, 7, 2), pm.reshape(-1, 7, 2)
                selected = torch.where((cm > cs).unsqueeze(-1), pm, ps)
                for name, p in [('sf', ps), ('mf', pm), ('selected', selected)]:
                    predictions[name + suffix].append(p.cpu().numpy())
    uv = {k: (np.concatenate(v) - A[:2, 2]) / np.diag(A)[:2] for k, v in predictions.items()}
    truth = gt[valid]
    summary = dict(checkpoint=checkpoint.best_model_path, units='original1080x1200 pixels', validation_images=len(valid),
                   comparison_note=settings['comparison_note'], models={})
    def report(p):
        err = np.linalg.norm(p - truth, axis=-1)
        return dict(all=stats(err.ravel()), paws=stats(err[:, 3:].ravel()),
                    severe_gt100px=int(np.sum(err > 100)), keypoints={str(k): stats(err[:, j]) for j, k in enumerate(m['ids_1based'])})
    for k, p in uv.items():
        summary['models'][k] = report(p)
    for name, file, key in [('static_seed17', 'lp_seed17_kl', 'uv'), ('static_ensemble_median', 'lp_ensemble_17_31_73', 'uv_median'), ('DLC70', 'dlc_extension_review', 'uv')]:
        with np.load(OUT / file / 'predictions_source_pixels.npz') as z:
            summary['models'][name] = report(z[key][old[valid]])
    np.savez_compressed(RUN / 'predictions_source_pixels.npz', **{'uv_' + k: p for k, p in uv.items()}, gt=truth,
                        old_indices=old[valid], manifest_indices=valid, ids=m['ids_1based'])
    (RUN / 'summary.json').write_text(json.dumps(summary, indent=2))
    write_state(stage='lp_context_evaluated', pid=None, active_run=None, mode='idle_between_heartbeats',
                job_deadline_utc=None, latest_result=str(RUN / 'summary.json'), next_action='Inspect context errors/learning curve before choosing further experiments.')
    print('CONTEXT_RESULT', json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--train-repeat-center', action='store_true')
    args = parser.parse_args()
    if args.train_repeat_center:
        RUN = OUT / 'lp_contextoff_seed17_kl'
    if args.launch:
        log = OUT / f'{RUN.name}.log'
        assert not log.exists()
        with log.open('w') as f:
            command = [sys.executable, '-u', __file__]
            if args.train_repeat_center:
                command.append('--train-repeat-center')
            process = subprocess.Popen(command, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        print('LAUNCHED', process.pid)
    else:
        try:
            run(args.smoke, args.train_repeat_center)
        except Exception as exc:
            if not args.smoke:
                write_state(stage='lp_context_failed', pid=None, error=repr(exc))
            traceback.print_exc()
            raise
