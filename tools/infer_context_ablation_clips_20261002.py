"""Bounded ON/OFF inference on four frozen PNG-exact development clips."""
import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from run_teacher_cycle_dlc_20261001 import ROOT, OUT, write_state

DEST = OUT / 'context_ablation_clips_v1'


def main():
    lock = (OUT / 'gpu_job.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = json.loads((OUT / 'plan.json').read_text())
    deadline = min(time.time() + 1500, dt.datetime.fromisoformat(plan['deadline_utc']).timestamp())
    def check():
        assert time.time() < deadline and not (OUT / 'STOP').exists()
    check()
    import cv2
    import numpy as np
    import torch
    from torchvision import transforms
    from lightning_pose.models.heatmap_tracker_mhcrnn import HeatmapTrackerMHCRNN
    torch.set_num_threads(8); cv2.setNumThreads(1)
    assert torch.cuda.is_available()
    DEST.mkdir(exist_ok=False)
    m = json.loads((OUT / 'lp_context_dataset_v1/manifest.json').read_text())
    A = np.array(m['affine_source_to_input'])
    clips = json.loads((OUT / 'png_exact_clips_v1/manifest.json').read_text())
    excluded = {(r['session'], r['frame']) for r in json.loads((ROOT / plan['exclusion_manifest']).read_text())['cases']}
    excluded.add(('AZR2L2', 584))
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    models = {}; saved = {}; checkpoints = {}
    for mode, folder, key in [('on', 'lp_context_seed17_kl', 'uv_selected'), ('off', 'lp_contextoff_seed17_kl', 'uv_selected_repeat')]:
        check()
        checkpoint = json.loads((OUT / folder / 'summary.json').read_text())['checkpoint']; checkpoints[mode] = checkpoint
        models[mode] = HeatmapTrackerMHCRNN.load_from_checkpoint(checkpoint, map_location='cpu', pretrained=False, loss_factory=None, weights_only=False).cuda().eval()
        with np.load(OUT / folder / 'predictions_source_pixels.npz') as z:
            saved[mode] = {(m['entries'][i]['session'], m['entries'][i]['camera'], m['entries'][i]['frame']): z[key][pos].copy() for pos, i in enumerate(z['manifest_indices'])}
    settings = dict(checkpoints=checkpoints, fps=52, input_wh=[704, 768], ids=m['ids_1based'],
        context='ON real +/-2frames; OFF exact repeatcenter. Trim two frames at BOTH clip ends, no artificial clip-boundary padding.',
        input_provenance='Existing png_exact_clips_v1 FFV1 data, already verified exact to original manual PNG. NO new rectification.',
        coordinate_units='original1080x1200pixels', confidence='native max(SF,MF) confidence; NOT calibrated probability',
        original_labels_modified=False, locked_holdout_opened=False, pseudo_gt_released=False, train_allowed=False)
    (DEST / 'settings.json').write_text(json.dumps(settings, indent=2))
    write_state(stage='context_ablation_clip_inference', pid=os.getpid(), active_run=DEST.name, mode='GPU_inference',
        job_log=str(OUT / 'context_ablation_clips_v1.log'), job_deadline_utc=dt.datetime.fromtimestamp(deadline, dt.timezone.utc).isoformat())
    records = []
    for clip in clips:
        check(); s = clip['session']; assert s in plan['development_validation_sessions']
        full_frames = np.arange(clip['source_start'], clip['source_end'] + 1)
        assert len(full_frames) == 208 and not any((s, int(f)) in excluded for f in full_frames)
        frames = full_frames[2:-2]
        outputs = {mode: {'uv': [], 'confidence': []} for mode in models}
        parities = []
        for c in clip['cameras']:
            check()
            files = list(Path(clip['video_folder']).glob(f'{s}_cam{c}_*.avi')); assert len(files) == 1
            cap = cv2.VideoCapture(str(files[0])); assert cap.isOpened() and cap.get(cv2.CAP_PROP_FPS) == 52
            images = []
            for _ in full_frames:
                ok, im = cap.read(); assert ok and im.shape == (768, 704, 3)
                images.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            cap.release()
            arrays = {mode: {'uv': [], 'confidence': []} for mode in models}
            with torch.inference_mode():
                for first in range(2, len(images) - 2, 4):
                    check()
                    centers = range(first, min(first + 4, len(images) - 2))
                    x = torch.stack([torch.stack([transform(images[j]) for j in range(i-2, i+3)]) for i in centers]).cuda()
                    for mode, model in models.items():
                        inp = x if mode == 'on' else x[:, 2:3].expand(-1, 5, -1, -1, -1).contiguous()
                        hs, hm = model(inp)
                        ps, cs = model.head.run_subpixelmaxima(hs); pm, cm = model.head.run_subpixelmaxima(hm)
                        ps = ps.reshape(-1, 7, 2); pm = pm.reshape(-1, 7, 2)
                        p = torch.where((cm > cs).unsqueeze(-1), pm, ps)
                        conf = torch.maximum(cs, cm)
                        arrays[mode]['uv'].append((p.cpu().numpy() - A[:2, 2]) / np.diag(A)[:2])
                        arrays[mode]['confidence'].append(conf.cpu().numpy())
            for mode in models:
                uv = np.concatenate(arrays[mode]['uv']); conf = np.concatenate(arrays[mode]['confidence'])
                assert uv.shape == (204, 7, 2) and np.isfinite(uv).all()
                outputs[mode]['uv'].append(uv); outputs[mode]['confidence'].append(conf)
                for i, frame in enumerate(frames):
                    key = (s, c, int(frame))
                    if key in saved[mode]:
                        error = float(np.max(np.abs(uv[i] - saved[mode][key])))
                        assert error < .1, (mode, key, error)
                        parities.append(dict(mode=mode, camera=c, frame=int(frame), max_abs_source_px=error))
            print('CONTEXT_CLIP_CAMERA', clip['cohort'], s, c, flush=True)
        folder = DEST / clip['cohort']; folder.mkdir(exist_ok=True)
        path = folder / f'{s}.npz'
        np.savez_compressed(path, **{f'{mode}_{key}': np.asarray(value) for mode, values in outputs.items() for key, value in values.items()},
            cameras=clip['cameras'], source_frames=frames, ids=m['ids_1based'], fps=52, train_allowed=False)
        record = dict(clip, file=str(path), retained_start=int(frames[0]), retained_end=int(frames[-1]), retained_frames=len(frames), sparse_png_parities=parities)
        records.append(record); (DEST / 'progress.json').write_text(json.dumps(records, indent=2))
        write_state(stage='context_ablation_clip_inference', completed_clips=len(records))
    (DEST / 'manifest.json').write_text(json.dumps(records, indent=2))
    write_state(stage='context_ablation_clips_ready', pid=None, active_run=None, mode='idle_between_heartbeats', job_deadline_utc=None,
        latest_result=str(DEST / 'manifest.json'), next_action='Inspect ON/OFF continuous dynamics and sparse manual accuracy, then fixed-parameter truly held-view-out geometry. No dense temporal GT available; do not equate smoothing with accuracy.')
    print('CONTEXT_CLIPS_READY', len(records), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--launch', action='store_true'); args = parser.parse_args()
    if args.launch:
        log = OUT / 'context_ablation_clips_v1.log'; assert not log.exists()
        with log.open('w') as f:
            proc = subprocess.Popen([sys.executable, '-u', __file__], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        print('CONTEXT_CLIP_INFERENCE_LAUNCHED', proc.pid)
    else:
        try:
            main()
        except Exception as exc:
            write_state(stage='context_ablation_clip_failed', pid=None, error=repr(exc))
            traceback.print_exc(); raise
