"""CPU-only matched-training pilot audit. No new training, labels or thresholds."""
import argparse
import datetime as dt
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from run_teacher_cycle_dlc_20261001 import ROOT, OUT, write_state
from audit_teacher_dlc_predictions_20261001 import stats

ON = OUT / 'lp_context_seed17_kl'
OFF = OUT / 'lp_contextoff_seed17_kl'
DEST = OUT / 'lp_context_matched_ablation_v1'


def main(wait=False):
    deadline = min(time.time() + 2700, dt.datetime.fromisoformat(json.loads((OUT / 'plan.json').read_text())['deadline_utc']).timestamp())
    while not (OFF / 'summary.json').exists():
        assert time.time() < deadline and not (OUT / 'STOP').exists()
        assert wait, 'Control not ready'
        state = json.loads((OUT / 'state.json').read_text())
        assert state.get('stage') != 'lp_context_failed', state
        time.sleep(20)
    assert time.time() < deadline
    DEST.mkdir(exist_ok=False)
    settings = [json.loads((p / 'settings.json').read_text()) for p in [ON, OFF]]
    matching = ['seed', 'backbone', 'pretrained', 'epochs_ceiling', 'batch_size', 'input_wh', 'loss', 'learning_rate', 'milestones', 'gamma', 'unfreeze_epoch', 'train_indices', 'validation_indices', 'old_indices']
    for key in matching:
        assert settings[0][key] == settings[1][key], key
    logs = [[json.loads(line) for line in (p / 'epochs.jsonl').read_text().splitlines()] for p in [ON, OFF]]
    assert all(len(log) == 30 and log[-1]['step'] == 7830 for log in logs), 'Incomplete training budgets: do not claim matched comparison'
    predictions = {}
    with np.load(ON / 'predictions_source_pixels.npz') as z:
        gt = z['gt'].copy(); old = z['old_indices'].copy(); idx = z['manifest_indices'].copy()
        predictions.update(on_selected=z['uv_selected'].copy(), on_mf=z['uv_mf'].copy(), on_sf=z['uv_sf'].copy())
    with np.load(OFF / 'predictions_source_pixels.npz') as z:
        assert np.array_equal(z['old_indices'], old) and np.allclose(z['gt'], gt, equal_nan=True)
        predictions.update(off_selected=z['uv_selected_repeat'].copy(), off_mf=z['uv_mf_repeat'].copy(), off_sf=z['uv_sf_repeat'].copy())
    with np.load(OUT / 'lp_ensemble_17_31_73/predictions_source_pixels.npz') as z:
        predictions['static_median'] = z['uv_median'][old]
    m = json.loads((OUT / 'lp_context_dataset_v1/manifest.json').read_text())
    entries = [m['entries'][i] for i in idx]
    ids = m['ids_1based']
    errors = {name: np.linalg.norm(p - gt, axis=-1) for name, p in predictions.items()}
    models = {}
    for name, err in errors.items():
        models[name] = dict(all=stats(err.ravel()), paws=stats(err[:, 3:].ravel()),
            severe_gt100px=int(np.sum(err > 100)), severe_paw_gt100px=int(np.sum(err[:, 3:] > 100)),
            keypoints={str(k): stats(err[:, j]) for j, k in enumerate(ids)},
            cameras={c: stats(err[[e['camera'] == c for e in entries]].ravel()) for c in sorted({e['camera'] for e in entries})},
            sessions={s: stats(err[[e['session'] == s for e in entries]].ravel()) for s in ['AZR2L2', 'FR2']})
    comparisons = {}
    groups = sorted({(e['session'], e['frame']) for e in entries})
    group_ids = np.array([groups.index((e['session'], e['frame'])) for e in entries])
    rng = np.random.default_rng(1702)
    for a, b in [('on_selected', 'off_selected'), ('on_mf', 'off_mf'), ('on_sf', 'off_sf'), ('on_selected', 'static_median')]:
        delta = errors[b] - errors[a]
        sums = np.array([np.nansum(delta[group_ids == i]) for i in range(len(groups))])
        counts = np.array([np.isfinite(delta[group_ids == i]).sum() for i in range(len(groups))])
        # Conditional descriptive resampling: keep all cameras of a frame together.
        samples = rng.integers(0, len(groups), size=(3000, len(groups)))
        boot = sums[samples].sum(1) / counts[samples].sum(1)
        comparisons[a + '_vs_' + b] = dict(positive_means_on_better=True, error_reduction_px=stats(delta.ravel()),
            improved_gt5px=int(np.sum(delta > 5)), worsened_gt5px=int(np.sum(delta < -5)),
            descriptive_frame_cluster_bootstrap_95_interval=np.percentile(boot, [2.5, 97.5]).tolist(),
            frame_groups=len(groups), caveat='Conditional on two reused videos, not an independent animal/session generalization interval.')
    summary = dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(), matched_fields=matching,
        epochs=[30, 30], optimizer_steps=[7830, 7830], seed=17, models=models, paired=comparisons,
        evaluation='Each model uses intended inputs: ON realneighbors, OFF repeatedcenter. Both select checkpoint by identical native validation metric.',
        limits=['Single seed; only two reused development sessions. No animal-level generalization claim.',
            'Same model capacity/optimizer steps/input dimensions; temporal-off is repeatedcenter MHCRNN, not the old standalone static network.',
            'Validation-best checkpoint selection is matched but uses this development set.',
            'No true3D evaluation; all candidates remain diagnostic, not production groundtruth.'],
        locked_holdout_opened=False, original_labels_modified=False, pseudo_gt_released=False)
    (DEST / 'summary.json').write_text(json.dumps(summary, indent=2))
    np.savez_compressed(DEST / 'predictions_source_pixels.npz', **predictions, gt=gt, old_indices=old, ids=ids)
    font = ImageFont.truetype('/mnt/c/Windows/Fonts/arial.ttf', 18)
    small = ImageFont.truetype('/mnt/c/Windows/Fonts/arial.ttf', 14)
    def pawmean(err):
        n = np.isfinite(err[:, 3:]).sum(1)
        return np.divide(np.nansum(err[:, 3:], axis=1), n, out=np.full(len(n), np.nan), where=n > 0)
    difference = pawmean(errors['off_selected']) - pawmean(errors['on_selected'])
    cases = []
    for category, values in [('on_gain', difference), ('on_harm', -difference), ('on_failure', pawmean(errors['on_selected']))]:
        for rank, i in enumerate([i for i in np.argsort(np.nan_to_num(values, nan=-np.inf))[::-1] if np.isfinite(values[i])][:6], 1):
            assert time.time() < deadline and not (OUT / 'STOP').exists()
            e = entries[i]
            assert e['session'] in ['AZR2L2', 'FR2']
            original = Image.open(ROOT / e['source']).convert('RGB').resize((540, 600), Image.Resampling.LANCZOS)
            canvas = Image.new('RGB', (1620, 690), '#202020'); head = ImageDraw.Draw(canvas)
            head.text((8, 8), f"{category} #{rank}: {e['session']} frame{e['frame']} cam{e['camera']} | paw difference/rank score {values[i]:.1f}px", font=font, fill='white')
            head.text((8, 32), 'Green square: manual; magenta cross: prediction. Best checkpoints, matched30epochs/7830steps/seed17.', font=font, fill='white')
            for column, name in enumerate(['off_selected', 'on_selected', 'static_median']):
                im = original.copy(); draw = ImageDraw.Draw(im)
                for j, k in enumerate(ids):
                    if not np.isfinite(gt[i, j]).all():
                        continue
                    x, y = gt[i, j] * .5; u, v = predictions[name][i, j] * .5
                    draw.line((x, y, u, v), fill='#ef73dc', width=1)
                    draw.rectangle((x-4, y-4, x+4, y+4), outline='#64ff6b', width=2)
                    draw.text((x+5, y-16), str(k), fill='#64ff6b', font=small, stroke_width=1, stroke_fill='black')
                    draw.line((u-4, v-4, u+4, v+4), fill='#ff75d5', width=2)
                    draw.line((u-4, v+4, u+4, v-4), fill='#ff75d5', width=2)
                    draw.text((u+5, v+3), str(k), fill='#ff75d5', font=small, stroke_width=1, stroke_fill='black')
                canvas.paste(im, (column*540, 90))
                head.text((column*540+5, 62), f'{name} | pawmean {pawmean(errors[name])[i]:.1f}px', fill='white', font=font)
            filename = f"{category}_{rank:02d}_{e['session']}_{e['frame']:06d}_cam{e['camera']}.jpg"
            canvas.save(DEST / filename, quality=94)
            cases.append(dict(file=filename, category=category, session=e['session'], frame=e['frame'], camera=e['camera'], score=float(values[i])))
    (DEST / 'visual_manifest.json').write_text(json.dumps(cases, indent=2))
    html = '<!doctype html><meta charset="utf-8"><title>Matched context ON/OFF</title><h1>Matched context ON/OFF</h1><p>Single-seed development pilot. Diagnostic predictions, not ground truth.</p>'
    html += '<table border="1"><tr><th>Model</th><th>Mean px</th><th>P90 px</th><th>Paw mean px</th><th>Paw >100px</th></tr>'
    for name, v in models.items():
        html += f'<tr><td>{name}</td><td>{v["all"]["mean"]:.2f}</td><td>{v["all"]["p90"]:.2f}</td><td>{v["paws"]["mean"]:.2f}</td><td>{v["severe_paw_gt100px"]}</td></tr>'
    html += '</table>' + ''.join(f'<h2>{e["file"]}</h2><img style="max-width:100%" src="{e["file"]}">' for e in cases)
    (DEST / 'index.html').write_text(html)
    export = Path('/mnt/c/Users/55395/Downloads/MouseGeoForge-TeacherTemporal-20261001/lp_context_matched_ablation_v1')
    assert not export.exists()
    shutil.copytree(DEST, export)
    write_state(latest_result=str(DEST / 'summary.json'), cpu_matched_review_pid=None,
        next_action='Visually inspect matched ON/OFF report and judge temporal benefit; no new training automatically launched by CPU reviewer.')
    print('MATCHED_ABLATION_READY', export, flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--wait', action='store_true'); parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    if args.launch:
        log = OUT / 'lp_context_matched_ablation.log'; assert not log.exists()
        with log.open('w') as f:
            proc = subprocess.Popen([sys.executable, '-u', __file__, '--wait'], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        write_state(cpu_matched_review_pid=proc.pid, cpu_matched_review_log=str(log))
        print('MATCHED_REVIEW_LAUNCHED', proc.pid)
    else:
        main(args.wait)
