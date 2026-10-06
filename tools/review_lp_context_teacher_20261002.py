"""CPU-only, bounded follow-up: paired diagnostics on the fixed development cohort."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from run_teacher_cycle_dlc_20261001 import ROOT, OUT
from audit_teacher_dlc_predictions_20261001 import stats

RUN = OUT / 'lp_context_seed17_kl'
DEST = RUN / 'review'


def main(wait=False):
    deadline = min(time.time() + 3600, dt.datetime.fromisoformat(json.loads((OUT / 'plan.json').read_text())['deadline_utc']).timestamp())
    def guard():
        assert time.time() < deadline and not (OUT / 'STOP').exists(), 'Review deadline/STOP reached'
    while not (RUN / 'summary.json').exists():
        guard()
        if not wait:
            raise RuntimeError('Training/evaluation results not yet ready')
        state = json.loads((OUT / 'state.json').read_text())
        assert state.get('stage') != 'lp_context_failed', state
        time.sleep(30)
    guard()
    DEST.mkdir(exist_ok=False)
    m = json.loads((OUT / 'lp_context_dataset_v1/manifest.json').read_text())
    with np.load(RUN / 'predictions_source_pixels.npz') as z:
        predictions = {k[3:]: z[k].copy() for k in z.files if k.startswith('uv_')}
        gt, indices, old = z['gt'].copy(), z['manifest_indices'].copy(), z['old_indices'].copy()
    entries = [m['entries'][i] for i in indices]
    assert len(entries) == len(gt) == 288
    for name, folder, key in [('static_seed17', 'lp_seed17_kl', 'uv'), ('static_median', 'lp_ensemble_17_31_73', 'uv_median'), ('DLC70', 'dlc_extension_review', 'uv')]:
        with np.load(OUT / folder / 'predictions_source_pixels.npz') as z:
            predictions[name] = z[key][old].copy()
    errors = {k: np.linalg.norm(v - gt, axis=-1) for k, v in predictions.items()}
    ids = m['ids_1based']
    models = {}
    for name, err in errors.items():
        confusions = []
        for i, e in enumerate(entries):
            for j in range(3, 7):
                if not np.isfinite(gt[i, j]).all():
                    continue
                distances = np.linalg.norm(gt[i, 3:] - predictions[name][i, j], axis=-1)
                nearest = int(np.nanargmin(distances)) + 3
                if nearest != j and err[i, j] > 20 and distances[nearest - 3] + 10 < err[i, j]:
                    confusions.append(dict(session=e['session'], frame=e['frame'], camera=e['camera'],
                        id=ids[j], nearest_labeled_paw_id=ids[nearest], target_error_px=float(err[i, j]),
                        other_distance_px=float(distances[nearest - 3])))
        models[name] = dict(all=stats(err.ravel()), paws=stats(err[:, 3:].ravel()),
            paw_gt100px=int(np.sum(err[:, 3:] > 100)), nearest_other_paw_candidates=len(confusions),
            sessions={s: stats(err[[e['session'] == s for e in entries]].ravel()) for s in sorted({e['session'] for e in entries})},
            cameras={c: stats(err[[e['camera'] == c for e in entries]].ravel()) for c in sorted({e['camera'] for e in entries})},
            confusion_candidates=confusions)
    comparisons = {}
    for a, b in [('mf', 'mf_repeat'), ('selected', 'selected_repeat'), ('selected', 'static_seed17'), ('selected', 'static_median'), ('mf', 'sf')]:
        diff = errors[b] - errors[a]
        finite = np.isfinite(diff)
        comparisons[a + '_vs_' + b] = dict(positive_means_improvement=True,
            error_reduction_px=stats(diff[finite]), improved_gt5px=int(np.sum(diff > 5)), worsened_gt5px=int(np.sum(diff < -5)),
            common_finite_points=int(finite.sum()),
            prediction_displacement_px=stats(np.linalg.norm(predictions[a] - predictions[b], axis=-1)[finite]))
    summary = dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(), models=models, paired=comparisons,
        caveats=['Two reused development sessions; no fresh independent test.',
            'Repeated-center input is a same-checkpoint inference control, not independently trained temporal-off ablation.',
            'Nearest-other-paw rule produces review candidates, not proof of identity error; manually close paws can be ambiguous.',
            'Missing manual annotations are unknown. Only annotated positions are scored.',
            'No true3D labels or pseudo-GT released.'], locked_holdout_opened=False)
    (DEST / 'summary.json').write_text(json.dumps(summary, indent=2))
    # Select per-image cases by paired paw error, explicitly including both directions.
    def image_means(err):
        count = np.isfinite(err[:, 3:]).sum(-1)
        return np.divide(np.nansum(err[:, 3:], axis=-1), count, out=np.full(len(err), np.nan), where=count > 0)
    gain = image_means(errors['mf_repeat']) - image_means(errors['mf'])
    failure = image_means(errors['selected'])
    choices = []
    for category, scores in [('context_gain', gain), ('context_harm', -gain), ('selected_failure', failure)]:
        selected = [i for i in np.argsort(np.nan_to_num(scores, nan=-np.inf))[::-1] if np.isfinite(scores[i])][:8]
        choices.extend((category, rank + 1, int(i), float(scores[i])) for rank, i in enumerate(selected))
    font = ImageFont.truetype('/mnt/c/Windows/Fonts/arial.ttf', 18)
    small = ImageFont.truetype('/mnt/c/Windows/Fonts/arial.ttf', 14)
    files = []
    for category, rank, i, score in choices:
        guard()
        e = entries[i]
        assert e['session'] in ['AZR2L2', 'FR2'] and not (e['session'] == 'AZR2L2' and e['frame'] == 584)
        original = Image.open(ROOT / e['source']).convert('RGB')
        names = ['mf_repeat', 'mf', 'sf', 'selected', 'static_seed17', 'static_median']
        sheet = Image.new('RGB', (540 * 3, 650 * 2 + 60), '#202020')
        header = ImageDraw.Draw(sheet)
        header.text((10, 8), f"{category} #{rank}: {e['session']} frame{e['frame']} cam{e['camera']} | score {score:.1f}px", font=font, fill='white')
        header.text((10, 34), 'Green square = manual; magenta cross = prediction; line = error. Coordinates original pixels.', font=font, fill='white')
        for col, name in enumerate(names):
            panel = original.resize((540, 600), Image.Resampling.LANCZOS)
            draw = ImageDraw.Draw(panel)
            for j, kp in enumerate(ids):
                if not np.isfinite(gt[i, j]).all():
                    continue
                x, y = gt[i, j] * .5
                u, v = predictions[name][i, j] * .5
                draw.line((x, y, u, v), fill='#ef73dc', width=1)
                draw.rectangle((x - 4, y - 4, x + 4, y + 4), outline='#64ff6b', width=2)
                draw.text((x + 5, y - 16), str(kp), font=small, fill='#64ff6b', stroke_width=1, stroke_fill='black')
                draw.line((u - 4, v - 4, u + 4, v + 4), fill='#ff75d5', width=2)
                draw.line((u - 4, v + 4, u + 4, v - 4), fill='#ff75d5', width=2)
                draw.text((u + 5, v + 3), str(kp), font=small, fill='#ff75d5', stroke_width=1, stroke_fill='black')
            cx, cy = (col % 3) * 540, (col // 3) * 650 + 60
            sheet.paste(panel, (cx, cy + 45))
            header.text((cx + 5, cy + 3), name, font=font, fill='white')
            header.text((cx + 5, cy + 24), f"paw mean {image_means(errors[name])[i]:.1f}px", font=small, fill='white')
        filename = f"{category}_{rank:02d}_{e['session']}_{e['frame']:06d}_cam{e['camera']}.jpg"
        sheet.save(DEST / filename, quality=94)
        files.append(dict(file=filename, category=category, rank=rank, session=e['session'], frame=e['frame'], camera=e['camera'], score=score))
    (DEST / 'visual_manifest.json').write_text(json.dumps(files, indent=2))
    # Simple local HTML, no service or network dependencies.
    html = '<!doctype html><meta charset="utf-8"><title>LP context diagnostic review</title><h1>LP context diagnostic review</h1><p>24 ranked cases: gains, harms, failures. Diagnostic only; not ground truth.</p>'
    html += ''.join(f'<h2>{e["file"]}</h2><img style="max-width:100%" src="{e["file"]}">' for e in files)
    (DEST / 'index.html').write_text(html)
    # Learning curve reports LP's own metric, not the source-pixel EPE above.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    epochs = [json.loads(s) for s in (RUN / 'epochs.jsonl').read_text().splitlines()]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for axis, metric in zip(ax, ['supervised_loss', 'supervised_rmse']):
        for role in ['train', 'val']:
            axis.plot([e['epoch'] for e in epochs], [e['metrics'][role + '_' + metric] for e in epochs], label=role)
        axis.set_title('Native LP ' + metric)
        axis.set_xlabel('Epoch'); axis.legend(); axis.grid(alpha=.2)
    fig.suptitle('Native LP logged metrics; train may be step-wise, not epoch mean')
    fig.tight_layout(); fig.savefig(DEST / 'learning_curve.png', dpi=150); plt.close(fig)
    export = Path('/mnt/c/Users/55395/Downloads/MouseGeoForge-TeacherTemporal-20261001/lp_context_seed17_kl')
    assert not export.exists(), 'Do not overwrite an existing user review'
    shutil.copytree(DEST, export)
    print('CONTEXT_REVIEW_READY', export, flush=True)
    print(json.dumps({k: {n: v for n, v in val.items() if n not in ['confusion_candidates', 'sessions', 'cameras']} for k, val in models.items()}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait', action='store_true')
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    if args.launch:
        log = OUT / 'lp_context_review.log'
        assert not log.exists()
        with log.open('w') as f:
            process = subprocess.Popen([sys.executable, '-u', __file__, '--wait'], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        print('REVIEW_WAITER_LAUNCHED', process.pid)
    else:
        main(args.wait)
