"""Read-only paired seed comparison; derived diagnostics, not pseudo-ground truth."""
import datetime as dt
import json
from pathlib import Path
import shutil
import fcntl
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from bottom_anchor_common_20261002 import ROOT, OUT, PRIOR, budget, write_state
from audit_extended_teachers_20261002 import metrics


def main():
    plan, _, check = budget(300)
    check()
    dest = OUT / 'quadruped_seed_replication_comparison_v1'
    assert not dest.exists(), 'Never overwrite an existing report'
    dm = json.loads((OUT / 'superanimal_quadruped_hrnet_v1/dataset_manifest.json').read_text())
    other = json.loads((OUT / 'superanimal_quadruped_hrnet_seed29_v1/dataset_manifest.json').read_text())
    assert {k:v for k,v in dm.items() if k!='project_config'} == {k:v for k,v in other.items() if k!='project_config'}
    entries = [dm['entries'][i] for i in dm['validation_indices']]
    assert len(entries) == 281 and all(e['session'] in plan['development_sessions'] for e in entries)
    scale = np.diag(np.array(dm['affine_source_to_input']))[:2]
    ids = dm['ids_1based']
    specs = {'seed17_best110': 'quadruped_epoch110_review_v1',
             'seed29_best85': 'quadruped_seed29_epoch085_review_v1',
             'seed29_final150': 'quadruped_seed29_epoch150_review_v1'}
    models = {}
    for name, folder in specs.items():
        with np.load(OUT / folder / 'predictions_source_pixels.npz') as z:
            if models:
                assert np.allclose(gt, z['gt'], equal_nan=True)
                assert np.array_equal(indices, z['context_manifest_indices'])
            gt = z['gt'].copy(); indices = z['context_manifest_indices'].copy()
            models[name] = z['uv'].copy()
    # No fitted weights, confidence gates, or threshold sweep. Diagnostic only.
    ensemble = (models['seed17_best110'] + models['seed29_best85']) / 2
    all_models = dict(models, equal_mean_diagnostic=ensemble)
    scores = {name: metrics(uv, gt, entries, ids, scale) for name, uv in all_models.items()}
    errors = {name: np.linalg.norm(uv - gt, axis=-1) for name, uv in all_models.items()}
    disagreement = np.linalg.norm(models['seed17_best110'] - models['seed29_best85'], axis=-1)
    observed = np.isfinite(gt).all(axis=-1)
    severe = []
    for name in models:
        for i, j in np.argwhere(errors[name] > 100):
            severe.append(dict(model=name, image_index=int(i), session=entries[i]['session'],
                frame=entries[i]['frame'], camera=entries[i]['camera'], id=ids[j],
                error_source=float(errors[name][i, j]), seed_disagreement_source=float(disagreement[i, j])))
    chosen = sorted({r['image_index'] for r in severe})
    for j in (1, 2):
        chosen.append(int(np.nanargmax(errors['seed17_best110'][:, j])))
    chosen = list(dict.fromkeys(chosen))
    dest.mkdir()
    font = ImageFont.truetype('/mnt/c/Windows/Fonts/arial.ttf', 22)
    images = []
    for i in chosen:
        check(); e = entries[i]
        canvas = Image.new('RGB', (3240, 1250), '#181818')
        for col, (name, uv) in enumerate(models.items()):
            im = Image.open(ROOT / e['source']).convert('RGB'); d = ImageDraw.Draw(im)
            for j, k in enumerate(ids):
                if not observed[i, j]: continue
                x, y = gt[i, j]; u, v = uv[i, j]
                d.ellipse((x-7,y-7,x+7,y+7), outline='#70ff70', width=3)
                d.text((x+8,y-23), str(k), font=font, fill='#70ff70', stroke_width=1, stroke_fill='black')
                d.line((x,y,u,v), fill='#ff77df', width=2)
                d.line((u-7,v-7,u+7,v+7), fill='#ff77df', width=3)
                d.line((u-7,v+7,u+7,v-7), fill='#ff77df', width=3)
                d.text((u+9,v+3), str(k), font=font, fill='#ff77df', stroke_width=1, stroke_fill='black')
            canvas.paste(im, (1080*col,50))
            ImageDraw.Draw(canvas).text((1080*col+8,10), f'{name} | {e["session"]} f{e["frame"]} cam{e["camera"]}', font=font, fill='white')
        filename = f'{e["session"]}_{e["frame"]}_cam{e["camera"]}.jpg'
        canvas.save(dest / filename, quality=95); images.append(filename)
    logs = [json.loads(line) for line in (OUT/'superanimal_quadruped_hrnet_seed29_v1/epochs.jsonl').read_text().splitlines()]
    late = [r for r in logs if r['mode']=='eval' and r['epoch']>=100]
    assert late[-1]['epoch']==150 and np.isclose(late[-1]['lr'][0],9e-6)
    with (ROOT/plan['gpu_lock']).open('a+') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); held=False
        except BlockingIOError: held=True
    report = dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(), canonical_metrics=scores,
        severe_errors=severe, gallery=images, validation_images=281, no_confidence_filter=True,
        seed_disagreement_definition='Always between seed17_best110 and seed29_best85, including rows whose model is final150; not disagreement with final150.',
        disagreement_observed=dict(mean=float(np.mean(disagreement[observed])), p90=float(np.percentile(disagreement[observed],90))),
        late_validation=late, shared_GPU_lock_held=held, prior_STOP_preserved=(PRIOR/'STOP').exists(),
        decision='Both Quadruped seeds completed150 and both LR decays. No sustained late improvement justifies extending to200/300. Retain seed17 best110 as incumbent and preserve seed29 best85/final150. Ensemble is a diagnostic, not a released teacher or pseudo-GT.',
        next='Inspect all severe-error and dorsal overlays, then audit whether seed agreement misses shared errors before using teacher uncertainty for annotation triage. No geometry correction or label replacement in this comparison.',
        caveats=['Two seeds on reused development videos are not independent-dataset validation.',
                 'mAP-selected best checkpoints need not minimize Euclidean mean or severe-error count.',
                 'Equal-mean ensemble has no fitted weights but is exploratory on the same development set.',
                 'Only observed manual points evaluated; missing labels are unknown.'],
        locked_holdouts_accessed=False, source_labels_changed=False, pseudo_gt_released=False)
    (dest/'summary.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(dest/'paired_predictions.npz',gt=gt,context_manifest_indices=indices,ids=ids,**all_models)
    page='<meta charset="utf-8"><h1>Quadruped双随机种子复验</h1><p>左seed17最佳110；中seed29最佳85；右seed29末轮150。绿圈人工、粉叉预测。全部超过100原图像素错误所在图片及6/7号点困难样本。开发集诊断，不修改标签。</p>'
    page+=''.join(f'<h2>{p}</h2><img style="max-width:100%" src="{p}">' for p in images)
    page+='<pre>'+json.dumps(report,indent=2)+'</pre>'
    (dest/'index.html').write_text(page)
    export=Path('/mnt/c/Users/55395/Downloads/MouseGeoForge-BottomAnchor-20261002')/dest.name
    shutil.copytree(dest,export)
    write_state(latest_decision=str(dest/'summary.json'),latest_stage_comparison=str(dest/'summary.json'),
                latest_training_visual_directory=str(export),convergence_decision=report['decision'],next_action=report['next'])
    print(json.dumps(dict(export=str(export),metrics={k:v['all'] for k,v in scores.items()},severe=severe,late_validation=late),indent=2))


if __name__=='__main__': main()
