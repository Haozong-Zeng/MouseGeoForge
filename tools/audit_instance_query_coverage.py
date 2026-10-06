"""Read-only cached all-query coverage audit. GT oracles are diagnostics only."""
import json
from collections import defaultdict
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from run_depth_pose_ablation import ROOT, torch, write_json, fingerprint
from training.instance_pose_losses import hungarian_match_instances, _linear_sum_assignment
from mousegeoforge.pilot_metrics import box_iou

OUT = ROOT/'output/instance_query_coverage_v1'
BOX = ROOT/'output/mask_box_decode_v1'
MASK = ROOT/'output/mask_box_failures_v1_actual_train'


def maximum_coverage(overlap, scores=None):
    """GT-assisted upper bound on simultaneous matches at fixed .5 IoU/score."""
    eligible = overlap >= .5
    if scores is not None:
        eligible = eligible & (scores[None] >= .5)
    qs, ms = _linear_sum_assignment(-eligible.float())
    return int(eligible[ms, qs].sum())


def mapping(qs, ms):
    return {int(m): int(q) for q, m in zip(qs, ms)}


def main():
    started = time.monotonic()
    if OUT.exists():
        raise RuntimeError('Preserve existing audit; do not overwrite')
    OUT.mkdir(parents=True)
    write_json(OUT/'status.json', dict(state='auditing_cpu_cache', model_forwards=0, optimizer_steps=0))
    torch.set_num_threads(2)
    assert maximum_coverage(torch.tensor([[.9, .8], [.2, .1]])) == 1
    assert maximum_coverage(torch.tensor([[.9, .8], [.8, .1]])) == 2
    assert maximum_coverage(torch.tensor([[.9, .1], [.1, .9]]), torch.tensor([1., 0.])) == 1
    inputs = {}

    def read_json(path):
        inputs[str(path)] = fingerprint(path)
        return json.loads(path.read_text())

    assert read_json(BOX/'verification.json')['passed']
    assert read_json(MASK/'verification.json')['passed']
    assert read_json(MASK/'selection_verification.json')['passed']
    stats = read_json(MASK/'query_statistics.json')
    spec = read_json(BOX/'protocol.json')
    code_paths = ('tools/audit_instance_query_coverage.py', 'mousegeoforge/pilot_metrics.py',
                  'vggt-omega/training/instance_pose_losses.py')
    code = {p:fingerprint(ROOT/p) for p in code_paths}
    write_json(OUT/'protocol.json', dict(scope='All88 saved box/keypoint records per seed; existing48 mask diagnostic records per seed only.',
        seeds=[17,29], thresholds=dict(score=.5, box_iou=.5, mask_iou=.5),
        mask_duplicate_definition='At least2 score>=.5 queries each independently having best GT mask IoU>=.5, with duplicate best GT identities.',
        oracle='Maximum one-to-one threshold coverage using GT; diagnostic upper bound, never inference or replacement evaluation.',
        model_forwards=0, optimizer_steps=0, code_sha256=code, production_promoted=False))
    box_rows = []
    for seed in ('seed17', 'seed29'):
        for idx, entry in enumerate(spec['evaluation']):
            path = BOX/seed/'regressed'/f'prediction_{idx:03d}.pt'
            inputs[str(path)] = fingerprint(path)
            saved = torch.load(path, weights_only=True, map_location='cpu')
            pred, target, record = saved['predictions'], saved['targets'], saved['record']
            train_matches = hungarian_match_instances(pred, target)[0]
            for view, (tq, tm) in enumerate(train_matches):
                valid = torch.where(target['instance_valid'][0, view])[0]
                boxes = pred['instance_boxes_cxcywh'][0, view]
                scores = pred['instance_objectness_logits'][0, view].sigmoid()
                gtboxes = target['boxes_cxcywh'][0, view, valid]
                overlap = box_iou(gtboxes, boxes)
                eq, em = saved['oracle_assignment'][view]
                q2, m2 = _linear_sum_assignment(5*torch.cdist(gtboxes, boxes, p=1)-scores[None])
                assert mapping(q2, valid[m2]) == mapping(eq, em)
                emap, tmap = mapping(eq, em), mapping(tq, tm)
                found = sum(scores[emap[int(m)]] >= .5 and overlap[j, emap[int(m)]] >= .5 for j, m in enumerate(valid))
                xy = pred['instance_keypoint_xy'][0,view]
                scale = xy.new_tensor(record['image_hw'][::-1])
                nme = []
                for m in valid:
                    good = target['keypoint_valid'][0,view,m]
                    diag = (target['boxes_cxcywh'][0,view,m,2:]*scale).norm().clamp_min(1.)
                    errors = ((xy-target['keypoint_xy'][0,view,m])*scale).norm(dim=-1)/diag
                    nme.append(errors[:,good].mean(-1).tolist() if good.any() else [None]*len(scores))
                box_rows.append(dict(seed=seed,index=idx,group=entry['group'],frame=record['frame'],
                    scene=record['scene'],view=view,camera=record['views'][view],role=record['physical_roles'][view],
                    target_instances=valid.tolist(),scores=scores.tolist(),box_iou=overlap.tolist(),keypoint_nme=nme,
                    evaluation_assignment=emap,training_assignment=tmap,assignment_changed=emap!=tmap,
                    evaluation_detected=int(found),gt_instances=len(valid),active_queries=int((scores>=.5).sum()),
                    diagnostic_best_active_box_coverage=maximum_coverage(overlap,scores),
                    diagnostic_best_any_box_coverage=maximum_coverage(overlap)))
    groups = defaultdict(list)
    for row in stats:
        groups[(row['seed'],row['tag'],row['view'])].append(row)
    mask_rows = []
    for (seed,tag,view), rows in sorted(groups.items()):
        rows.sort(key=lambda r:r['query'])
        assert [r['query'] for r in rows] == list(range(6))
        ids = [m['instance'] for m in rows[0]['gt_mask_ious']]
        assert all([m['instance'] for m in r['gt_mask_ious']] == ids for r in rows)
        overlap = torch.tensor([[r['gt_mask_ious'][j]['iou'] for r in rows] for j in range(len(ids))])
        scores = torch.tensor([r['score'] for r in rows])
        best = overlap.argmax(0)
        meaningful = (scores>=.5)&(overlap.max(0).values>=.5)
        identities = best[meaningful].tolist()
        matched = sum(r['score']>=.5 and next(m['iou'] for m in r['gt_mask_ious'] if m['instance']==r['matched_instance'])>=.5
                      for r in rows if r['matched_instance'] is not None)
        mask_rows.append(dict(seed=seed,tag=tag,view=view,subset=rows[0]['subset'],scene=rows[0]['scene'],
            role=rows[0]['role'],camera=rows[0]['camera'],gt_instances=len(ids),target_instances=ids,
            scores=scores.tolist(),mask_iou=overlap.tolist(),
            duplicate_strong_mask_queries=len(identities)>len(set(identities)),
            diagnostic_evaluation_assigned_mask_coverage=int(matched),
            diagnostic_best_active_mask_coverage=maximum_coverage(overlap,scores),
            diagnostic_best_any_mask_coverage=maximum_coverage(overlap),
            best_iou_per_gt=overlap.max(1).values.tolist()))
    summaries = []
    for seed in ('seed17','seed29'):
        for group in sorted({r['group'] for r in box_rows}):
            rows = [r for r in box_rows if r['seed']==seed and r['group']==group]
            keys = ('gt_instances','evaluation_detected','active_queries','assignment_changed',
                    'diagnostic_best_active_box_coverage','diagnostic_best_any_box_coverage')
            summaries.append(dict(seed=seed,group=group,views=len(rows),**{k:sum(r[k] for r in rows) for k in keys}))
    mask_summary = []
    for seed in ('seed17','seed29'):
        for subset in ('train','selected_dev_failure'):
            rows = [r for r in mask_rows if r['seed']==seed and r['subset']==subset]
            keys = ('gt_instances','duplicate_strong_mask_queries','diagnostic_evaluation_assigned_mask_coverage',
                    'diagnostic_best_active_mask_coverage','diagnostic_best_any_mask_coverage')
            mask_summary.append(dict(seed=seed,subset=subset,views=len(rows),**{k:sum(r[k] for r in rows) for k in keys}))
    focus = [r for r in mask_rows if r['tag']=='dev_007' and r['role']=='top']
    # Show every query, including rejected low-score ones; no GT-guided output selection.
    folder = MASK/'seed29/dev_007'
    r = next(r for r in focus if r['seed']=='seed29')
    rgbpath = ROOT/r['scene']/'blended_images'/f'{r["camera"]:08d}.jpg'
    inputs[str(rgbpath)] = fingerprint(rgbpath)
    rgb = np.array(Image.open(rgbpath).resize((768,432)))
    fig, axes = plt.subplots(2,4,figsize=(16,5),layout='constrained')
    for ax in axes.flat: ax.imshow(rgb);ax.axis('off')
    axes.flat[0].set_title('RGB / frame296 top')
    axes.flat[1].set_title('GT visible masks')
    for m in r['target_instances']:
        p=folder/f'v{r["view"]}_gt{m}.png'; inputs[str(p)]=fingerprint(p)
        mask=np.array(Image.open(p))>0
        color=[mask,np.zeros_like(mask),np.zeros_like(mask)] if m==0 else [np.zeros_like(mask),mask,np.zeros_like(mask)]
        axes.flat[1].imshow(np.dstack([*color,mask*.5]))
    for q in range(6):
        p=folder/f'v{r["view"]}_q{q}.png'; inputs[str(p)]=fingerprint(p)
        mask=np.array(Image.open(p))>0
        axes.flat[q+2].imshow(np.dstack([mask,np.zeros_like(mask),mask,mask*.5]))
        ious=[row[q] for row in r['mask_iou']]
        axes.flat[q+2].set_title(f'q{q} score={r["scores"][q]:.5g}\nGT mask IoU={ious[0]:.3f}/{ious[1]:.3f}',fontsize=9)
    fig.savefig(OUT/'frame296_all_queries_seed29.png',dpi=140);plt.close(fig)
    assert all(fingerprint(Path(p))==h for p,h in inputs.items())
    assert all(fingerprint(ROOT/p)==h for p,h in code.items())
    report=dict(passed=True,box_summaries=summaries,mask_summaries=mask_summary,frame296_top=focus,
                input_sha256=inputs,code_sha256=code,model_forwards=0,optimizer_steps=0,production_promoted=False,
                limitations=['GT optimal coverage is an upper bound, not an inference method.',
                             'Selected8 development failure records are biased, not complete88 mask evaluation.',
                             'Existing40 train poses use representative views only, not full356 inputs.',
                             'Duplicate definition requires mask IoU>=.5; weaker overlaps remain visible in raw matrices.'])
    write_json(OUT/'box_rows.json',box_rows);write_json(OUT/'mask_rows.json',mask_rows)
    write_json(OUT/'report.json',report)
    write_json(OUT/'status.json',dict(state='completed',elapsed_s=time.monotonic()-started,model_forwards=0,optimizer_steps=0))
    print(json.dumps({k:report[k] for k in ('box_summaries','mask_summaries','frame296_top')},indent=2))


if __name__=='__main__':
    main()
