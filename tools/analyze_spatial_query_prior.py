"""CPU assessment of spatial prior; frozen ON initialization is not learned benefit."""
import json
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from torch.nn import functional as F
from run_depth_pose_ablation import ROOT, torch, write_json, fingerprint
from analyze_cylinder_view_balance import gate_results
from mousegeoforge.retention_geometry_audit import matched_views

OUT = ROOT / 'output/spatial_query_prior_seed17_v1'
PRIOR = ROOT / 'output/spatial_query_prior_preflight_v3'
ARMS = ('spatial_prior_off', 'spatial_prior_on')


def visualize():
    tags = ('initial', *ARMS)
    fig, axes = plt.subplots(4, 3, figsize=(15, 11), layout='constrained')
    for row, index in enumerate((0, 7, 9, 13)):
        for col, tag in enumerate(tags):
            p = torch.load(OUT / tag / f'prediction_{index:03d}.pt', map_location='cpu', weights_only=True)
            m = torch.load(OUT / tag / f'mask_visual_{index:03d}.pt', map_location='cpu', weights_only=True)
            rec = p['record']; v = rec['physical_roles'].index('top')
            h, w = rec['image_hw']; scale = np.array([w, h])
            ax = axes[row, col]
            ax.imshow(Image.open(ROOT / rec['scene'] / 'blended_images' / f'{rec["views"][v]:08d}.jpg').resize((w, h)))
            for target in torch.where(p['targets']['instance_valid'][0, v])[0]:
                b = p['targets']['boxes_cxcywh'][0, v, target].numpy()
                ax.add_patch(Rectangle((b[:2]-b[2:]/2)*scale, *(b[2:]*scale), fill=False, edgecolor='lime', lw=1))
            active = torch.where(p['predictions']['instance_objectness_logits'][0, v].sigmoid() >= .5)[0]
            for q in active:
                color = np.array(plt.get_cmap('tab10')(int(q))[:3])
                mask = F.interpolate(m['mask_logits'][0, v, q][None, None], size=(h, w), mode='bilinear', align_corners=False)[0, 0].numpy() > 0
                ax.imshow(np.dstack([np.broadcast_to(color, (h, w, 3)), mask * .4]))
                b = p['predictions']['instance_boxes_cxcywh'][0, v, q].numpy()
                ax.add_patch(Rectangle((b[:2]-b[2:]/2)*scale, *(b[2:]*scale), fill=False, edgecolor=color, lw=1))
                ax.text(b[0]*w, b[1]*h, f'q{int(q)}', color=color, fontsize=9)
            ax.set_title(f'{tag} / frame {rec["frame"]} / top / {len(active)} queries', fontsize=9)
            ax.axis('off')
    fig.suptitle('All score>=0.5 queries; colors are slots, not identity. GT boxes green; no GT output selection.')
    fig.savefig(OUT / 'fixed_dual_masks.png', dpi=140); plt.close(fig)


def main():
    if (OUT / 'acceptance.json').exists():
        raise RuntimeError('Preserve completed analysis')
    torch.set_num_threads(2)
    verification = json.loads((OUT / 'verification.json').read_text())
    assert verification['passed'] and verification['optimizer_steps_per_arm'] == 1600
    results = json.loads((OUT / 'results.json').read_text())
    prior = json.loads((PRIOR / 'report.json').read_text())['evaluations']
    baseline = json.loads((ROOT / 'output/cylinder_source_mix_v1/results.json').read_text())['baseline']
    control = json.loads((ROOT / 'output/cylinder_view_balance_v1/results.json').read_text())['arms']['side_control']
    values = {'initial': results['baseline'], **results['arms']}
    gates = {}; summaries = {}; hashes = {}; coords = {}
    for tag, value in values.items():
        gates[tag] = gate_results(dict(config=results['config'], baseline=baseline,
                                      arms=dict(side_control=control, balanced_views=value)))['balanced_views']
        assert gates[tag]['total'] == 17
        quality = json.loads((OUT / f'{tag}_instance_quality.json').read_text())
        groups = defaultdict(list); parts = defaultdict(list)
        for index, record in enumerate(value['records']):
            path = OUT / tag / f'prediction_{index:03d}.pt'; hashes[str(path)] = fingerprint(path)
            saved = torch.load(path, map_location='cpu', weights_only=True)
            for row in matched_views(saved):
                groups[record['group']].append(row)
                if record['group'] == 'old_dual_pair':
                    for part, ids in (('body', list(range(7))+list(range(14, 23))), ('tail', list(range(7, 14)))):
                        parts[part].extend(row['errors'][ids][row['valid'][ids]].tolist())
        summaries[tag] = {group: dict(tp=sum(r['found'] for r in rows), gt=len(rows),
                    box_iou=float(np.mean([r['iou'] for r in rows])),
                    active=sum(q['active_queries'] for q in quality if q['group']==group),
                    duplicate_views=sum(q['duplicate_strong_mask'] for q in quality if q['group']==group))
                    for group, rows in groups.items()}
        coords[tag] = {part: float(np.mean(errors)) for part, errors in parts.items()}
    off, on = [values[a]['summary']['old_dual_pair'] for a in ARMS]
    def total(tag, key):
        return sum(row[key] for row in summaries[tag].values())
    criteria = dict(all17_gates=gates[ARMS[1]]['passed'],
        pck_gain_at_least_01=on['detection_aware_pck_0.1']-off['detection_aware_pck_0.1'] >= .01,
        coordinate_nme_ratio_at_most_102=on['box_matched_nme'] <= 1.02*off['box_matched_nme'],
        tp_not_lower=summaries[ARMS[1]]['old_dual_pair']['tp'] >= summaries[ARMS[0]]['old_dual_pair']['tp'],
        fp_not_higher=total(ARMS[1], 'active')-total(ARMS[1], 'tp') <= total(ARMS[0], 'active')-total(ARMS[0], 'tp'),
        duplicate_views_not_higher=total(ARMS[1], 'duplicate_views') <= total(ARMS[0], 'duplicate_views'))
    budget_rows = []
    for arm in ARMS:
        for steps, data in ((0, prior['off' if arm == ARMS[0] else 'on_initial']), (1600, results['arms'][arm])):
            row = data['summary']['old_dual_pair']
            budget_rows.append(dict(arm=arm, steps=steps, run='frozen_initial' if steps==0 else 'current1600',
                                   metrics=row))
    report = dict(passed_analysis=True, gates=gates, groups=summaries, dual_coordinate_parts=coords,
        pck_gain=on['detection_aware_pck_0.1']-off['detection_aware_pck_0.1'], criteria=criteria,
        budget_comparison=budget_rows, seed29_eligible=all(criteria.values()), production_promoted=False,
        decision='Eligible for second seed after visual review.' if all(criteria.values()) else
                 'No practical spatial-prior benefit at fixed1600; inspect learned spatial parameters before another independently preregistered mechanism. Do not blindly extend.',
        limitations=['Frozen ON initialization is nonzero. Its metrics are not learned gains; compare trained ON to same-budget OFF.',
                     'Reused synthetic development; no real Cylinder test or statistical significance claim.',
                     'Native-grid mask duplicate counts are diagnostic; historical full-image metrics unchanged.'],
        source_sha256={**hashes, str(OUT/'results.json'): fingerprint(OUT/'results.json'),
                       str(PRIOR/'report.json'): fingerprint(PRIOR/'report.json')},
        analysis_code_sha256=fingerprint(ROOT/'tools/analyze_spatial_query_prior.py'))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for arm in ARMS:
        history = json.loads((OUT / arm / 'history.json').read_text()); assert len(history)==1600
        axes[0].plot(np.arange(50, 1601), np.convolve([r['pose_loss'] for r in history], np.ones(50)/50, mode='valid'), label=arm)
        points = [r for r in budget_rows if r['arm']==arm]
        axes[1].plot([r['steps'] for r in points], [r['metrics']['detection_aware_pck_0.1'] for r in points], 'o-', label=arm)
    axes[0].set(title='50-slot training pose loss', xlabel='step'); axes[0].legend()
    axes[1].set(title='Dual PCK: frozen initialization vs1600', xlabel='fixed final budget', ylim=(0, 1)); axes[1].legend()
    fig.savefig(OUT/'comparison.png', dpi=140); plt.close(fig)
    visualize()
    write_json(OUT / 'acceptance.json', report)
    print(json.dumps(dict(criteria=criteria, pck_gain=report['pck_gain'],
                         gates={a:g['passed_count'] for a,g in gates.items()},
                         dual={a:summaries[a]['old_dual_pair'] for a in values},
                         decision=report['decision']), indent=2))


if __name__ == '__main__':
    main()


