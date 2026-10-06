"""Fixed predeclared snapshots and same-pose changes from frozen coverage audit."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from run_depth_pose_ablation import ROOT, torch, fingerprint, write_json
from mousegeoforge.retention_geometry_audit import matched_views

OUT = ROOT / 'output/cylinder_replay_coverage_v1'


def main():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    report = json.loads((OUT / 'report.json').read_text())
    assert report['passed'] and json.loads((OUT / 'status.json').read_text())['state'] == 'completed'
    if (OUT / 'visualization_verification.json').exists():
        raise RuntimeError('Do not overwrite visualization')
    for idx in protocol['fixed_visual_indices']:
        entry = protocol['cohort'][idx]; r = entry['record']; h, w = r['image_hw']; pixel = np.array([w, h])
        fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), layout='constrained')
        for col, tag in enumerate(('original', 'mixed')):
            saved = torch.load(OUT / tag / f'prediction_{idx:03d}.pt', weights_only=True, map_location='cpu')
            pred, target = saved['predictions'], saved['targets']; views = matched_views(saved)
            for v in range(2):
                ax = axes[v, col]
                ax.imshow(Image.open(ROOT / r['scene'] / 'blended_images' / f'{r["views"][v]:08d}.jpg').resize((w, h)))
                descriptions = []
                for row in [x for x in views if x['view'] == v]:
                    q, m = row['query'], row['instance']; valid = row['valid'].numpy().copy()
                    gt = target['keypoint_xy'][0, v, m].numpy() * pixel
                    xy = pred['instance_keypoint_xy'][0, v, q].numpy() * pixel
                    body = valid.copy(); body[7:14] = False; tail = valid & ~body
                    for points, color, marker in ((gt[body], 'lime', 'o'), (xy[body], 'magenta', 'x'),
                                                   (gt[tail], 'gold', 'o'), (xy[tail], 'cyan', 'x')):
                        ax.scatter(points[:, 0], points[:, 1], c=color, s=15, marker=marker, linewidths=.8)
                    for a, b in zip(gt[body], xy[body]):
                        ax.plot([a[0], b[0]], [a[1], b[1]], c='magenta', lw=.4, alpha=.55)
                    for box, color, ls in ((target['boxes_cxcywh'][0, v, m], 'lime', '--'),
                                           (pred['instance_boxes_cxcywh'][0, v, q], 'magenta', '-')):
                        box = box.numpy(); lo = (box[:2] - box[2:]/2) * pixel; size = box[2:] * pixel
                        ax.add_patch(Rectangle(lo, *size, fill=False, edgecolor=color, linewidth=1, linestyle=ls))
                    descriptions.append(f'ID {int(target["instance_ids"][0,v,m])}: IoU {row["iou"]:.2f}, body {float(row["errors"][body].mean()):.3f}')
                ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.axis('off')
                ax.set_title(f'{tag} / {r["physical_roles"][v]}\n' + ' | '.join(descriptions), fontsize=10)
        fig.suptitle(f'{entry["cohort"]} TRAIN pose frame {r["frame"]}, source frame {r["source_frame"]}\n'
                     'Body: GT green / prediction magenta; tail: GT gold / prediction cyan. Same box-assisted assignment.')
        fig.savefig(OUT / f'fixed_{entry["cohort"]}.png', dpi=150); plt.close(fig)
    summary = report['summary']
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), layout='constrained')
    for ax, key in zip(axes, ('box_iou', 'body_nme', 'tail_nme', 'recall')):
        for offset, tag, color in ((-.18, 'original', '#698fa9'), (.18, 'mixed', '#c38a4c')):
            values = [next(r[key] for r in summary if (r['checkpoint'], r['cohort'], r['role']) == (tag, c, 'all')) for c in ('replayed', 'nonreplayed')]
            ax.bar(np.arange(2) + offset, values, width=.36, color=color, label=tag)
        ax.set_title(key); ax.set_xticks([0, 1], ['8 replayed', '16 nonreplayed'])
    axes[0].legend(); fig.suptitle('Synthetic TRAIN, not unseen by original training | no optimizer updates')
    fig.savefig(OUT / 'coverage_summary.png', dpi=150); plt.close(fig)
    metrics = json.loads((OUT / 'metrics.json').read_text())
    paired = []
    for cohort in ('replayed', 'nonreplayed'):
        records = sorted({m['record'] for m in metrics if m['cohort'] == cohort})
        lookup = {(m['checkpoint'], m['record']): m['metrics'] for m in metrics}
        paired.append(dict(cohort=cohort, records=len(records),
            **{key + '_improved_poses': int(sum(lookup['mixed', i][key] < lookup['original', i][key] for i in records))
               for key in ('box_matched_nme', 'body_keypoint_nme', 'tail_nme')}))
    write_json(OUT / 'paired_changes.json', paired)
    write_json(OUT / 'visualization_verification.json', dict(passed=True,
        protocol_sha256=fingerprint(OUT / 'protocol.json'), report_sha256=fingerprint(OUT / 'report.json'),
        code_sha256=fingerprint(ROOT / 'tools/visualize_cylinder_replay_coverage.py'), fixed_indices=protocol['fixed_visual_indices']))
    print(json.dumps(paired, indent=2))


if __name__ == '__main__': main()
