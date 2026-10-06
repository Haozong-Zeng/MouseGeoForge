"""Compact diagnostic figure from the locked calibration run; no model changes."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / 'output/occlusion_pose_v2/calibration'
    report = json.loads((folder / 'calibration_report.json').read_text())
    geometry = json.loads((folder / 'geometry_test.json').read_text())['summary']
    metrics = report['variants']['test']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.9), layout='constrained')
    keys = ['original_sigma', 'temperature_original_sigma', 'failed_train_sigma',
            'temperature_failed_train_sigma', 'constant']
    labels = ['Original', 'Original + T\nPRIMARY', 'Learned', 'Learned + T\nsecondary', 'Constant']
    colors = ['#8e99a6', '#416ea5', '#b49979', '#168b83', '#abb0b3']
    ax = axes[0]
    values = [metrics[k]['coverage_95'] * 100 for k in keys]
    ax.bar(range(len(keys)), values, color=colors)
    ax.axhline(95, color='#444444', ls='--', lw=1, label='Nominal 95%')
    for i, y in enumerate(values):
        ax.text(i, y + 1.2, f'{y:.1f}', ha='center', fontsize=9)
    ax.set(xticks=range(len(keys)), xticklabels=labels, ylim=(0, 108),
           ylabel='Empirical coverage (%)', title='2D uncertainty coverage')
    ax.tick_params(axis='x', labelsize=8)
    ax.legend(loc='lower left', fontsize=9)
    ax = axes[1]
    for key, label, color in [('original_sigma', 'Original ranking', colors[1]),
                               ('failed_train_sigma', 'Learned ranking (secondary)', colors[3])]:
        curve = metrics[key]['risk_coverage']
        ax.plot([25, 50, 75, 100], [curve[str(f)] for f in (.25, .5, .75, 1.)],
                marker='o', color=color, label=label)
    ax.set(xlabel='Lowest-uncertainty points retained (%)', ylabel='Mean 2D error (normalized UV)',
           title='Reliability ranking; T does not alter ranking')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(alpha=.2)
    ax = axes[2]
    rows = [geometry[k] for k in ('uniform', 'temperature_original_sigma')]
    coverage = [r['coverage'] * 100 for r in rows]
    ax.bar([0, 1], coverage, color=[colors[0], colors[1]], width=.55)
    for i, r in enumerate(rows):
        ax.text(i, coverage[i] + 2, f'{coverage[i]:.1f}%', ha='center')
        ax.text(i, 35, f'Paired error\n{r["paired_mean_mm"]:.3f} mm', ha='center', color='white')
    ax.set(xticks=[0, 1], xticklabels=['Uniform robust', 'Original + T'], ylim=(0, 100),
           ylabel='Supported 3D points (%)', title='Primary fusion gate: FAIL')
    fig.suptitle('Fixed 768 pose model | 48 calibration + 48 new test scenes | no pose training', fontsize=14)
    fig.supxlabel('Synthetic two-mouse data; same held-out source session across blocks; GT-assisted instance pairing.\n'
                  'No production fusion enabled. Constant scale cannot rank reliability.', fontsize=10)
    path = folder / 'calibration_summary.png'
    fig.savefig(path, dpi=160)
    print(path)


if __name__ == '__main__':
    main()
