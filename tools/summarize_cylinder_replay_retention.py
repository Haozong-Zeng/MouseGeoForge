"""Recover completed preflight rows after numpy-int JSON failure, without forwards."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_depth_pose_ablation import ROOT, fingerprint, write_json

SOURCE = ROOT / 'output/cylinder_retention_preflight_v1_retry'
OUT = ROOT / 'output/cylinder_retention_preflight_v1_summary'


def summarize(quality, gradients):
    metrics = {}
    for tag in ('teacher', 'student'):
        subset = [q for q in quality if q['checkpoint'] == tag]
        metrics[tag] = dict(instance_views=len(subset), box_iou=float(np.mean([q['iou'] for q in subset])),
            **{k + '_nme': float(np.mean([e for q in subset for e in q[k + '_errors']])) for k in ('body', 'tail')})
    gates = {k: bool(metrics['teacher'][k] >= metrics['student'][k] if k == 'box_iou' else
                    metrics['teacher'][k] <= metrics['student'][k]) for k in ('box_iou', 'body_nme', 'tail_nme')}
    ratios = {g: float(np.median([r['ratio'] for r in gradients if r['checkpoint'] == 'student' and r['group'] == g]))
              for g in ('shared_adapter', 'instance_head')}
    gates.update({g + '_gradient': x <= .25 for g, x in ratios.items()})
    gates['nonempty_qualified_targets'] = metrics['teacher']['instance_views'] > 0
    by_tag = {t: {(q['record'], q['view'], q['identity']): q for q in quality if q['checkpoint'] == t} for t in metrics}
    teacher, student = by_tag['teacher'], by_tag['student']
    assert teacher.keys() == student.keys()
    violations = dict(box=int(sum(teacher[k]['iou'] < student[k]['iou'] for k in teacher)),
        **{part: int(sum(np.mean(teacher[k][part + '_errors']) > np.mean(student[k][part + '_errors']) for k in teacher))
           for part in ('body', 'tail')})
    return dict(training_supported=all(gates.values()), gates=gates, quality=metrics,
                individual_quality_violations=violations, student_median_gradient_ratios=ratios)


def main():
    if OUT.exists():
        raise RuntimeError('Do not overwrite recovery')
    protocol = json.loads((SOURCE / 'protocol.json').read_text())
    quality = json.loads((SOURCE / 'quality.json').read_text())
    gradients = json.loads((SOURCE / 'gradients.json').read_text())
    assert len(gradients) == 32 and {r['record'] for r in gradients} == set(range(8))
    assert len({(r['checkpoint'], r['record'], r['group']) for r in gradients}) == 32
    assert all(np.isfinite(r[k]) for r in gradients for k in ('ratio', 'base_norm', 'aux_norm', 'auxiliary_loss'))
    assert all(r['aux_norm'] == r['auxiliary_loss'] == 0 for r in gradients if r['checkpoint'] == 'teacher')
    for p, h in protocol['code_sha256'].items():
        assert fingerprint(ROOT / p) == h
    for ck in protocol['checkpoints'].values():
        assert fingerprint(ROOT / ck['path']) == ck['sha256']
    teachers = {str(p.relative_to(ROOT)): fingerprint(p) for p in sorted(SOURCE.glob('teacher_*.pt'))}
    assert len(teachers) == 8
    report = summarize(quality, gradients)
    report.update(passed=True, teacher_targets_sha256=teachers, teacher_initialization_auxiliary_zero=True,
        zero_weight_max_difference=0, student_matcher_calls_per_loss=1,
        verification_evidence='Saved rows exist only after all eight per-record exact OFF/self-zero/one-match/state checks. Final serialization failed on numpy.int64 violation count, after source/code/checkpoint assertions.',
        optimizer_steps=0, recovery_encoder_forwards=0, recovery_head_forwards=0,
        preflight_encoder_forwards=8, preflight_head_forwards=24,
        source_files_sha256={p: fingerprint(SOURCE / p) for p in ('protocol.json', 'quality.json', 'gradients.json')},
        recovery_code_sha256=fingerprint(Path(__file__)))
    # Serialize the exact recovered report before publishing any artifact.
    json.dumps(report, allow_nan=False)
    write_json(OUT / 'report.json', report)
    write_json(OUT / 'status.json', dict(state='completed', training_supported=report['training_supported']))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout='constrained')
    for ax, key in zip(axes, ('box_iou', 'body_nme', 'tail_nme')):
        values = [report['quality'][t][key] for t in ('teacher', 'student')]
        ax.bar(['original teacher', 'mixed student'], values, color=['#698fa9', '#c38a4c'])
        ax.set_title(key + (' (higher better)' if key == 'box_iou' else ' (lower better)'))
        for i, value in enumerate(values):
            ax.text(i, value, f'{value:.4f}', ha='center', va='bottom')
        ax.set_ylim(0, max(values) * 1.15)
    fig.suptitle('8 dual TRAIN poses | same teacher-qualified cohort | no updates')
    fig.savefig(OUT / 'teacher_quality.png', dpi=145)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
