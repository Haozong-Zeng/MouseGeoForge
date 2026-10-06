"""Read-only source audit; generated report is evidence, not convergence proof."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/query_position_budget_audit_v1'
SOURCE = ROOT / 'output/query_position_seed17_v1'


def main():
    if (OUT / 'report.json').exists():
        raise RuntimeError('Preserve existing report')
    paths = [SOURCE / 'loss_components.json']
    components = json.loads(paths[0].read_text())
    assert len(components) == 1600
    rows = []
    for arm_index, arm in enumerate(('query_position_off', 'query_position_on')):
        path = SOURCE / arm / 'history.json'
        paths.append(path)
        history = json.loads(path.read_text())
        assert [r['step'] for r in history] == list(range(1, 801))
        for kind in ('dual', 'single', 'cylinder'):
            selected = [(i, r) for i, r in enumerate(history) if r['slot_kind'] == kind]
            groups = defaultdict(list)
            for i, r in selected:
                groups[(r['scene'], tuple(r['views']))].append((i, r))
            repeated = [v for v in groups.values() if len(v) >= 4]
            first = mean(mean(r['pose_loss'] for _, r in v[:2]) for v in repeated)
            last = mean(mean(r['pose_loss'] for _, r in v[-2:]) for v in repeated)
            parts = {key: {
                'first_two': mean(mean(components[arm_index * 800 + i][key] for i, _ in v[:2]) for v in repeated),
                'last_two': mean(mean(components[arm_index * 800 + i][key] for i, _ in v[-2:]) for v in repeated)
            } for key in components[0]}
            rows.append(dict(arm=arm, kind=kind, comparable_repeated_inputs=len(repeated),
                             first_two_pose_loss=first, last_two_pose_loss=last,
                             relative_change=last / first - 1, components=parts,
                             quarter_means=[mean(r['pose_loss'] for _, r in selected if start < r['step'] <= start + 200)
                                            for start in (0, 200, 400, 600)]))
    report = dict(rows=rows, source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                  source_model_forwards=0, optimizer_steps=0,
                  interpretation='Paired-input training losses still change; ON has no distinct advantage. Only initial/final development exists, so convergence is unestablished, not disproven. Single-mouse losses worsen slightly. One bounded paired1600 check is warranted as budget sensitivity, not evidence of benefit.',
                  limitations=['Repeated-input losses are pre-update samples at different steps, not simultaneous train evaluation.',
                               'No independent validation trajectory or confidence intervals; do not infer generalization from training loss.'])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
