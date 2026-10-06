"""Deterministic adaptation-replay coverage diagnostics (not a new test split)."""
import numpy as np


def select_unreplayed(records, replay_scenes, count=16):
    train = [r for r in records if r['split'] == 'train' and r['expected_instances'] == 2]
    if len({r['scene'] for r in train}) != len(train):
        raise ValueError('Duplicate training scene')
    replay = [r for r in train if r['scene'] in replay_scenes]
    if len(replay) != len(replay_scenes):
        raise ValueError('Replay scene is not in dual TRAIN manifest')
    groups = {r['source_group'] for r in replay}
    if len(groups) != 1:
        raise ValueError('This audit requires one fixed replay source')
    candidates = sorted([r for r in train if r['scene'] not in replay_scenes and r['source_group'] in groups],
                        key=lambda r: (r['source_frame'], r['frame'], r['scene']))
    if count < 2 or len(candidates) < count:
        raise ValueError('Insufficient nonreplay TRAIN poses')
    return [candidates[round(i * (len(candidates) - 1) / (count - 1))] for i in range(count)]


def summarize_views(rows):
    result = []
    for tag in sorted({r['checkpoint'] for r in rows}):
        for cohort in sorted({r['cohort'] for r in rows}):
            for role in ('all', 'side', 'top'):
                selected = [r for r in rows if r['checkpoint'] == tag and r['cohort'] == cohort
                            and (role == 'all' or r['role'] == role)]
                if not selected:
                    continue
                values = dict(checkpoint=tag, cohort=cohort, role=role, instance_views=len(selected),
                              recall=float(np.mean([r['found'] for r in selected])),
                              box_iou=float(np.mean([r['iou'] for r in selected])))
                for part in ('body', 'tail', 'core23'):
                    errors = [e for r in selected for e in r[part + '_errors']]
                    passed = [e <= .1 and r['found'] for r in selected for e in r[part + '_errors']]
                    values[part + '_nme'] = float(np.mean(errors)) if errors else None
                    values[part + '_detection_pck'] = float(np.mean(passed)) if passed else None
                result.append(values)
    return result
