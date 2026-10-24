"""Replay the recovered v3 design generator without Blender or rendering.

The original generator was an inline Python command on 2026-10-18. This
extracted CLI preserves its random draw order and the visibility amendment.
It verifies both historical designs and final shard configs before writing
into a NEW directory. Historical STOP/deadline fields are audit data only.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random


RUN = Path('output/bottom_geometry_v3_local12h_20261018')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate_initial(sources):
    # Original seed and exact draw order recovered from the executed command.
    sources = json.loads(json.dumps(sources))
    for source in sources:
        source['frames'] = [49, 97, 145]
    rng = random.Random(2026101803)
    rigs = []
    for split, n in [('train', 48), ('val', 12), ('sealed', 8)]:
        for i in range(n):
            extrap = split != 'train' and i >= n // 2
            focal = (rng.uniform(26, 29) if extrap and i % 2 else
                     rng.uniform(56, 60) if extrap else rng.uniform(30, 55))
            theta = rng.uniform(-math.pi, math.pi)
            rad = rng.uniform(105, 125) if extrap else rng.uniform(0, 95)
            h = rng.uniform(150, 245) if not extrap else rng.uniform(250, 280)
            sideangle = rng.uniform(0, 2 * math.pi)
            radius = rng.uniform(215, 290)
            pos = [[rad * math.cos(theta), rad * math.sin(theta), -h]]
            for angle in [sideangle, sideangle + rng.uniform(1.5, 2.7)]:
                pos.append([radius * math.cos(angle), radius * math.sin(angle),
                            rng.uniform(45, 110)])
            rigs.append(dict(id=f'{split}_rig{i:02d}', split=split,
                             lens_mm=focal, relative_positions=pos,
                             roll_deg=[rng.uniform(-30, 30), rng.uniform(-8, 8),
                                       rng.uniform(-8, 8)],
                             dev_tier='extrapolation' if extrap else 'interpolation'))
    return dict(sources=sources, rigs=rigs, expected_scenes=408,
                expected_views=1224, water_reaching_excluded=True,
                anipose_restrained_excluded=True,
                limitation='Two admitted motion groups crossed with all rigs; CRIM13 free-moving motion included in replay only, not a third crossed rig-motion asset. Held-out motions and TEST not promoted to train.',
                stopped_reach_smoke='Completed render-only engineering smoke before exclusion; never admitted to training or scoring.')


def replay(reference):
    archived = reference / 'render_preparation_attempt0'
    # Read only the two admitted source descriptors; no .blend file is opened.
    initial = generate_initial(read(archived / 'render_design.json')['sources'])
    screen = read(reference / 'visibility_prescreen.json')
    if not screen['passed'] or screen['model_calls'] != 0 or screen['RGB_renders'] != 0:
        raise ValueError('Expected the passed, model-free v3 visibility receipt')
    screen_script = reference.parents[1] / 'BlenderScripts/screen_v3_rig_visibility_20261018.py'
    if sha(screen_script) != screen['script_sha256']:
        raise ValueError('Visibility script hash mismatch')
    if initial['sources'] != screen['sources']:
        raise ValueError('Visibility source descriptors changed')
    accepted = {rig['id']: rig for rig in screen['rigs']}
    if len(accepted) != 68 or set(accepted) != {r['id'] for r in initial['rigs']}:
        raise ValueError('Visibility rig IDs changed')
    changed = [r['id'] for r in initial['rigs'] if r != accepted[r['id']]]
    if changed != screen['changed_rigs'] or changed != ['train_rig18', 'train_rig20']:
        raise ValueError('Unexpected v3 visibility changes')
    final = dict(initial)
    final['rigs'] = screen['rigs']
    final['visibility_admission'] = dict(
        changed_rigs=screen['changed_rigs'], rule=screen['rule'], model_calls=0,
        receipt_sha256=sha(reference / 'visibility_prescreen.json'))
    outputs = {
        'render_preparation_attempt0/render_design.json': initial,
        'render_design.json': final,
    }
    legacy = sha(archived / 'renderer_source.py')
    plan = read(reference / 'plan.json')
    for shard in [0, 1]:
        cfg = dict(output=f'{RUN.as_posix()}/render_shard{shard}',
                   stop=f'{RUN.as_posix()}/STOP',
                   deadline_utc=plan['optimization_deadline_utc'], resolution=384,
                   min_body_pixels=100, sources=initial['sources'],
                   rigs=initial['rigs'][shard::2], smoke_allow_rejected=False)
        # The original inline visibility amendment replaced rigs by ID.
        cfg['rigs'] = [accepted[r['id']] for r in cfg['rigs']]
        cfg['accepted_legacy_script_sha256'] = legacy
        cfg['visibility_prescreen_sha256'] = sha(reference / 'visibility_prescreen.json')
        outputs[f'render_config_shard{shard}.json'] = cfg
    encoded = {name: json.dumps(value, indent=2).encode('utf-8')
               for name, value in outputs.items()}
    for name, payload in encoded.items():
        if payload != (reference / name).read_bytes():
            raise ValueError('Historical byte-for-byte replay failed: ' + name)
    return encoded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', type=Path,
                        default=Path(__file__).resolve().parents[1] / RUN)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    payloads = replay(args.reference_dir.resolve())
    # mkdir(exist_ok=False) prevents writes into any existing run/directory.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for name, payload in payloads.items():
        target = args.output_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        rows.append(dict(path=name, bytes=len(payload),
                         sha256=hashlib.sha256(payload).hexdigest()))
    receipt = dict(passed=True, byte_identical_files=rows, random_seed=2026101803,
                   rig_counts=dict(train=48, val=12, sealed=8),
                   scenes=408, views=1224, blender_invoked=False,
                   historical_outputs_modified=False,
                   warning='Historical v3 config deadlines are expired; do not launch rendering from these audit files.')
    (args.output_dir / 'replay_receipt.json').write_text(
        json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
