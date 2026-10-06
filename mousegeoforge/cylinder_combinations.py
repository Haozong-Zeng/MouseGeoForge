"""Finite-budget physical-camera coverage, independent of pose-number parity."""
from collections import Counter, defaultdict
from itertools import combinations
import random


def sample_orders(slots, roles, seed=17):
    """slots: (pose identifier, V). No development labels or camera IDs required.

    Decks must exhaust before recycling. Per-pose repetition breaks ties within
    a deck; camera ordering varies independently from the source pose number.
    V1 retains top/bottom focus; replay supplies side-only supervision.
    """
    side = sorted(k for k, v in roles.items() if v == 'side')
    poles = {r: [k for k, v in roles.items() if v == r] for r in ('bottom', 'top')}
    if len(side) != 6 or any(len(v) != 1 for v in poles.values()):
        raise ValueError('Expected verified six-side, one-bottom, one-top rig')
    if any(n not in (1, 2, 3, 4) for _, n in slots):
        raise ValueError('Only V=1..4 supported')
    b, t = poles['bottom'][0], poles['top'][0]
    rng = random.Random(seed)
    pools = {
        (2, 'pole'): [(s, p) for s in side for p in (b, t)] + [(p, s) for s in side for p in (b, t)],
        (3, 'both'): [(*s, b, t) for s in combinations(side, 1)],
        (3, 'bottom'): [(*s, b) for s in combinations(side, 2)],
        (3, 'top'): [(*s, t) for s in combinations(side, 2)],
        (4, 'both'): [(*s, b, t) for s in combinations(side, 2)],
        (4, 'bottom'): [(*s, b) for s in combinations(side, 3)],
        (4, 'top'): [(*s, t) for s in combinations(side, 3)],
    }
    counts = Counter(n for _, n in slots)
    strata = {}
    for n, pattern in ((3, ['both', 'bottom', 'top']), (4, ['both', 'both', 'bottom', 'top'])):
        strata[n] = [pattern[i % len(pattern)] for i in range(counts[n])]
        rng.shuffle(strata[n])
    decks = {}; subsets_seen = Counter(); orders_seen = Counter(); refs = Counter(); mono = defaultdict(list)
    orders = []
    for pose, n in slots:
        if n == 1:
            if not mono[pose]:
                mono[pose] = [b, t]; rng.shuffle(mono[pose])
            order = (mono[pose].pop(),)
        else:
            key = (n, 'pole' if n == 2 else strata[n].pop())
            if not decks.get(key):
                decks[key] = list(pools[key]); rng.shuffle(decks[key])
            candidates = decks[key]
            order = min(candidates, key=lambda x: (subsets_seen[pose, tuple(sorted(x))], orders_seen[pose, x]))
            candidates.remove(order)
            if n >= 3:
                choices = list(order); rng.shuffle(choices)
                first = min(choices, key=lambda v: (refs[pose, n, v], refs[None, n, v]))
                rest = [v for v in order if v != first]; rng.shuffle(rest)
                order = (first, *rest)
        subsets_seen[pose, tuple(sorted(order))] += 1
        orders_seen[pose, order] += 1
        refs[pose, n, order[0]] += 1; refs[None, n, order[0]] += 1
        orders.append(list(order))
    return orders


def coverage(slots, orders, roles):
    if len(slots) != len(orders):
        raise ValueError('Slot/order length mismatch')
    subsets = defaultdict(Counter); directed = Counter(); mono = defaultdict(set)
    refs = defaultdict(Counter); physical = Counter(); per_pose = defaultdict(set)
    for (pose, n), order in zip(slots, orders):
        if len(order) != n or len(set(order)) != n or any(v not in roles for v in order):
            raise ValueError('Invalid camera subset')
        subsets[n][tuple(sorted(order))] += 1
        refs[n][order[0]] += 1
        physical.update(roles[v] for v in order)
        per_pose[pose].add(tuple(order))
        if n == 1: mono[pose].add(roles[order[0]])
        if n == 2: directed[tuple(order)] += 1
    side = [k for k, v in roles.items() if v == 'side']
    poles = [k for k, v in roles.items() if v in ('top', 'bottom')]
    expected_pairs = {(s,p) for s in side for p in poles} | {(p,s) for s in side for p in poles}
    expected = {n: {x for x in combinations(sorted(roles), n) if any(v in poles for v in x)} for n in (3,4)}
    gates = dict(all_pose_mono_both_poles=bool(mono) and all(x == {'top','bottom'} for x in mono.values()),
        all_directed_side_pole_pairs=expected_pairs <= set(directed),
        all_pole_containing_triples=expected[3] <= set(subsets[3]),
        all_pole_containing_quadruples=expected[4] <= set(subsets[4]),
        every_camera_is_reference_v2_v3_v4=all(set(refs[n]) == set(roles) for n in (2,3,4)))
    return dict(passed=all(gates.values()), gates=gates,
        unordered_subset_counts={str(n): len(rows) for n, rows in subsets.items()},
        ordered_pairs={','.join(map(str,k)):v for k,v in sorted(directed.items())},
        reference_counts={str(n): {str(k):v for k,v in rows.items()} for n,rows in refs.items()},
        physical_image_counts=dict(physical),
        per_pose_mono_roles={p: sorted(r) for p,r in mono.items()},
        per_pose_distinct_orders={p: len(r) for p,r in per_pose.items()},
        limitation='Global coverage, not all combinations per pose. V1 uses poles only; original side replay is unchanged.')
