"""CPU-only paired summaries. Rig, never frame, is the unseen-rig bootstrap unit."""
import argparse
import collections
import json
from pathlib import Path

import numpy as np


def identity(row):
    return row['index'], tuple(row['context'])


def endpoints(rows, role):
    rotations = [c['rotation_deg'] for r in rows for c in r['cameras']
                 if c['star'] and c['pair_role'] == role]
    errors = [e for r in rows for e in r['errors']['raw_joint'][role]]
    depth = [v['common_pack_body_oracle']['absrel'] for r in rows for v in r['body_depth']
             if (v['role'] == 'bottom') == (role == 'bottom')]
    return dict(rotation_median=float(np.median(rotations)) if rotations else None,
                rotation_below15=float(np.mean(np.asarray(rotations) < 15)) if rotations else None,
                raw_joint_coverage32=sum(e is not None and e <= 32 for e in errors)/len(errors) if errors else None,
                body_common_pack_oracle_absrel=float(np.mean(depth)) if depth else None)


def compare(reference, treatment, diagnostic_common_support=False):
    if reference['version'] != 4 or treatment['version'] != 4:
        raise ValueError('Both reports must use v4')
    if reference['provenance']['manifest_sha256'] != treatment['provenance']['manifest_sha256']:
        raise ValueError('Different manifests cannot be a matched comparison')
    if reference['partial'] or treatment['partial']:
        raise ValueError('Smoke subsets are not endpoints')
    a = {identity(r): r for r in reference['rows']}; b = {identity(r): r for r in treatment['rows']}
    if set(a) != set(b) and not diagnostic_common_support:
        raise ValueError('Sample sets differ; inspect missing masks/results before comparison')
    common = sorted(set(a) & set(b)); grouped = collections.defaultdict(list)
    for key in common:
        aa, bb = a[key], b[key]
        if aa['evaluation_role'] != bb['evaluation_role']:
            raise ValueError('Role mismatch')
        for role in ('bottom', 'other'):
            if len(aa['errors']['raw_joint'][role]) != len(bb['errors']['raw_joint'][role]):
                raise ValueError('GT transfer support differs')
        source = aa['evaluation_role']
        unit = aa['rig'] or ('ONE_REAL_RIG' if source == 'known_rig_diagnostic' else aa['group'])
        grouped[source, unit].append(key)
    units = collections.defaultdict(list)
    for (source, unit), keys in sorted(grouped.items()):
        for role in ('bottom', 'other'):
            av, bv = endpoints([a[k] for k in keys], role), endpoints([b[k] for k in keys], role)
            delta = {name: bv[name]-value for name, value in av.items() if value is not None and bv[name] is not None}
            units[source, role].append(dict(unit=unit, n_packs=len(keys), reference=av, treatment=bv, delta=delta))
    result = []
    for (source, role), rows in sorted(units.items()):
        estimates = {}
        for metric in sorted({name for r in rows for name in r['delta']}):
            values = np.array([r['delta'][metric] for r in rows if metric in r['delta']])
            rng = np.random.default_rng(17)
            draws = rng.choice(values, (10000, len(values)), replace=True).mean(1) if len(values) > 1 else None
            estimates[metric] = dict(n_units=len(values), mean_paired_delta=float(values.mean()),
                                     descriptive_ci95=[float(x) for x in np.percentile(draws, [2.5, 97.5])] if draws is not None else None)
        result.append(dict(source=source, role=role, units=rows, macro_paired=estimates,
                           unit_definition='rig' if source in ('unseen_rig_gate', 'randomized_rig_train') else
                           'one physical real rig' if source == 'known_rig_diagnostic' else 'source/motion group; not guaranteed independent rigs'))
    return dict(version=4, diagnostics_only=diagnostic_common_support, common_packs=len(common),
                reference_unpaired=len(set(a)-set(b)), treatment_unpaired=len(set(b)-set(a)), comparisons=result,
                warning='Posthoc descriptive intervals; one seed does not estimate seed variability. No promotion or sealed scoring is dispatched.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--treatment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--diagnostic-common-support', action='store_true')
    args = parser.parse_args()
    result = compare(json.loads(args.reference.read_text()), json.loads(args.treatment.read_text()), args.diagnostic_common_support)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False); f.write('\n')


if __name__ == '__main__':
    main()
