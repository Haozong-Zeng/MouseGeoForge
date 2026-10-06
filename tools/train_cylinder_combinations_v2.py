"""Sampling-only control against completed physical-view V1; no production edit."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import train_cylinder_view_balance as parent
from run_depth_pose_ablation import ROOT, fingerprint, write_json
from mousegeoforge.cylinder_combinations import sample_orders, coverage
from audit_bottom_view_coverage import cameras

PARENT = ROOT/'output/cylinder_view_balance_v1'
OUTPUT = ROOT/'output/cylinder_combinations_v2'
ARM = 'combination_coverage'


def config():
    cfg = json.loads((PARENT/'protocol.json').read_text())['config']
    cfg.update(arms=[ARM], arm_learning_rates={ARM: cfg['arm_learning_rates']['balanced_views']},
        sampling_policy='Per-pose pole alternation; exhaustive directed side/pole pair decks; stratified pole-containing V3/V4 decks; independent reference ordering',
        trainable_scope='Same as V1 balanced: shared adapter, dense depth and instance pose heads; frozen encoder')
    return cfg


def build_protocol(cfg):
    old = json.loads((PARENT/'protocol.json').read_text())
    if not json.loads((PARENT/'delivery_verification.json').read_text())['passed']:
        raise ValueError('Parent verification required')
    if parent.build_protocol(old['config']) != old:
        raise ValueError('Parent data/code/weights changed')
    if cfg != config(): raise ValueError('Unexpected experiment configuration')
    spec = copy.deepcopy(old); spec['config'] = cfg
    sequence = [old['records'][x[0]] for x in old['arms']['balanced_views']]
    indices = [i for i,s in enumerate(old['slots']) if s['kind'] == 'cylinder']
    role_maps = [{r['view_id']: r['orientation'] for r in cameras(ROOT/scene)}
                 for scene in sorted({sequence[i]['scene'] for i in indices})]
    if any(r != role_maps[0] for r in role_maps): raise ValueError('Inconsistent Cylinder rigs')
    roles = role_maps[0]
    slots = [(sequence[i]['scene'], len(sequence[i]['views'])) for i in indices]
    orders = sample_orders(slots, roles, cfg['seed'])
    audit = coverage(slots, orders, roles)
    if not audit['passed']: raise ValueError('Insufficient coverage: '+str(audit['gates']))
    spec['arms'] = {ARM: copy.deepcopy(old['arms']['balanced_views'])}
    for i,order in zip(indices,orders):
        record = dict(sequence[i], views=order, diagnostic_view_order=order,
                      physical_roles=[roles[v] for v in order])
        key = hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest()
        spec['records'][key] = record; spec['arms'][ARM][i] = [key]
    spec['sampling_coverage'] = audit
    spec['previous_sampling_coverage'] = coverage(slots,[sequence[i]['views'] for i in indices],roles)
    spec['scope'] = 'Synthetic same-rig development; camera subset/order sampling control, including changed physical-role frequencies; no real Cylinder validation'
    spec['parent_protocol_sha256'] = fingerprint(PARENT/'protocol.json')
    spec['parent_results_sha256'] = fingerprint(PARENT/'results.json')
    for path in ('tools/train_cylinder_combinations_v2.py','mousegeoforge/cylinder_combinations.py'):
        spec['code_sha256'][path] = fingerprint(ROOT/path)
    write_json(OUTPUT/'coverage.json',dict(new=audit,previous=spec['previous_sampling_coverage']))
    return spec


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only',action='store_true'); parser.add_argument('--launch',action='store_true')
    args=parser.parse_args()
    if args.launch:
        OUTPUT.mkdir(parents=True,exist_ok=True)
        with (OUTPUT/'run.log').open('a') as log:
            proc=subprocess.Popen([sys.executable,'-u',__file__],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(pid=proc.pid,log=str(OUTPUT/'run.log')))); return
    parent.engine.build_protocol=build_protocol
    parent.engine.load_batch=parent.oriented_batch
    parent.engine.evaluate_scene=parent.physical_metrics
    parent.engine.run(OUTPUT,config(),args.prepare_only)


if __name__=='__main__': main()
