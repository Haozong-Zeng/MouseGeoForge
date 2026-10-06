"""CPU-only observed-loss trends, controlling input composition within windows."""
import argparse
from collections import defaultdict
import json
import numpy as np
from run_depth_pose_ablation import ROOT,write_json,fingerprint


def summarize(path):
    history=json.loads(path.read_text());assert len(history)==800
    rows=[]
    for start,middle,end in ((1,400,800),(401,600,800)):
        groups=defaultdict(lambda:[[],[]])
        for h in history:
            if not start<=h['step']<=end:continue
            key=(h['scene'],tuple(h['views']),tuple(h['image_hw']),h['slot_kind'])
            groups[key][int(h['step']>middle)].append(h)
        for kind in sorted({k[-1] for k in groups}):
            common=[v for k,v in groups.items() if k[-1]==kind and all(v)]
            row=dict(start=start,middle=middle,end=end,kind=kind,common_input_keys=len(common),
                earlier_observations=sum(len(v[0]) for v in common),later_observations=sum(len(v[1]) for v in common))
            for metric in ('pose_loss','depth_loss'):
                a=[float(np.mean([r[metric] for r in v[0]])) for v in common]
                b=[float(np.mean([r[metric] for r in v[1]])) for v in common]
                row[metric]=dict(earlier=float(np.mean(a)),later=float(np.mean(b)),ratio=float(np.mean(b)/np.mean(a)),
                    improved_inputs=sum(y<x for x,y in zip(a,b))) if common else None
            rows.append(row)
    return dict(source_sha256=fingerprint(path),rows=rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',choices=('17','29'),required=True);args=parser.parse_args()
    if args.seed=='17':
        out=ROOT/'output/cylinder_adaptive_temperature_v1'
        paths=dict(off=ROOT/'output/cylinder_source_mix_v1/source_mix/history.json',on=out/'adaptive_temperature/history.json')
    else:
        out=ROOT/'output/cylinder_temperature_seed29_v1';paths={a:out/f'temperature_{a}/history.json' for a in ('off','on')}
    report=dict(seed=args.seed,arms={a:summarize(p) for a,p in paths.items()},
        interpretation='Same-input-key means reduce composition confounding; these are observed training-mode losses across different model states, not a fixed-checkpoint loss probe or proof of development convergence.',
        weighting='Each shared (scene, ordered views, HW, slot kind) has equal weight inside its source kind.',optimizer_steps=0)
    write_json(out/'matched_training_trends.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
