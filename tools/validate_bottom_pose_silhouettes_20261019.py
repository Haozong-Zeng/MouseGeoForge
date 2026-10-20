"""Finite CPU-only geometry rerun with seeded subsets and immutable case records.

Uses cached release predictions; does not load neural weights, render, or score
sealed rigs. Default subsets rebuild the rotation prior from selected frames.
"""
import argparse
import json
import time
from pathlib import Path
import bottom_pose_silhouette_data_20261019 as data
import bottom_pose_frontier_20261019 as frontier
import bottom_pose_from_hull_20261019 as hull

def verify_repeat(output_dir):
    import numpy as np
    case=output_dir/'frontier_real_random24_seed20261019_selected.json'
    old=json.loads(case.read_text())['rows'][0]
    new=json.loads(json.dumps(frontier.run_package(data.real_package(24,'random',20261019))))
    for key in ('frame_ids','prior_frame_ids'):
        if old['metadata'][key]!=new['metadata'][key]:raise ValueError(f'Repeat changed {key}')
    differences={mode:max(float(np.max(np.abs(np.asarray(old[mode]['solver'][key])-np.asarray(new[mode]['solver'][key])))) for key in ('rotation','center')) for mode in ('full','azimuth_fixed')}
    if max(differences.values())>1e-8:raise ValueError(differences)
    receipt=dict(passed=True,seed=20261019,frames=24,frame_ids=old['metadata']['frame_ids'],pose_max_abs_differences=differences,neural_updates=0,source_sha256=data.envelope('',[])['source_sha256'])
    data.write_new(output_dir/'deterministic_repeat.json',receipt)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--part',choices=['frontier','hull'],required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    a=ap.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    plan=dict(part=a.part,neural_updates=0,synthetic_role='unseen_rig_gate only; no sealed scoring',random_seeds=list(range(20261019,20261027)),random_counts=[6,12,24],random_prior_scope='selected',modes=['full','azimuth_fixed'],case_plan_version=2)
    data.write_new(a.output_dir/f'{a.part}_plan.json',data.envelope(__doc__,[plan]))
    results=[]
    def run(name,package,scale):
        started=time.monotonic()
        row=frontier.run_package(package) if a.part=='frontier' else hull.run_package(package,scale)
        row.update(case=name,elapsed_s=time.monotonic()-started)
        evidence=data.envelope(__doc__,[row]);data.write_new(a.output_dir/f'{a.part}_{name}.json',evidence)
        results.append(dict(case=name,path=f'{a.part}_{name}.json',sha256=data.sha(a.output_dir/f'{a.part}_{name}.json'),elapsed_s=row['elapsed_s']))
        print('COMPLETE',a.part,name,row['elapsed_s'],flush=True)
    for package in data.synthetic_packages():run('synthetic_'+package['rig'],package,.5)
    if a.part=='frontier':
        run('real_all71_selected',data.real_package(None),.25)
        run('real_uniform36_selected',data.real_package(36),.25)
        run('real_legacy36_allprior',data.real_package(36,'legacy',prior_scope='all'),.25)
        for n in plan['random_counts']:
            for seed in plan['random_seeds']:run(f'real_random{n}_seed{seed}_selected',data.real_package(n,'random',seed),.25)
        for session in ('AZR2L2','FR2','FR3'):run('real_session_'+session+'_selected',data.real_package(None,session=session),.25)
    else:
        run('real_legacy36_allprior',data.real_package(36,'legacy',prior_scope='all'),.25)
        run('real_uniform36_selected',data.real_package(36),.25)
    if a.part=='frontier':verify_repeat(a.output_dir)
    data.write_new(a.output_dir/f'{a.part}_completed.json',dict(plan=plan,cases=results,neural_updates=0))

if __name__=='__main__':main()
