"""Verify CPU rerun receipts; print summary or explicitly write compact evidence."""
import argparse
import copy
import json
import shutil
from pathlib import Path
import numpy as np
import bottom_pose_silhouette_data_20261019 as data
ROOT=data.ROOT
COMPACT=ROOT/'docs/experiments/bottom_pose_from_silhouettes_20261019.json'

def stats(rows,mode='full'):
    valid=[r[mode]['error'][0] for r in rows if r[mode]['error'] is not None and r[mode]['solver'].get('success',True)]
    return dict(n=len(rows),valid=len(valid),failed=len(rows)-len(valid),median_deg=float(np.median(valid)) if valid else None,max_deg=float(max(valid)) if valid else None,lt5=sum(v<5 for v in valid)/len(rows),lt15=sum(v<15 for v in valid)/len(rows))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input-dir',type=Path,required=True)
    ap.add_argument('--write-compact',action='store_true')
    a=ap.parse_args();a.input_dir=a.input_dir.resolve();allrows={};hashes={};sources={};runtime=None
    for method in ('frontier','hull'):
        completion=json.loads((a.input_dir/f'{method}_completed.json').read_text())
        for item in completion['cases']:
            p=a.input_dir/item['path']
            if data.sha(p)!=item['sha256']:raise ValueError(f'Changed case: {p}')
            case=json.loads(p.read_text());row=case['rows'][0]
            for name,digest in case['source_sha256'].items():
                if digest!=data.sha(ROOT/'tools'/name):raise ValueError(f'Source changed since validation: {name}')
            for path,digest in row['metadata']['input_sha256'].items():
                if path in hashes and hashes[path]!=digest:raise ValueError(f'Input changed: {path}')
                # The historical prediction alias has this unique resolution.
                actual=ROOT/path if path!='release_bg_gray.json' else data.agg.RUN/path
                if data.sha(actual)!=digest:raise ValueError(f'Input changed on disk: {path}')
                hashes[path]=digest
            row=copy.deepcopy(row);del row['metadata']['input_sha256']
            row['artifact_path']=str(p.relative_to(ROOT));row['artifact_sha256']=item['sha256']
            allrows[f'{method}/{item["case"]}']=row;sources.update(case['source_sha256']);runtime=case['runtime']
    for name in ('validate_bottom_pose_silhouettes_20261019.py','summarize_bottom_pose_silhouettes_20261019.py','../tests/test_bottom_pose_silhouettes_20261019.py'):sources[name]=data.sha(ROOT/'tools'/name)
    synth={m:stats([v for k,v in allrows.items() if k.startswith(m+'/synthetic_')]) for m in ('frontier','hull')}
    synthfixed={m:stats([v for k,v in allrows.items() if k.startswith(m+'/synthetic_')],'azimuth_fixed') for m in ('frontier','hull')}
    random={str(n):stats([v for k,v in allrows.items() if k.startswith(f'frontier/real_random{n}_')]) for n in (6,12,24)}
    invalid=sum(v[m]['solver'].get('invalid_final_pairs',0) for k,v in allrows.items() if k.startswith('frontier/') for m in ('full','azimuth_fixed'))
    failures=sum(not v[m]['solver'].get('success',False) for k,v in allrows.items() if k.startswith('frontier/') for m in ('full','azimuth_fixed'))
    historic=ROOT/'output/bottom_geometry_v3_local12h_20261018/eval_v4_2_20261019'
    original=json.loads((historic/'bottom_from_hull_synthetic_v1.json').read_text())['rows']
    oldhashes={n:data.sha(historic/n) for n in ('bottom_from_hull_synthetic_v1.json','bottom_frontier_synthetic_v1.json','bottom_frontier_real_v1.json','bottom_from_hull_real_v1.json')}
    repeat=json.loads((a.input_dir/'deterministic_repeat.json').read_text())
    if not repeat['passed'] or max(repeat['pose_max_abs_differences'].values())>1e-8:raise ValueError('Repeat verification failed')
    if repeat['frame_ids']!=allrows['frontier/real_random24_seed20261019_selected']['metadata']['frame_ids']:raise ValueError('Repeat support differs')
    for name,digest in repeat['source_sha256'].items():
        if digest!=sources[name]:raise ValueError('Repeat source differs')
    result=dict(schema_version=2,review_status='CPU-only classical geometry; no neural inference/training or sealed scoring',side_poses='GT/provided calibration',mask_source='render GT / human-confirmed masks',coordinate_rule='Each synthetic frame transformed into its reference side-camera frame before pooling',random_protocol=dict(seeds=list(range(20261019,20261027)),counts=[6,12,24],selection='random.Random(seed).sample without replacement; sorted indices',initialization='only selected frame cached release predictions',unit='one physical real rig; overlapping subsets are not independent rigs'),summary=dict(synthetic=synth,synthetic_fixed=synthfixed,random_selected_prior=random,frontier_invalid_final_pairs=invalid,frontier_failed_solver_endpoints=failures,total_case_files=len(allrows)),deterministic_repeat=repeat,historical_v1=dict(files=oldhashes,hull_fixed_median_deg=float(np.median([r['azimuth_fixed']['error'][0] for r in original])),note='Raw v1 results preserved; synthetic pooled-world results superseded. Unarchived random v1 results withdrawn, not reconstructed.'),runtime=runtime,source_sha256=sources,input_sha256=hashes,prediction_alias_path=str((data.agg.RUN/'release_bg_gray.json').relative_to(ROOT)),cases=allrows)
    if a.write_compact:
        history=a.input_dir/'historical_documents';history.mkdir(exist_ok=True)
        for p in (ROOT/'docs/bottom_pose_from_silhouettes_20261019.md',COMPACT):
            dest=history/p.name
            if not dest.exists():shutil.copyfile(p,dest)
        header={k:v for k,v in result.items() if k!='cases'}
        cases=',\n'.join('    '+json.dumps(k)+': '+json.dumps(v,separators=(',',':'),allow_nan=False) for k,v in allrows.items())
        COMPACT.write_text(json.dumps(header,indent=2,allow_nan=False)[:-2]+',\n  \"cases\": {\n'+cases+'\n  }\n}\n')
    print(json.dumps(result['summary'],indent=2))

if __name__=='__main__':main()
