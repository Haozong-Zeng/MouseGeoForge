"""Read-only grouped inventory for a new localization-quality stage."""
import collections,json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];CYCLE=ROOT/'output/singleview_fusion_cycle_20261005'
for role in ['train','validation']:
    data=torch.load(CYCLE/'singleview_fusion_inputs_seed17_v1'/f'{role}.pt',map_location='cpu',weights_only=False)
    counts=collections.Counter()
    for m in data['metadata']:
        r=m['record'];kind='real' if r['kind']=='real' else 'dual' if m['expected_instances']>1 else 'single'
        counts[kind,r.get('source_group'),r.get('session')]+=1
    print(role,json.dumps([dict(kind=k[0],source_group=k[1],session=k[2],images=v) for k,v in sorted(counts.items(),key=str)],indent=2))
    print('example metadata',json.dumps(data['metadata'][0]));print('keys',list(data));print('shapes',{k:tuple(v.shape) for k,v in data.items() if isinstance(v,torch.Tensor)})
    missing=next((m for m in data['metadata'] if m['record']['kind']!='real' and not m['record'].get('source_group')),None)
    if missing:print('missing source_group example',json.dumps(missing))
