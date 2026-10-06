"""Experimental TRAIN sampler only; no camera-ID routing in model inference."""
import collections
import random


def balance_bottom_real_batches(schedule,metadata,eligible_real_ids,seed):
    pools={k:collections.defaultdict(list) for k in ['bottom','other']}
    for i in sorted(set(eligible_real_ids)):
        m=metadata[i];r=m['record']
        if r['kind']!='real' or r['role']!='train' or r.get('session') in ['GR2','UR2','UR3'] or not m['detection_complete']:
            raise ValueError('Only complete-mask eligible real TRAIN images may be sampled')
        pools['bottom' if int(m['view'])==0 else 'other'][r['session']].append(i)
    if any(not p for p in pools.values()):raise ValueError('Both view pools required')
    rng=random.Random(seed);result=[]
    for step in schedule:
        if len(step['real'])!=8 or len(step['synthetic'])!=8:raise ValueError('Expected8real+8synthetic schedule')
        selected=[rng.choice(pools[k][rng.choice(sorted(pools[k]))]) for k in ['bottom','other'] for _ in range(4)]
        result.append(dict(real=selected,synthetic=list(step['synthetic'])))
    return result
