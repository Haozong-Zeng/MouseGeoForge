"""Matched per-image context choices; labels and schedules are never mutated."""
import random

def single_context_flags(schedule, metadata, seed):
    rng=random.Random(seed);result=[]
    for step in schedule:
        if set(step)!= {'real','synthetic'} or any(len(step[k])!=8 for k in step):
            raise ValueError('Expected8real+8synthetic image schedule')
        for domain,ids in step.items():
            for i in ids:
                m=metadata[i];r=m['record']
                if r['role']!='train' or r.get('session') in ['GR2','UR2','UR3'] or not m['detection_complete']:
                    raise ValueError('Only eligible TRAIN images')
                if (r['kind']=='real')!=(domain=='real'):raise ValueError('Domain mismatch')
        views=[int(metadata[i]['view']) for i in step['real']]
        if not all(v==0 for v in views[:4]) or not all(v!=0 for v in views[4:]):
            raise ValueError('Expected4bottom then4other real images')
        real=set(rng.sample(range(4),2)+rng.sample(range(4,8),2));syn=set(rng.sample(range(8),4))
        result.append(dict(real=[i in real for i in range(8)],synthetic=[i in syn for i in range(8)]))
    return result

def random_reference_contexts(schedule, single_flags, metadata, seed):
    """Reverse half the paired inputs, preserving all single/context exposures."""
    if len(schedule)!=len(single_flags):raise ValueError('Mismatched schedules')
    rng=random.Random(seed);result=[]
    for images,flags in zip(schedule,single_flags):
        # Reuse eligibility/order validation without depending on its RNG.
        single_context_flags([images],metadata,0)
        if set(flags)!=set(images) or any(len(flags[k])!=8 for k in flags):raise ValueError('Invalid flags')
        context={k:['single' if f else 'paired' for f in flags[k]] for k in flags}
        for domain,positions in [('real',range(4)),('real',range(4,8)),('synthetic',range(8))]:
            candidates=[j for j in positions if not flags[domain][j]]
            expected=4 if domain=='synthetic' else 2
            if len(candidates)!=expected:raise ValueError('Expected balanced single and paired exposures')
            for j in rng.sample(candidates,len(candidates)//2):context[domain][j]='reversed'
        result.append(context)
    return result


def full_view_contexts(schedule, single_flags, metadata, seed):
    """Preserve single exposure; replace half paired occurrences with full context.

    'full' means all eligible cameras in the same scene. Some synthetic scenes
    have only2eligibleviews; their full/paired contexts are intentionally equal.
    """
    reference=random_reference_contexts(schedule,single_flags,metadata,seed)
    names={'single':'single','paired':'shared','reversed':'full'}
    return [{domain:[names[c] for c in values] for domain,values in step.items()} for step in reference]
