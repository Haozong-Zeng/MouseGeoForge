"""Matched-camera cohorts for read-only training-pose/development diagnostics."""
GROUPS=('old_cylinder_side_pair','bottom_only','side_bottom','sides_bottom','sides_bottom_top','old_dual_pair')


def build_training_cohort(spec):
    sequence=[spec['records'][x[0]] for x in spec['arms']['combination_coverage']]
    pools={'cylinder':{},'dual':{}}
    for slot,r in zip(spec['slots'],sequence):
        if slot['kind'] in pools:pools[slot['kind']][r['scene']]=r
    prototypes={}
    for entry in spec['evaluation']:
        group=entry['group']
        if group not in GROUPS:continue
        r=spec['records'][entry['key']]
        if group in prototypes:
            for key in ('views','physical_roles','image_hw'):assert r[key]==prototypes[group][key]
        else:prototypes[group]=r
    if set(prototypes)!=set(GROUPS):raise ValueError('Incomplete camera conditions')
    out=[]
    for kind,pool in pools.items():
        for scene,r in sorted(pool.items()):
            if r['split']!='train':raise ValueError('Training cohort contains a nontrain scene')
            for group in (('old_dual_pair',) if kind=='dual' else GROUPS[:-1]):
                proto=prototypes[group]
                if r['image_hw']!=proto['image_hw']:raise ValueError('Train/development image shape differs')
                order=list(proto['views'])
                record=dict(r,views=order,diagnostic_view_order=order,physical_roles=list(proto['physical_roles']))
                visits=[x['views'] for x in sequence if x['scene']==scene]
                out.append(dict(record=record,group=group,
                    exact_order_visits=sum(v==order for v in visits),
                    unordered_subset_visits=sum(sorted(v)==sorted(order) for v in visits)))
    return out
