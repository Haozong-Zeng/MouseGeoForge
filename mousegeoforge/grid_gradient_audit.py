"""Pure, deterministic selection and vector diagnostics for grid-gradient audits."""
import numpy as np


def select_records(spec):
    schedule=[spec['records'][x[0]] for x in spec['arms']['combination_coverage']]
    cylinder=sorted({r['scene'] for i,r in enumerate(schedule) if spec['slots'][i]['kind']=='cylinder'})
    dual=sorted({r['scene'] for r in schedule if r['expected_instances']==2})
    def four(items):
        if len(items)<4:raise ValueError('Need four independent source scenes')
        return [items[round(i*(len(items)-1)/3)] for i in range(4)]
    chosen=[]
    for scene in four(cylinder):
        candidates=[r for r in schedule if r['scene']==scene]
        for role in ('bottom','top'):
            chosen.append(next(r for r in candidates if len(r['views'])==1 and r['physical_roles']==[role]))
        for n in (2,3,4):chosen.append(next(r for r in candidates if len(r['views'])==n))
    for scene in four(dual):chosen.append(next(r for r in schedule if r['scene']==scene))
    if any(r['split']!='train' for r in chosen):raise ValueError('Only training records allowed')
    return chosen


def cosine(a,b):
    a,b=np.asarray(a,dtype=float).ravel(),np.asarray(b,dtype=float).ravel()
    den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a,b)/den) if den>1e-12 else None


def point_gradient(full,direct,residual):
    full,direct,residual=map(lambda x:np.asarray(x,dtype=float),(full,direct,residual))
    if not all(x.shape==(2,) and np.isfinite(x).all() for x in (full,direct,residual)):
        raise ValueError('Expected finite UV gradients and residual')
    indirect=full-direct
    f,d,i=cosine(-full,residual),cosine(-direct,residual),cosine(-indirect,residual)
    return dict(full_descent_to_gt_cos=f,direct_descent_to_gt_cos=d,indirect_descent_to_gt_cos=i,
        full_direct_cos=cosine(full,direct),indirect_direct_cos=cosine(indirect,direct),
        full_norm=float(np.linalg.norm(full)),direct_norm=float(np.linalg.norm(direct)),
        indirect_norm=float(np.linalg.norm(indirect)),
        harmful_reversal=f is not None and d is not None and f<0 and d>0,
        indirect_away=i is not None and i<0)
