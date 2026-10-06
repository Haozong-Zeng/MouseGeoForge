"""Offline finite-state identity diagnostic, not interpolating3D coordinates."""
import itertools
import numpy as np

SETTINGS=dict(hypotheses_per_paw=3,min_distinct_paw_mm=5,
    emission='sum(support*log(10)+heatmap_geometry_score); unknown contributes0',
    motion_cost='0.5*min(distance_mm/5,4) per observed paw transition; unknown/observed transition0.5; both unknown0',
    motion_weight=.5,motion_scale_mm=5,motion_cap=4,unknown_switch_cost=.5,
    ambiguity_score_margin=float(np.log(3)),ambiguity_distance_mm=5,
    state_search='all valid combinations of <=3hypotheses+unknown for each of4paws; <=256states, no beampruning',
    inference='offline fullclip forward/backward max-sum; not causal',
    no_interpolation=True,train_allowed=False,
    warning='Heuristic scores, not calibrated probabilities. Strong temporal consistency can preserve wrongidentity. No dense3Dtruth.')

def states(hs):
    xx=[];scores=[]
    for combo in itertools.product(*([[None]+h[:3] for h in hs])):
        observed=[h for h in combo if h is not None]
        if any(np.linalg.norm(np.array(a['xyz_mm'])-b['xyz_mm'])<5 for a,b in itertools.combinations(observed,2)):continue
        x=np.full((4,3),np.nan)
        for j,h in enumerate(combo):
            if h is not None:x[j]=h['xyz_mm']
        xx.append(x);scores.append(sum(h['support']*np.log(10)+h['score'] for h in observed))
    return np.array(xx),np.array(scores)

def transition(a,b,weight=.5):
    va=np.isfinite(a).all(-1);vb=np.isfinite(b).all(-1)
    both=va[:,None]&vb[None];switch=va[:,None]^vb[None]
    d=np.linalg.norm(np.nan_to_num(a)[:,None]-np.nan_to_num(b)[None],axis=-1)
    return (np.where(both,weight*np.minimum(d/5,4),0)+switch*(weight)).sum(-1)

def decode(xs,es,weight=.5):
    # weight0 is an exact same-state NO-temporal ablation, including unknownswitch0.
    forward=[es[0].copy()];backpointer=[None]
    for t in range(1,len(xs)):
        value=forward[-1][:,None]-transition(xs[t-1],xs[t],weight)
        best=value.argmax(0);forward.append(es[t]+value[best,np.arange(len(es[t]))]);backpointer.append(best)
    path=[int(forward[-1].argmax())]
    for t in range(len(xs)-1,0,-1):path.append(int(backpointer[t][path[-1]]))
    path=path[::-1];backward=[None]*len(xs);backward[-1]=np.zeros_like(es[-1])
    for t in range(len(xs)-2,-1,-1):
        backward[t]=(es[t+1][None]+backward[t+1][None]-transition(xs[t],xs[t+1],weight)).max(1)
    bestscore=float(forward[-1].max());chosen=np.array([x[i] for x,i in zip(xs,path)]);accepted=chosen.copy();ambiguous=np.zeros((len(xs),4),bool)
    plausible_counts=[]
    for t in range(len(xs)):
        scores=forward[t]+backward[t];plausible=xs[t][bestscore-scores<np.log(3)];plausible_counts.append(len(plausible))
        for j in range(4):
            if not np.isfinite(chosen[t,j]).all():continue
            if np.any(~np.isfinite(plausible[:,j]).all(-1)) or np.any(np.linalg.norm(plausible[:,j]-chosen[t,j],axis=-1)>5):
                accepted[t,j]=np.nan;ambiguous[t,j]=True
    return dict(xyz=accepted,chosen_xyz=chosen,ambiguous=ambiguous,plausible_counts=np.array(plausible_counts),bestscore=bestscore,path=np.array(path))

def selftest():
    from joint_paw_assignment_cycle_20261002 import assign
    def h(x,score=0):return dict(xyz_mm=[x,0,100],support=4,score=score)
    sets=[[[h(t*2)],[h(30+t*2)],[h(60+t*2)],[h(90+t*2)]] for t in range(5)]
    pairs=[states(hs) for hs in sets];xs=[p[0] for p in pairs];es=[p[1] for p in pairs]
    zero=decode(xs,es,0)['xyz'];expected=np.array([assign(hs)[0] for hs in sets]);assert np.allclose(zero,expected,equal_nan=True)
    assert np.allclose(decode(xs,es,.5)['xyz'],expected,equal_nan=True)
    empty=states([[],[],[],[]]);gap=decode(xs[:2]+[empty[0]]+xs[3:],es[:2]+[empty[1]]+es[3:],.5)['xyz'];assert np.isnan(gap[2]).all()
    fast=[states([[h(t*20)],[h(50+t*20)],[h(100+t*20)],[h(150+t*20)]]) for t in range(4)]
    got=decode([p[0] for p in fast],[p[1] for p in fast],.5)['xyz'];assert np.allclose(np.diff(got[:,:,0],axis=0),20)
    # Exact enumeration control on a tiny ambiguous path checks forward/backward recursion.
    tx=[np.array([[[0.,0,0]]*4,[[10.,0,0]]*4]) for _ in range(3)];te=[np.array([1.,0.]),np.array([0.,1.]),np.array([1.,0.])]
    score=max(sum(te[t][p[t]] for t in range(3))-sum(transition(tx[t-1],tx[t])[p[t-1],p[t]] for t in range(1,3)) for p in itertools.product(range(2),repeat=3))
    assert abs(decode(tx,te)['bestscore']-score)<1e-9
    return dict(zero_weight_matches_independent=True,missing_frame_not_filled=True,fast_single_hypothesis_not_lagged=True,exhaustive_tiny_path_matches=True)
