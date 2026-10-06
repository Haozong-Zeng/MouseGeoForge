"""Conservative partial grouping; no labels/counts or forced assignments.

Pairwise mutual nearest plus a frozen absolute geometry cutoff precedes maximal
clique grouping. Overlapping plausible groups are all rejected. This does not
establish identity under every failure mode and does not alter any pose.
"""
import numpy as np


def conservative_camera_groups(nodes,distances,cutoff,*,enabled=False,min_views=3,max_cliques=4096):
    n=len(nodes)
    empty=dict(groups=[],edges=[],unassigned=list(range(n)),ambiguous_groups=[],overflow=False)
    if not enabled:return empty
    d=np.asarray(distances,float)
    if d.shape!=(n,n) or not np.isfinite(cutoff) or cutoff<0 or min_views<3:raise ValueError('Invalid geometry threshold or shape')
    if not np.allclose(d,d.T,equal_nan=True,atol=1e-7,rtol=0):raise ValueError('Expected symmetric distances')
    keys=[(x['view'],x['query']) for x in nodes]
    if len(set(keys))!=n:raise ValueError('Duplicate physical camera/query node')
    best={};views=sorted({x['view'] for x in nodes})
    for i,node in enumerate(nodes):
        for view in views:
            if view==node['view']:continue
            candidates=[j for j,x in enumerate(nodes) if x['view']==view and np.isfinite(d[i,j]) and d[i,j]>=0]
            if not candidates:continue
            minimum=min(d[i,j] for j in candidates)
            winners=[j for j in candidates if abs(d[i,j]-minimum)<=1e-9]
            if len(winners)==1 and minimum<cutoff:best[(i,view)]=winners[0]
    edges=[];neighbors={i:set() for i in range(n)}
    for (a,view),b in best.items():
        if a<b and best.get((b,nodes[a]['view']))==a:
            edges.append((a,b));neighbors[a].add(b);neighbors[b].add(a)
    cliques=[];overflow=False
    def visit(r,p,x):
        nonlocal overflow
        if overflow:return
        if not p and not x:
            if len(r)>=min_views:
                cliques.append(tuple(sorted(r)))
                if len(cliques)>max_cliques:overflow=True
            return
        union=p|x;pivot=max(sorted(union),key=lambda z:len(p&neighbors[z])) if union else None
        for v in sorted(p-(neighbors[pivot] if pivot is not None else set())):
            visit(r|{v},p&neighbors[v],x&neighbors[v]);p.remove(v);x.add(v)
    visit(set(),set(range(n)),set())
    if overflow:return dict(empty,edges=sorted(edges),overflow=True)
    cliques=sorted(set(cliques));ambiguous={i for i,a in enumerate(cliques) for j,b in enumerate(cliques) if i!=j and set(a)&set(b)}
    groups=[list(g) for i,g in enumerate(cliques) if i not in ambiguous]
    assert all(len({nodes[j]['view'] for j in g})==len(g) for g in groups)
    assigned={j for g in groups for j in g}
    return dict(groups=groups,edges=sorted(edges),unassigned=[j for j in range(n) if j not in assigned],
                ambiguous_groups=[list(g) for i,g in enumerate(cliques) if i in ambiguous],overflow=False)
