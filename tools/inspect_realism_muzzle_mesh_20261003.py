"""Static topology/weight diagnostic, not image editing."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1];out=root/'output/pymousestudio_realism_cycle_20261002/v048_muzzle_audit';m=json.loads((out/'audit.json').read_text());z=np.load(out/'mesh.npz');v=z['rest'];w=z['weights'];names=z['group_names'].tolist();jaw=w[:,names.index('Jaw_M')];head=w[:,names.index('Head1_M')]+w[:,names.index('HeadEnd_M')]+jaw;mask=head>.2
adj=[set() for _ in v]
for f in m['faces']:
    for a,b in zip(f,f[1:]+f[:1]):adj[a].add(b);adj[b].add(a)
seen=set();components=[]
for i in range(len(v)):
    if i in seen:continue
    todo=[i];seen.add(i);members=[]
    while todo:
        a=todo.pop();members.append(a)
        for b in adj[a]:
            if b not in seen:seen.add(b);todo.append(b)
    q=v[members];components.append(dict(count=len(members),ids=members,lo=q.min(0).tolist(),hi=q.max(0).tolist(),head_count=int(mask[members].sum()),jaw_count=int((jaw[members]>.01).sum())))
summary=dict(components=components,group_bounds={n:dict(count=int((w[:,names.index(n)]>.01).sum()),lo=v[w[:,names.index(n)]>.01].min(0).tolist(),hi=v[w[:,names.index(n)]>.01].max(0).tolist()) for n in m['groups']})
(out/'topology.json').write_text(json.dumps(summary,indent=2));print(json.dumps({**summary,'components':[{k:val for k,val in c.items() if k!='ids'} for c in components]},indent=2))
fig,axes=plt.subplots(1,3,figsize=(18,7));pairs=[(2,1),(0,2),(0,1)]
for ax,(x,y) in zip(axes,pairs):
    ax.scatter(v[mask,x],v[mask,y],c=jaw[mask],s=5,cmap='viridis',vmin=0,vmax=1)
    for n,b in m['bones'].items():
        a=np.array(b['head_rest']);ax.plot(a[x],a[y],'rx');ax.text(a[x],a[y],n,fontsize=7)
    ax.set_aspect('equal');ax.set_xlabel('XYZ'[x]+' rest local');ax.set_ylabel('XYZ'[y]+' rest local');ax.grid(alpha=.2)
fig.suptitle('Head vertices, colored by Jaw_M skinning weight (yellow = 1)');fig.tight_layout();fig.savefig(out/'head_weights.png',dpi=140)
