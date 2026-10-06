import collections,json
from pathlib import Path
import numpy as np
from audit_review_return2_20261003 import points,OUT,FOCUS,write
r=json.loads((OUT/'summary.json').read_text());old=next((OUT/'sent_package').iterdir());new=Path(r['extracted']);hist=collections.Counter();moves=[];nonpoints=[];missing=[]
for row in r['files']:
    path=Path(row['annotation']);a=json.loads((old/path.relative_to(new)).read_text());b=json.loads(path.read_text());pa,pb=points(a),points(b)
    valid=[k for k in [5,6,7,8] if len(pb.get(k,[]))==1];hist[len(valid)]+=1
    if len(valid)<3:missing.append(dict(group=row['group'],camera=row['camera'],present=valid))
    shapes=lambda ann:[s for s in ann.get('shapes',[]) if s.get('shape_type')!='point']
    if shapes(a)!=shapes(b):nonpoints.append(dict(group=row['group'],camera=row['camera']))
    for d in row['changes']:
        k=d['id']
        if k not in [6,7] or len(d['before'])!=1 or len(d['after'])!=1:continue
        if not all(len(pa.get(q,[]))==1 for q in [k-1,k+1]):continue
        t=np.array(pa[k+1][0])-pa[k-1][0];norm=np.linalg.norm(t)
        if norm<1e-6:continue
        t/=norm;delta=np.array(d['after'][0])-d['before'][0];along=float(abs(delta@t));across=float(abs(delta@np.array([-t[1],t[0]])))
        moves.append(dict(group=row['group'],camera=row['camera'],id=k,along_old_neighbor_chord_px=along,across_old_neighbor_chord_px=across,total=float(np.linalg.norm(delta))))
scope=[dict(group=row['group'],camera=row['camera'],ids=[d['id'] for d in row['changes'] if d['id'] not in FOCUS]) for row in r['files'] if any(d['id'] not in FOCUS for d in row['changes'])]
result=dict(available_05_06_07_08_point_counts=dict(hist),bracketed_back_changes=moves,nonpoint_shape_changed_images=nonpoints,out_of_focus_edits=scope,matched_valid_3plus_loo_failures=[v for v in r['loo_observations'] if v['before'] is None or v['after'] is None],caveat='Chord decomposition is diagnostic only, not a known anatomical tangent or validated curve. Curve fitting was not used to correct labels.')
if moves:result['movement_summary']=dict(n=len(moves),median_along_px=float(np.median([v['along_old_neighbor_chord_px'] for v in moves])),median_across_px=float(np.median([v['across_old_neighbor_chord_px'] for v in moves])),along_greater_than_across=sum(v['along_old_neighbor_chord_px']>v['across_old_neighbor_chord_px'] for v in moves))
write(OUT/'curve_evidence.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['bracketed_back_changes','out_of_focus_edits','matched_valid_3plus_loo_failures']},ensure_ascii=False,indent=2));print('outside_focus_images',len(scope))
