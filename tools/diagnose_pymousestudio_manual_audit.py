"""Read-only structural clarification, directional residual and paw parity checks.

No relabeling or calibration adjustment. Relative left/right parity has a global
sign ambiguity; lower epipolar cost is a review hypothesis, not anatomical truth.
"""
import itertools,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from audit_pymousestudio_manual_multiview import fundamental,epi

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'dataset/PyMouseStudio/annotation_returns/20260930_full_v2'
OUT=ROOT/'output/pymousestudio_manual_multiview_20260930'

def main():
    ingest=json.loads((BASE/'ingestion_manifest.json').read_text());duplicates=[];boundary=[];bias=defaultdict(list);parity=[]
    for issue in ingest['issues']:
        p=BASE/Path(issue['image']).with_suffix('.json');a=json.loads(p.read_text());bylabel=defaultdict(list)
        for s in a['shapes']:
            if s['shape_type']=='point':bylabel[s['label']].append(s['points'])
            if s['shape_type']=='polygon':
                xy=np.asarray(s['points']);outside=(xy<0).any() or (xy[:,0]>=1080).any() or (xy[:,1]>=1200).any()
                if outside:
                    within=bool((xy>=-1e-6).all() and (xy[:,0]<=1080+1e-6).all() and (xy[:,1]<=1200+1e-6).all())
                    boundary.append(dict(image=issue['image'],classification='polygon_canvas_boundary_or_roundoff' if within else 'genuine_outside_canvas',
                        min_xy=xy.min(0).tolist(),max_xy=xy.max(0).tolist(),requires_manual_relabel=not within))
        for label,values in bylabel.items():
            if len(values)>1:duplicates.append(dict(image=issue['image'],label=label,coordinates=values,normalized_observation_excluded=True))
    for g in ingest['frame_groups']:
        p=BASE/'geometry_diagnostics_NOT_GT'/g['split']/g['session']/f"frame_{g['frame']:06d}.npz"
        with np.load(p) as z:a=dict(z)
        if g['split']=='real_development':
            delta=a['uv_leave_one_view_out']-a['uv_observed']
            for i,c in enumerate(g['cameras']):
                for k in [0,7,14,15,16,17]:
                    if np.isfinite(delta[i,k]).all():bias[(g['session'],c)].append(delta[i,k])
        uv=a['uv_observed'];K=a['K'];E=a['world_to_camera_mm']
        for name,ks in [('forepaws',[14,15]),('hindpaws',[16,17])]:
            ids=[i for i in range(len(uv)) if np.isfinite(uv[i,ks]).all()]
            if len(ids)<3:continue
            costs={}
            for ii,jj in itertools.combinations(range(len(ids)),2):
                i,j=ids[ii],ids[jj];f=fundamental(K[i],E[i],K[j],E[j])
                same=sum(min(epi(uv[i,ks[k]],uv[j,ks[k]],f),100.) for k in range(2))
                flip=sum(min(epi(uv[i,ks[k]],uv[j,ks[1-k]],f),100.) for k in range(2))
                costs[(ii,jj)]=(same,flip)
            baseline=sum(c[0] for c in costs.values());best=(baseline,[0]*len(ids))
            for signs in itertools.product([0,1],repeat=len(ids)-1):
                s=[0]+list(signs);value=sum(c[s[i]^s[j]] for (i,j),c in costs.items())
                if value<best[0]:best=(value,s)
            if any(best[1]):
                parity.append(dict(split=g['split'],session=g['session'],frame=g['frame'],pair=name,
                    reference_camera=g['cameras'][ids[0]],relative_flip_cameras=[g['cameras'][i] for i,s in zip(ids,best[1]) if s],
                    cameras=[g['cameras'][i] for i in ids],baseline_mean_epipolar_px=baseline/(len(costs)*2),
                    hypothesis_mean_epipolar_px=best[0]/(len(costs)*2),relative_cost_reduction=(baseline-best[0])/max(baseline,1e-9),
                    warning='Relative parity only, global left/right ambiguity; NEVER applied to labels.'))
    bias_rows=[dict(session=s,camera=c,n=len(v),median_prediction_minus_annotation_xy=np.median(v,axis=0).tolist()) for (s,c),v in bias.items()]
    parity.sort(key=lambda r:(r['baseline_mean_epipolar_px']-r['hypothesis_mean_epipolar_px']),reverse=True)
    result=dict(duplicate_points=duplicates,polygon_boundary_classification=boundary,
        development_core_camera_bias=bias_rows,paw_relative_parity_hypotheses=parity,
        unchanged_originals=True,calibration_unchanged=True,labels_unchanged=True)
    (OUT/'structural_and_bias_diagnostics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print('DUPLICATES',len(duplicates),'BOUNDARY_ONLY',sum(not b['requires_manual_relabel'] for b in boundary))
    for r in parity[:12]:print(json.dumps(r))

if __name__=='__main__':main()
