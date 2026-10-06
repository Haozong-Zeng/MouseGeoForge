"""Paired surface-curve integral readout review; synthetic material GT only."""
import json
import numpy as np
import torch
import review_diverse_curve_features_20261003 as review
from review_sparse_real_curve_adaptation_20261003 import crossings
review.BASE=review.ROOT/'output/surface_curve_cycle_20261003/curve_heatmap_supervision_v1'
review.INPUTS=review.ROOT/'output/surface_curve_cycle_20261003/expanded_curve_features_v1/frozen_inputs.pt'
review.DISPLAY_ARMS=['base_seed17','coordinate_seed17','heatmap_seed17']


if __name__=='__main__':
    review.main();rows=json.loads((review.BASE/'results.json').read_text())['results'];pairs=[]
    for seed in [17,29,43]:
        g=next(r for r in rows if r['seed']==seed and r['mode']=='coordinate')
        s=next(r for r in rows if r['seed']==seed and r['mode']=='heatmap')
        pairs.append(dict(seed=seed,**{k:dict(coordinate_only=g['validation'][k]['mean'],heatmap=s['validation'][k]['mean'],
            heatmap_minus_coordinate=s['validation'][k]['mean']-g['validation'][k]['mean']) for k in ['pred_to_gt_px','gt_to_pred_px','endpoints_px']}))
    saved=torch.load(review.INPUTS,map_location='cpu',weights_only=False);meta=saved['metadata'];shapes=[]
    with np.load(review.BASE/'predictions.npz') as z:
        for arm in z.files:
            for role in ['train','validation']:
                ids=[i for i,m in enumerate(meta) if m['role']==role];v=[crossings(z[arm][i]) for i in ids]
                shapes.append(dict(arm=arm,role=role,n=len(ids),curves_with_self_crossing=sum(x>0 for x in v),total_crossings=sum(v)))
    payload=dict(pairs=pairs,shape_diagnostics=shapes,no_production_promotion=True,limitations='Fixedsyntheticdevelopmentvalidation; sameassets, no externalrealtest. Frozenbase onlydescriptive: fewerheadupdates. Projection selfcrossingscanbevalid; visuallyinspect. Sixpoints areknownmaterialsamples onlyforBlendersurface, NOT fractionsassignedtoreal06/07.')
    (review.BASE/'paired_summary.json').write_text(json.dumps(payload,indent=2));print(json.dumps(payload,indent=2))
    path=review.BASE/'visual_review/index.html';html=path.read_text().replace('Equal2400updates.','Equal1600additionalupdates from the same per-seed frozenbase.')
    html=html.replace('<h1>Frozen-model curve feature comparison</h1>',
        '<h1>Known-material heatmap supervision</h1><p>Base=frozen6controlprediction. Global/heatmap=sixon-curveheatmapexpectations, interpolatedwithcubicbasis. Sameparameters/init/minibatches/updatebudget, differingonlyimagefeature spatialvariation. No realcurveGT or realfractionassignment.</p>');path.write_text(html)
