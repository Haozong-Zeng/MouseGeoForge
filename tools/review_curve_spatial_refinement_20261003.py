"""Frozen validation local/global spatial refinement comparison."""
import json
import numpy as np
import review_diverse_curve_features_20261003 as review
review.BASE=review.ROOT/'output/surface_curve_cycle_20261003/curve_spatial_refinement_v1'
review.INPUTS=review.ROOT/'output/surface_curve_cycle_20261003/expanded_curve_features_v1/frozen_inputs.pt'
review.DISPLAY_ARMS=['base_seed17','global_seed17','local_seed17']
if __name__=='__main__':
    review.main()
    rows=json.loads((review.BASE/'results.json').read_text())['results'];pairs=[]
    for seed in [17,29,43]:
        g=next(r for r in rows if r['seed']==seed and r['mode']=='global')
        l=next(r for r in rows if r['seed']==seed and r['mode']=='local')
        pairs.append(dict(seed=seed,**{k:dict(global_pool=g['validation'][k]['mean'],local=l['validation'][k]['mean'],
            local_minus_global=l['validation'][k]['mean']-g['validation'][k]['mean']) for k in ['pred_to_gt_px','gt_to_pred_px','endpoints_px']}))
    (review.BASE/'paired_summary.json').write_text(json.dumps(dict(pairs=pairs,
        means={k:float(np.mean([p[k]['local_minus_global'] for p in pairs])) for k in ['pred_to_gt_px','gt_to_pred_px','endpoints_px']},
        no_production_promotion=True,limitation='Development validation, not independent real test.'),indent=2))
    path=review.BASE/'visual_review/index.html';html=path.read_text()
    html=html.replace('Equal2400updates.','Equal1600refinementupdates on the same per-seed frozen2400step base.')
    path.write_text(html.replace('<h1>Frozen-model curve feature comparison</h1>',
        '<h1>Prediction-only spatial refinement</h1><p>Base=frozen prior prediction; global/local=equal-parameter shared residual refiner. Local samples ON predicted curves, never GT or latent control-point positions. No production promotion.</p>'))
