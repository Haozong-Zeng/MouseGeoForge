"""Review paired data scaling; all seed metrics, seed17 visual columns."""
import review_diverse_curve_features_20261003 as review
import json
import numpy as np
review.BASE = review.ROOT/'output/surface_curve_cycle_20261003/curve_data_scaling_v1'
review.INPUTS = review.ROOT/'output/surface_curve_cycle_20261003/expanded_curve_features_v1/frozen_inputs.pt'
review.DISPLAY_ARMS = ['small_seed17', 'expanded_seed17']
if __name__ == '__main__':
    review.main()
    results=json.loads((review.BASE/'results.json').read_text())['results']
    paired=[]
    for seed in [17,29,43]:
        small=next(r for r in results if r['seed']==seed and r['size']=='small')
        expanded=next(r for r in results if r['seed']==seed and r['size']=='expanded')
        paired.append(dict(seed=seed,**{metric:dict(small=small['validation'][metric]['mean'],
            expanded=expanded['validation'][metric]['mean'],
            expanded_minus_small=expanded['validation'][metric]['mean']-small['validation'][metric]['mean'])
            for metric in ['pred_to_gt_px','gt_to_pred_px','endpoints_px']}))
    summary=dict(paired=paired,summary={metric:dict(
        mean_delta=float(np.mean([p[metric]['expanded_minus_small'] for p in paired])),
        min_delta=float(min(p[metric]['expanded_minus_small'] for p in paired)),
        max_delta=float(max(p[metric]['expanded_minus_small'] for p in paired)),
        improved_seeds=sum(p[metric]['expanded_minus_small']<0 for p in paired))
        for metric in ['pred_to_gt_px','gt_to_pred_px','endpoints_px']},
        limitation='Three development seeds, not an inferential significance test. Same synthetic assets; no external real benchmark.')
    (review.BASE/'paired_summary.json').write_text(json.dumps(summary,indent=2))
    path=review.BASE/'visual_review/index.html'
    html=path.read_text().replace('<h1>Frozen-model curve feature comparison</h1>',
        '<h1>Data scaling: 18 versus 89 training poses</h1><p>Same model, objective,2400updates and frozen validation. Three paired seeds; displayed columns seed17. Train label means expanded-arm TRAIN membership; added frames were NOT trained by small arm. Both arms are predictions, not fitted ground truth.</p>')
    path.write_text(html)
    print(json.dumps(summary,indent=2))
