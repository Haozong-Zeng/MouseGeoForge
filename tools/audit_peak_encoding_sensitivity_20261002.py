"""Paired same-frame encoding sensitivity; displacement is not GT error."""
import json
import numpy as np
from run_teacher_cycle_dlc_20261001 import OUT
from audit_teacher_dlc_predictions_20261001 import stats

def main():
    old=json.loads((OUT/'temporal_paw_peaks_v1/manifest.json').read_text());new=json.loads((OUT/'temporal_paw_peaks_pngexact_v1/manifest.json').read_text());results=[]
    for r in new:
        other=next(o for o in old if all(o[k]==r[k] for k in ['seed','cohort','session']))
        with np.load(r['file']) as z:a=dict(z)
        with np.load(other['file']) as z:b=dict(z)
        assert np.array_equal(a['source_frames'],b['source_frames']) and np.array_equal(a['cameras'],b['cameras'])
        delta=np.linalg.norm(a['peaks_source_px'][...,0,:]-b['peaks_source_px'][...,0,:],axis=-1)
        results.append(dict(seed=r['seed'],cohort=r['cohort'],session=r['session'],n=delta.size,
            top1_displacement_px=stats(delta.ravel()),over15px=int((delta>15).sum()),over50px=int((delta>50).sum()),
            exact_sparse_peak_max_difference_px=max(x['peak_max_distance_px'] for x in r['sparse_peak_parities']),
            exact_sparse_ratio_max_difference=max(x['ratio_max_difference'] for x in r['sparse_peak_parities'])))
    report=dict(results=results,total_point_predictions=sum(r['n'] for r in results),top1_displacement_over50px=sum(r['over50px'] for r in results),
        warning='Same pose/frame with extraH264CRF16encoding vs raw-remapPNG-exactinput. Top1displacement measures sensitivity, NOT whichpredictioniscorrect. Points correlated acrossmodels/views/frames; not independentdata.',train_allowed=False)
    (OUT/'temporal_paw_peaks_pngexact_v1/encoding_sensitivity.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
