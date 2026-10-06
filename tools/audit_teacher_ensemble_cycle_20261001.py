"""Paired quality/coverage diagnostic; agreement is not calibrated correctness."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from audit_teacher_dlc_predictions_20261001 import stats
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/teacher_temporal_cycle_20261001'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',nargs='+',type=int,default=[17,31]);args=ap.parse_args()
    m=json.loads((OUT/'dlc_dataset_manifest.json').read_text());ps=[]
    for seed in args.seeds:
        with np.load(OUT/f'lp_seed{seed}_kl/predictions_source_pixels.npz') as z:ps.append(dict(z))
    assert all(np.allclose(p['gt'],ps[0]['gt'],equal_nan=True) for p in ps)
    gt=ps[0]['gt'];stack=np.stack([p['uv'] for p in ps]);mean=stack.mean(0);median=np.median(stack,axis=0)
    # Radial population SD in original pixels; no division by sqrt(number of teachers).
    spread=np.sqrt(np.square(stack-mean).sum(-1).mean(0))
    conf=np.stack([p['confidence'] for p in ps]).mean(0)
    err=np.linalg.norm(mean-gt,axis=-1);errmed=np.linalg.norm(median-gt,axis=-1)
    ind=np.array(m['validation_indices']);valid=np.isfinite(err[ind]);e=err[ind][valid];d=spread[ind][valid];c=conf[ind][valid]
    ids=np.broadcast_to(np.array(m['ids_1based']),err[ind].shape)[valid]
    report=dict(seeds=args.seeds,units='original1080x1200 pixels',validation_n=len(e),
        single_models={str(seed):stats(p['error_px'][ind].ravel()) for seed,p in zip(args.seeds,ps)},
        mean_ensemble=stats(e),median_ensemble=stats(errmed[ind][valid]),
        spread_error_spearman=float(spearmanr(d,e).statistic),confidence_error_spearman=float(spearmanr(c,e).statistic),
        spread_thresholds={},matched_coverage={},train_allowed=False,
        warning='Development-only selection diagnostic, not calibrated uncertainty or final holdout evaluation. Two-seed mean can fall between conflicting modes. Agreement cannot rule out shared systematic error.')
    for threshold in [2,5,10,20,40]:
        keep=d<=threshold
        report['spread_thresholds'][str(threshold)]=dict(coverage=float(keep.mean()),errors=stats(e[keep]),
            over50px_fraction=float((e[keep]>50).mean()) if keep.any() else None,
            keypoint_coverage={str(k):float(keep[ids==k].mean()) for k in m['ids_1based']})
    for fraction in [.5,.75,.9]:
        n=int(len(e)*fraction);keep_d=np.argsort(d,kind='stable')[:n];keep_c=np.argsort(-c,kind='stable')[:n]
        report['matched_coverage'][str(fraction)]=dict(low_spread=stats(e[keep_d]),high_confidence=stats(e[keep_c]))
    # Export low-disagreement/high-error cases for human inspection, never auto-correct labels.
    rows=[]
    for i in ind:
        entry=m['entries'][int(i)]
        for j,k in enumerate(m['ids_1based']):
            if np.isfinite(err[i,j]) and spread[i,j]<=5 and err[i,j]>50:
                rows.append(dict(session=entry['session'],frame=entry['frame'],camera=entry['camera'],id=k,
                    error_px=float(err[i,j]),spread_px=float(spread[i,j]),confidence=float(conf[i,j])))
    name='lp_ensemble_'+'_'.join(map(str,args.seeds));dest=OUT/name;dest.mkdir(exist_ok=False)
    (dest/'summary.json').write_text(json.dumps(report,indent=2));(dest/'shared_failure_cases.json').write_text(json.dumps(rows,indent=2))
    np.savez_compressed(dest/'predictions_source_pixels.npz',uv=mean,uv_median=median,confidence=conf,gt=gt,error_px=err,spread_px=spread,ids=m['ids_1based'],train_allowed=False)
    print(json.dumps(report,indent=2));print('SHARED_FAILURES',len(rows))

if __name__=='__main__':main()
