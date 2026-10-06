"""Strong TRAIN-only two-parameter calibration reference; rank is preserved."""
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import expit
from train_point_quality_20261005 import eval_sets
from run_depth_pose_ablation import fingerprint,write_json
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';OUT=CYCLE/'point_quality_simple_baselines_v1'


def main():
    assert not OUT.exists();OUT.mkdir();started=time.monotonic();torch.set_num_threads(2)
    source=CYCLE/'point_quality_inputs_v1/inputs.pt';data=torch.load(source,map_location='cpu',weights_only=False);meta=data['metadata'];weights=torch.zeros_like(data['target'])
    for kind,fraction in [('real',.5),('single',.25),('dual',.25)]:
        ids=[i for i,m in enumerate(meta) if m['stage']=='fit' and m['kind']==kind and data['valid'][i].any()]
        for i in ids:weights[i]=data['valid'][i]*fraction/len(ids)/int(data['valid'][i].sum())
    mask=weights>0;w=weights[mask].numpy().astype(float);w/=w.sum();x=data['scalars'][...,1][mask].numpy().astype(float);y=data['target'][mask].numpy().astype(float)
    prior=float(w@y);a0=float(np.log(prior/(1-prior)))
    protocol=dict(target='P(error<=32px)',fit='Exact same expected8/4/4 cohort/image/knownpoint objective as neural sampling, deterministic weighted BCE',
        predictors=['balanced_fit_constant','monotonic_std'],formula='sigmoid(a-b*log1p(pixel_std)/log769), b>=0. Rank exactly follows inverse_std; NOT visibility.',
        parameter_bounds=[[-20,20],[0,100]],maxiter=500,source_sha256=fingerprint(source),scope='TRAIN fit only, no heldcal/VAL objective or tuning. No pose changes. Baselines added after first pilot failure, not pristine preregistered comparison.')
    write_json(OUT/'protocol.json',protocol)
    def objective(theta):
        a,b=theta;z=a-b*x;p=expit(z);loss=float(w@(np.logaddexp(0,z)-y*z));grad=np.array([w@(p-y),-(w*((p-y)*x)).sum()]);return loss,grad
    fit=minimize(objective,[a0,1.],jac=True,method='L-BFGS-B',bounds=protocol['parameter_bounds'],options=dict(maxiter=500,ftol=1e-12,gtol=1e-10))
    assert fit.success and np.isfinite(fit.x).all(),fit
    scores=dict(balanced_fit_constant=torch.full_like(data['target'],prior),monotonic_std=torch.from_numpy(expit(fit.x[0]-fit.x[1]*data['scalars'][...,1].numpy())).float())
    results={name:{stage:eval_sets(score,data,stage) for stage in ['fit','calibration','validation']} for name,score in scores.items()}
    # Strictly monotone calibration should preserve quality ranking (within float precision).
    reference=json.loads((CYCLE/'point_quality_pilot_v1/results.json').read_text())['results']['inverse_std_rank']
    max_auc_delta=max(abs(results['monotonic_std'][stage][kind]['auc']-reference[stage][kind]['auc']) for stage in results['monotonic_std'] for kind in results['monotonic_std'][stage])
    assert max_auc_delta<1e-4,max_auc_delta
    torch.save(scores,OUT/'scores.pt');write_json(OUT/'results.json',dict(results=results,fit=dict(constant=prior,intercept=float(fit.x[0]),negative_std_slope=float(fit.x[1]),loss=float(fit.fun),iterations=int(fit.nit),message=str(fit.message)),max_auc_rank_delta=max_auc_delta,elapsed_seconds=time.monotonic()-started,script_sha256=fingerprint(Path(__file__)),production_changed=False))
    print(json.dumps({name:{kind:{k:m[k] for k in ['auc','brier','ece']} for kind,m in r['validation'].items()} for name,r in results.items()},indent=2))


if __name__=='__main__':main()

