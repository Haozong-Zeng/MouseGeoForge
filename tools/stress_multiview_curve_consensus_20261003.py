"""Synthetic TRAIN-camera-only consensus; compare longer-solver control."""
import datetime as dt
import numpy as np
from scipy.spatial.distance import cdist
import stress_multiview_surface_curve_20261003 as base


def camera_errors(curve, observations, cameras):
    # Along-curve identity is intentionally unknown. Endpoints remain separately
    # anchored in the fit. This score never receives the held-out camera.
    return np.array([np.median(cdist(obs[1:-1],base.project(curve,camera)).min(axis=1))
                     for obs,camera in zip(observations,cameras)])


def consensus(observations,cameras):
    n=len(cameras);candidates=[]
    for omitted in range(n):
        assert dt.datetime.now(dt.timezone.utc)<base.DEADLINE
        used=[i for i in range(n) if i!=omitted]
        curve,solver=base.fit(observations[used],cameras[used],True,max_nfev=150)
        errors=camera_errors(curve,observations,cameras)
        inliers=errors<6.
        # Six-pixel threshold fixed for sigma1px synthetic stress, not real QA.
        score=(-int(inliers.sum()),float(np.minimum(errors,6.).sum()))
        candidates.append((score,used,inliers,errors,solver))
    winner=min(candidates,key=lambda x:x[0]);inliers=np.flatnonzero(winner[2])
    if len(inliers)<3:inliers=np.arange(n)
    curve,solver=base.fit(observations[inliers],cameras[inliers],True,max_nfev=500)
    solver.update(selected_training_camera_indices=inliers.tolist(),
                  candidate_camera_residual_px=winner[3].tolist(),
                  candidate_solver_successes=sum(c[-1]['success'] for c in candidates),
                  threshold_px=6.,held_out_camera_used=False,
                  candidate_nfev_total=sum(c[-1]['nfev'] for c in candidates))
    return curve,solver


if __name__=='__main__':
    base.OUT=base.ROOT/'output/surface_curve_cycle_20261003/multiview_curve_consensus_v1'
    base.FITTERS={'latent_500eval':lambda obs,cams:base.fit(obs,cams,True,max_nfev=500),
                  'consensus_500eval':consensus}
    base.EXTRA_PROTOCOL=dict(candidate_count=6,candidate_max_nfev=150,final_max_nfev=500,
        threshold_px=6.,minimum_inlier_cameras=3,
        selection='Max TRAIN-camera inlier count, then clipped TRAIN-camera median point-to-curve distance. Held camera and full meshGT NEVER used.',
        limitations='Consensus uses additional solver work, not equal compute. Threshold is synthetic sigma1 stress only; no real acceptance rule.')
    try:base.main()
    except Exception:
        base.OUT.mkdir(exist_ok=True);base.write_json(base.OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()));raise
