"""Bottom-camera pose from epipolar tangency of silhouettes (frontier points).

For each frame and each side camera s with known K_s, R_s, C_s, and a candidate
bottom pose (R_b, C_b): epipolar lines tangent to the mouse silhouette in one
image must be tangent to the silhouette in the other image (cf. Sinha &
Pollefeys, camera network calibration from dynamic silhouettes). Residual =
distance of the other image's tangent point to the transferred tangent line,
both directions, both tangents. No visual hull is used, so the residual is not
biased by the hull being a superset of the mouse. All frames of a fixed rig are
fitted jointly with robust least squares.

Initialisation: rotation from the cached mouse-only VGGT selected-frame rotation composed
with the reference side rotation; centre from the hull-centroid rays (as in
bottom_pose_from_hull_20261019). Side poses: GT / provided calibration (stage 1).
Masks: rendered (synthetic) or human-confirmed (real): oracle inputs.
"""
import argparse
import collections
import json
import math
import sys
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bottom_pose_aggregation_20261019 as agg
import bottom_pose_from_hull_20261019 as hullmod
import bottom_pose_silhouette_data_20261019 as data


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def fundamental(Ka, Ra, Ca, Kb, Rb, Cb):
    """F with x_b^T F x_a = 0 (a -> line in b)."""
    R = Rb @ Ra.T; t = Rb @ (Ca - Cb)
    return np.linalg.inv(Kb).T @ skew(t) @ R @ np.linalg.inv(Ka)


def outline(mask):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = np.concatenate([c.reshape(-1, 2) for c in cs]).astype(float)
    return cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2).astype(float)


def tangents(H, e):
    """Two convex-hull points with extreme angle as seen from epipole e (homogeneous)."""
    if not np.isfinite(e).all() or np.linalg.norm(e[:2]) < 1e-14:
        return None
    if abs(e[2]) < 1e-9:  # epipole at infinity: direction
        d = e[:2] / np.linalg.norm(e[:2]); nrm = np.array([-d[1], d[0]])
        s = H @ nrm
        return H[s.argmin()], H[s.argmax()]
    p = e[:2] / e[2]
    if cv2.pointPolygonTest(H.astype(np.float32).reshape(-1, 1, 2), (float(p[0]), float(p[1])), False) >= 0:
        return None
    v = H - p; c = v.mean(0); c /= np.linalg.norm(c)
    ang = np.arctan2(v @ np.array([-c[1], c[0]]), v @ c)
    return H[ang.argmin()], H[ang.argmax()]


def line_dist(l, x):
    norm=math.hypot(l[0],l[1])
    return (l[0]*x[0]+l[1]*x[1]+l[2])/norm if norm>1e-14 else float("inf")


def pair_residuals(Ha, Ka, Ra, Ca, Hb, Kb, Rb, Cb):
    F = fundamental(Ka, Ra, Ca, Kb, Rb, Cb)            # a -> b
    if not np.isfinite(F).all() or np.linalg.norm(F) < 1e-14:
        return None
    ea = np.linalg.svd(F)[2][-1]                       # epipole in a (F ea = 0)
    eb = np.linalg.svd(F.T)[2][-1]                     # epipole in b
    ta = tangents(Ha, ea); tb = tangents(Hb, eb)
    if ta is None or tb is None:
        return None
    la = [F @ np.array([*x, 1.]) for x in ta]          # lines in b
    lb = [F.T @ np.array([*x, 1.]) for x in tb]        # lines in a
    best = None
    for perm in ((0, 1), (1, 0)):
        r = [line_dist(la[i], tb[perm[i]]) for i in range(2)] + [line_dist(lb[perm[i]], ta[i]) for i in range(2)]
        if best is None or sum(x * x for x in r) < sum(x * x for x in best):
            best = r
    return best if np.isfinite(best).all() else None


def fit(obs, R0, C0, up_world, mode, return_details=False):
    """Fixed support: invalid tangencies never become zero-cost observations.

    Reject invalid initial states. Penalize invalid trial pairs above the image
    distance bound and reject invalid final states (retain best valid iterate).
    No support is dropped; successful fits require all registered pairs valid.
    """
    if mode not in ('full','azimuth_fixed'):raise ValueError(mode)
    n = R0 @ up_world; n /= np.linalg.norm(n)
    e1 = np.cross(n,[1.,0,0]) if abs(n[0])<.9 else np.cross(n,[0,1.,0]);e1/=np.linalg.norm(e1);e2=np.cross(n,e1)
    nr=3 if mode=='full' else 2
    scale=np.linalg.norm(C0)+1.
    pairs=[(Hs,Ks,Rs,Cs,Hb,Kb) for Hb,Kb,sides in obs for Hs,Ks,Rs,Cs in sides]
    if not pairs:raise ValueError('No registered constraints')
    # Any valid point-to-line distance is bounded by the image diameter when
    # that line intersects its silhouette; use a deliberately larger penalty.
    penalties=[10.*(np.linalg.norm(np.ptp(Hs,axis=0))+np.linalg.norm(np.ptp(Hb,axis=0))+1.) for Hs,Ks,Rs,Cs,Hb,Kb in pairs]
    def unpack(x):
        w=x[0]*e1+x[1]*e2+(x[2]*n if nr==3 else 0)
        return Rotation.from_rotvec(w).as_matrix()@R0,C0+x[nr:nr+3]*scale/100.
    def evaluate(x):
        R,C=unpack(x);out=[];invalid=0
        for pair,penalty in zip(pairs,penalties):
            r=pair_residuals(*pair[:4],pair[4],pair[5],R,C)
            if r is None:invalid+=1;r=[penalty]*4
            out.extend(r)
        return np.asarray(out),invalid
    def cost(r):return float(25.*np.sum(np.sqrt(1.+(r/5.)**2)-1.))
    x0=np.zeros(nr+3);initial,invalid=evaluate(x0)
    if invalid:raise ValueError(f'Invalid initial tangencies: {invalid}/{len(pairs)}; supply a valid initialization')
    best=[cost(initial),x0.copy()]; invalid_trials=0
    def res(x):
        nonlocal invalid_trials
        r,invalid=evaluate(x)
        if invalid:invalid_trials+=1
        elif cost(r)<best[0]:best[:]=[cost(r),x.copy()]
        return r
    sol=least_squares(res,x0,loss='soft_l1',f_scale=5.,x_scale=[.05]*nr+[1.]*3,max_nfev=3000)
    final,invalid_final=evaluate(sol.x)
    rejected=bool(invalid_final)
    selected=best[1] if rejected else sol.x
    final,invalid_selected=evaluate(selected);R,C=unpack(selected)
    details=dict(success=bool(sol.success and not rejected),status=int(sol.status),message=str(sol.message),nfev=int(sol.nfev),njev=None if sol.njev is None else int(sol.njev),optimality=float(sol.optimality),robust_cost=cost(final),initial_robust_cost=cost(initial),registered_pairs=len(pairs),invalid_trial_evaluations=invalid_trials,invalid_optimizer_final_pairs=invalid_final,invalid_final_pairs=invalid_selected,rejected_invalid_final=rejected,jacobian_singular_values=np.linalg.svd(sol.jac,compute_uv=False).tolist(),rotation=R.tolist(),center=C.tolist(),initial_rotation=R0.tolist(),initial_center=C0.tolist(),f_scale_px=5.,max_nfev=3000,parameters=nr+3)
    result=(R,C,float(np.median(np.abs(initial))),float(np.median(np.abs(final))))
    return (*result,details) if return_details else result


def run_package(package, modes=('full','azimuth_fixed')):
    obs=[];hframes=[]
    for f in package['frames']:
        b=f['bottom'];sides=f['sides']
        obs.append((outline(b['mask']),b['K'],[(outline(s['mask']),s['K'],s['R'],s['C']) for s in sides]))
        frame=hullmod.prepared_frame(f,1.)
        if frame is not None:hframes.append(frame)
    C0=hullmod.init_center(hframes,package['R0'])
    row=dict(rig=package['rig'],frames=len(obs),prior=list(hullmod.tilt_azimuth(package['R0'],package['Rgt'],package['up'])),metadata=package['metadata'])
    for mode in modes:
        try:
            R,C,r0,r1,details=fit(obs,package['R0'],C0,package['up'],mode,True)
            row[mode]=dict(error=list(hullmod.tilt_azimuth(R,package['Rgt'],package['up'])),median_residual_px_before=r0,after=r1,center_error=float(np.linalg.norm(C-package['Cgt'])),solver=details)
        except ValueError as exc:
            row[mode]=dict(error=None,solver=dict(success=False,message=str(exc)))
    residuals=[pair_residuals(*s,Hb,Kb,package['Rgt'],package['Cgt']) for Hb,Kb,sides in obs for s in sides]
    valid=[x for r in residuals if r is not None for x in r]
    row['gt_diagnostic']=dict(invalid_pairs=sum(r is None for r in residuals),registered_pairs=len(residuals),median_residual_px=float(np.median(np.abs(valid))) if valid else None,note='GT used only after fitting; median residual is not the robust optimizer objective')
    print(package['rig'],len(obs),{m:row[m]['error'] for m in modes},flush=True)
    return row


def synthetic():return [run_package(p) for p in data.synthetic_packages()]

def real(max_frames=36,selection='uniform',seed=20261019,session=None,prior_scope='selected'):
    return [run_package(data.real_package(max_frames,selection,seed,session,prior_scope))]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',choices=['synthetic','real'],required=True)
    ap.add_argument('--max-frames',type=int,default=36)
    ap.add_argument('--selection',choices=['uniform','random','legacy'],default='uniform')
    ap.add_argument('--seed',type=int,default=20261019)
    ap.add_argument('--session')
    ap.add_argument('--prior-scope',choices=['selected','all'],default='selected')
    ap.add_argument('--output',type=Path,required=True,help='New file only; preserve historical v1 results')
    a=ap.parse_args()
    rows=synthetic() if a.data=='synthetic' else real(a.max_frames,a.selection,a.seed,a.session,a.prior_scope)
    data.write_new(a.output,data.envelope(__doc__,rows))
    print('WROTE',a.output)

if __name__=='__main__':main()
