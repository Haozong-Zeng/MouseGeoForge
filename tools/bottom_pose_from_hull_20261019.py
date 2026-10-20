"""Bottom-camera pose from side-view visual hulls and the bottom silhouette.

Stage 1 (this script): side cameras use the provided/GT K and E, so only the
bottom estimate is tested. For each fixed rig, all frames are fitted jointly
(the mouse moves between frames, which separates camera rotation from camera
translation). Objective per frame: symmetric chamfer between the projected hull
points and the bottom mouse mask (distance transforms on a downsampled grid).

Bottom pose initialisation: rotation from the mouse-only VGGT all-frame
relative rotation composed with the reference side rotation (no GT bottom);
centre by linear least squares from hull-centroid / mask-centroid rays.
Variants: 'full' optimises 3 rotation + 3 centre parameters; 'azimuth_fixed'
keeps the rotation about the prior's world-up axis fixed (tilt/roll + centre).
Masks are rendered (synthetic) or human-confirmed (real): oracle inputs.
"""
import argparse
import collections
import json
import math
import sys
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bottom_pose_aggregation_20261019 as agg
import bottom_pose_silhouette_data_20261019 as data

ROOT = agg.ROOT
MANIFEST = ROOT / 'output/bottom_geometry_v3_local12h_20261018/manifest.json'
CAL = ROOT / 'dataset/PyMouseStudio/annotation_returns/20260930_full_v2/calibration/{}/rectified_calibration.json'
EFFECTIVE = ROOT / 'dataset/PyMouseStudio/annotation_versions/20261003_effective_v1/manifest.json'


def project(P, K, R, C):
    X = (P - C) @ R.T
    z = X[:, 2]
    uv = X[:, :2] / np.maximum(z, 1e-6)[:, None] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    return uv, z


def hull(sides, half=70., step=2.):
    """sides: list of (mask bool HxW, K, R, C). Returns hull points (N,3) or None."""
    # centre: least-squares intersection of rays through mask centroids
    A = np.zeros((3, 3)); b = np.zeros(3)
    for m, K, R, C in sides:
        ys, xs = np.nonzero(m)
        d = R.T @ np.linalg.solve(K, [xs.mean(), ys.mean(), 1.]); d /= np.linalg.norm(d)
        Pm = np.eye(3) - np.outer(d, d); A += Pm; b += Pm @ C
    center = np.linalg.solve(A, b)
    g = np.arange(-half, half + 1e-6, step)
    P = np.stack(np.meshgrid(g, g, g, indexing='ij'), -1).reshape(-1, 3) + center
    keep = np.ones(len(P), bool)
    for m, K, R, C in sides:
        mm = cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        uv, z = project(P, K, R, C)
        u = np.round(uv[:, 0]).astype(int); v = np.round(uv[:, 1]).astype(int)
        ok = (z > 0) & (u >= 0) & (v >= 0) & (u < m.shape[1]) & (v < m.shape[0])
        inside = np.zeros(len(P), bool); inside[ok] = mm[v[ok], u[ok]]
        keep &= inside
    return P[keep] if keep.sum() > 50 else None


class Frame:
    def __init__(self, hull_pts, mask, K, scale):
        self.P = hull_pts
        h, w = mask.shape
        self.s = scale; self.hw = (int(h * scale), int(w * scale))
        small = cv2.resize(mask.astype(np.uint8), (self.hw[1], self.hw[0]), interpolation=cv2.INTER_NEAREST)
        self.mask = small > 0
        self.dt_out = cv2.distanceTransform((~self.mask).astype(np.uint8), cv2.DIST_L2, 3)  # distance to mask
        ys, xs = np.nonzero(self.mask)
        self.mask_pts = (xs, ys)
        Ks = K.copy(); Ks[:2] *= scale
        self.K = Ks
        self.centroid = np.array([xs.mean(), ys.mean()])

    def cost(self, R, C):
        uv, z = project(self.P, self.K, R, C)
        if (z <= 0).mean() > .1:
            return 1e3
        h, w = self.hw
        u = np.clip(np.round(uv[:, 0]).astype(int), 0, w - 1); v = np.clip(np.round(uv[:, 1]).astype(int), 0, h - 1)
        outside = (uv[:, 0] < 0) | (uv[:, 1] < 0) | (uv[:, 0] >= w) | (uv[:, 1] >= h)
        t1 = np.where(outside, 50., self.dt_out[v, u]).mean()
        img = np.zeros((h, w), np.uint8); img[v[~outside], u[~outside]] = 1
        img = cv2.dilate(img, np.ones((3, 3), np.uint8))
        dt_proj = cv2.distanceTransform((1 - img).astype(np.uint8), cv2.DIST_L2, 3)
        t2 = dt_proj[self.mask_pts[1], self.mask_pts[0]].mean()
        return t1 + t2


def init_center(frames, R):
    """Given R, solve camera centre C so hull centroids project to mask centroids (linear)."""
    A = []; b = []
    for f in frames:
        X = f.P.mean(0)
        d = R.T @ np.linalg.solve(f.K, [*f.centroid, 1.]); d /= np.linalg.norm(d)
        Pm = np.eye(3) - np.outer(d, d)
        A.append(Pm); b.append(Pm @ X)
    return np.linalg.lstsq(np.concatenate(A), np.concatenate(b), rcond=None)[0]


def fit(frames, R0, up_world, mode, return_details=False):
    if not frames:raise ValueError("No usable hull frames")
    nr=3 if mode=="full" else 2
    C0 = init_center(frames, R0)
    n = R0 @ up_world; n /= np.linalg.norm(n)
    e1 = np.cross(n, [1., 0, 0]) if abs(n[0]) < .9 else np.cross(n, [0, 1., 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(n, e1)
    def unpack(x):
        w = x[0] * e1 + x[1] * e2 + (x[2] * n if mode == 'full' else 0)
        return Rotation.from_rotvec(w).as_matrix() @ R0, C0 + x[nr:nr+3]
    def total(x):
        R, C = unpack(x)
        return float(np.mean([f.cost(R, C) for f in frames]))
    x0 = np.zeros(nr+3)
    best = (total(x0), x0)
    attempts=[]
    for rad, mm in ((np.radians(8), 15.), (np.radians(3), 6.), (np.radians(1), 2.)):
        r = minimize(total, best[1], method='Powell', options=dict(xtol=1e-4, ftol=1e-5, maxiter=4000,
                     direc=np.diag([rad]*nr+[mm]*3)))
        attempts.append(dict(success=bool(r.success),status=int(r.status),message=str(r.message),nfev=int(r.nfev),cost=float(r.fun)))
        if r.fun < best[0]:
            best = (r.fun, r.x)
    R, C = unpack(best[1])
    result=(R,C,best[0],total(x0))
    details=dict(attempts=attempts,rotation=R.tolist(),center=C.tolist(),initial_rotation=R0.tolist(),initial_center=C0.tolist(),parameters=nr+3)
    return (*result,details) if return_details else result


def tilt_azimuth(R, Rgt, up_world):
    """Split the error into tilt (angle between estimated and true up in camera) and the rest."""
    u = R @ up_world; g = Rgt @ up_world
    tilt = math.degrees(math.acos(np.clip(u @ g / np.linalg.norm(u) / np.linalg.norm(g), -1, 1)))
    return agg.angle(R, Rgt), tilt


# Cache is input-object-local and only avoids repeating the same deterministic hull.
# Keep the frame object alive alongside the result so Python id reuse cannot alias.
_FRAME_CACHE={}
def prepared_frame(frame,scale):
    key=(id(frame),scale)
    if key not in _FRAME_CACHE:
        b=frame['bottom']
        H=hull([(s['mask'],s['K'],s['R'],s['C']) for s in frame['sides']])
        _FRAME_CACHE[key]=(frame,Frame(H,b['mask'],b['K'],scale) if H is not None else None)
    return _FRAME_CACHE[key][1]


def run_package(package,scale):
    frames=[prepared_frame(f,scale) for f in package['frames']]
    used=[f for f in frames if f is not None]
    row=dict(rig=package['rig'],frames=len(used),metadata=package['metadata'],excluded_hull_frame_ids=[f['frame_id'] for f,h in zip(package['frames'],frames) if h is None],prior_error=list(tilt_azimuth(package['R0'],package['Rgt'],package['up'])),mask_resize_scale=scale,pixel_resize_convention='historical K[:2]*scale preserved; no additional half-pixel change')
    for mode in ('full','azimuth_fixed'):
        R,C,c1,c0,details=fit(used,package['R0'],package['up'],mode,True)
        row[mode]=dict(error=list(tilt_azimuth(R,package['Rgt'],package['up'])),cost_before=c0,cost_after=c1,center_error=float(np.linalg.norm(C-package['Cgt'])),solver=details)
    print(package['rig'],len(used),{m:row[m]['error'] for m in ('full','azimuth_fixed')},flush=True)
    return row


def synthetic(scale=.5):return [run_package(p,scale) for p in data.synthetic_packages()]

def real(scale=.25,max_frames=36,selection='uniform',seed=20261019,session=None,prior_scope='selected'):
    return [run_package(data.real_package(max_frames,selection,seed,session,prior_scope),scale)]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',choices=['synthetic','real'],required=True)
    ap.add_argument('--max-frames',type=int,default=36)
    ap.add_argument('--selection',choices=['uniform','random','legacy'],default='uniform')
    ap.add_argument('--seed',type=int,default=20261019)
    ap.add_argument('--session')
    ap.add_argument('--prior-scope',choices=['selected','all'],default='selected')
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    rows=synthetic() if a.data=='synthetic' else real(max_frames=a.max_frames,selection=a.selection,seed=a.seed,session=a.session,prior_scope=a.prior_scope)
    data.write_new(a.output,data.envelope(__doc__,rows))
    print('WROTE',a.output)

if __name__=='__main__':main()
