"""Synthetic-only leave-one-camera-out shared 3D surface curve diagnostic.

Artificial observations include known virtual endpoints, even in occluded cameras.
This is an identifiability stress test, NOT a real-data reconstruction benchmark.
No per-camera equal-arclength correspondence is used.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import datetime as dt
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
from PIL import Image, ImageDraw
from run_depth_pose_ablation import write_json, fingerprint

BASE = ROOT / 'dataset/BlenderPyMouse/PyMouseStudio-RealCapture-SurfaceCurve-Multiview'
OUT = ROOT / 'output/surface_curve_cycle_20261003/multiview_curve_stress_v1'
KNOTS = np.r_[np.zeros(4), [1/3, 2/3], np.ones(4)]
BASIS = BSpline(KNOTS, np.eye(6), 3)
U = np.linspace(0, 1, 65)
NOMINAL = np.linspace(0, 1, 5)
SECOND = np.diff(np.eye(6), n=2, axis=0)
DEADLINE = dt.datetime(2026, 10, 4, 8, 5, 16, tzinfo=dt.timezone.utc)


def project(x, camera):
    q = np.c_[x, np.ones(len(x))] @ camera.T
    return q[:, :2] / q[:, 2:3]


def dlt(observations, cameras):
    rows = []
    for (x, y), p in zip(observations, cameras):
        rows.extend([x*p[2]-p[0], y*p[2]-p[1]])
    _, _, vt = np.linalg.svd(rows)
    return vt[-1, :3] / vt[-1, 3]


def fit(observations, cameras, sliding, max_nfev=250):
    n = len(cameras)
    control_count=BASIS(NOMINAL).shape[1];control_size=3*control_count
    tri = np.array([dlt(observations[:, j], cameras) for j in range(5)])
    design = np.r_[BASIS(NOMINAL), .02*SECOND]
    controls = np.linalg.lstsq(design, np.r_[tri, np.zeros((len(SECOND), 3))], rcond=None)[0]
    initial = controls.ravel()
    if sliding:
        initial = np.r_[initial, np.tile(NOMINAL[1:-1], n)]
    lower = np.r_[np.full(control_size, -np.inf), np.tile([.05, .30, .55], n)] if sliding else -np.inf
    upper = np.r_[np.full(control_size, np.inf), np.tile([.45, .70, .95], n)] if sliding else np.inf

    def residual(theta):
        c = theta[:control_size].reshape(control_count, 3)
        t = np.c_[np.zeros(n), theta[control_size:].reshape(n, 3), np.ones(n)] if sliding else np.tile(NOMINAL, (n, 1))
        reproj = np.stack([project(BASIS(tt)@c, camera) for tt, camera in zip(t, cameras)]) - observations
        reproj[:, [0, -1]] *= 2  # Synthetic virtual endpoints: not manual05/08.
        parts = [reproj.ravel(), (.02*SECOND@c).ravel()]
        if sliding:
            parts.append((100*np.maximum(.03-np.diff(t, axis=1), 0)).ravel())
        return np.concatenate(parts)

    fitted = least_squares(residual, initial, bounds=(lower, upper), loss='soft_l1', f_scale=2., max_nfev=max_nfev)
    curve = BASIS(U) @ fitted.x[:control_size].reshape(control_count, 3)
    return curve, dict(success=bool(fitted.success), nfev=fitted.nfev, optimality=float(fitted.optimality),
                       robust_cost=float(fitted.cost), message=fitted.message)


def distances(a, b):
    # Dense-sample symmetric distance; same sampling used for both arms.
    d = cdist(a, b)
    return float(.5*(d.min(0).mean()+d.min(1).mean()))


FITTERS = {'fixed_nominal_parameter': lambda obs, cams: fit(obs, cams, False),
           'latent_ordered_parameter': lambda obs, cams: fit(obs, cams, True)}
EXTRA_PROTOCOL = {}


def main():
    assert not OUT.exists()
    OUT.mkdir()
    status = lambda stage, **kw: write_json(OUT/'status.json', dict(stage=stage, pid=os.getpid(), utc=dt.datetime.now(dt.timezone.utc).isoformat(), **kw))
    write_json(OUT/'protocol.json', dict(seed=71, poses=3, cameras=7, held_out_cameras='All7, separately',
        arms=list(FITTERS), observations_per_camera=5,
        cases=['clean', 'along_line_shift', 'along_shift_plus_corrupt_camera'],
        observation_scale='Native camera projection scaled by768/1200; noise sigma1px except clean',
        model=f'Shared3D{BASIS(NOMINAL).shape[1]}control clamped cubic B-spline; robust soft-L1; curvature0.02; virtual endpoints weight2',
        parameter_bounds=[[.05,.45],[.30,.70],[.55,.95]], order_min_gap=.03,
        initialization='DLT of noisy nominally associated points, then least-squares spline; no GT controls in initialization',
        important_limits=['Synthetic virtual endpoints assumed available; real labels lack these anchors',
                          'Artificial amodal observations include occluded cameras; not a visibility-aware real-data result',
                          'Held-out camera observations never enter fitting; scoring uses independent mesh3D/full projected curve',
                          'Not learned-model quality and no automatic real pseudo-label acceptance',
                          'No independent2D equal-arclength resampling or manual06/07fraction assignment'],
        hard_deadline=DEADLINE.isoformat(), extra_protocol=EXTRA_PROTOCOL))
    rows = []; visual_rows = []
    rng = np.random.default_rng(71)
    for scene in sorted(BASE.glob('frame_*')):
        with np.load(scene/'surface_curve.npz') as z:truth=z['world_xyz'][0].astype(float)
        with np.load(scene/'keypoints.npz') as z:K=z['intrinsics'].astype(float);E=z['extrinsics'].astype(float)
        cameras = K @ E[:, :3]
        cameras[:, :2] *= 768/1200
        hashes = {name:fingerprint(scene/name) for name in ['surface_curve.npz','keypoints.npz']}
        for case in ['clean','along_line_shift','along_shift_plus_corrupt_camera']:
            t = np.tile(NOMINAL,(7,1))
            if case != 'clean':t[:,1:-1] += rng.uniform(-.12,.12,(7,3))
            points = np.array([np.stack([np.interp(tt,U,truth[:,axis]) for axis in range(3)],axis=1) for tt in t])
            obs = np.stack([project(p,c) for p,c in zip(points,cameras)])
            if case != 'clean':obs += rng.normal(0,1,obs.shape)
            if case == 'along_shift_plus_corrupt_camera':obs[2,1:-1] += [25,-18]
            for held in range(7):
                assert dt.datetime.now(dt.timezone.utc)<DEADLINE
                used = [v for v in range(7) if v != held]
                curves = {}
                for arm, fitter in FITTERS.items():
                    curve, solver = fitter(obs[used],cameras[used])
                    held_pred = project(curve,cameras[held]); held_truth=project(truth,cameras[held])
                    # Full-curve geometry is independent from sparse observations used to fit.
                    row=dict(scene=scene.name,case=case,held_camera=held,arm=arm,
                             curve_3d_symmetric_mm=distances(curve,truth),
                             heldout_fullcurve_symmetric_px=distances(held_pred,held_truth),
                             heldout_endpoints_mean_px=float(np.linalg.norm(held_pred[[0,-1]]-held_truth[[0,-1]],axis=1).mean()),
                             all_cameras_positive_depth=bool(all((np.c_[curve,np.ones(len(curve))]@e[:3].T)[:,2].min()>0 for e in E)),
                             solver=solver,source_sha256=hashes)
                    rows.append(row);curves[arm]=held_pred
                if held==4 and case!='clean':
                    with Image.open(scene/'blended_images'/f'{held:08d}.jpg') as source:im=source.convert('RGB').resize((round(source.width*768/1200),768))
                    w,h=im.size;panel=Image.new('RGB',(3*w,h+26),(20,20,25));draw=ImageDraw.Draw(panel)
                    for col,arm in enumerate(['GT']+list(curves)):
                        panel.paste(im,(col*w,26));draw.text((col*w+5,5),arm,fill='white')
                        gt=project(truth,cameras[held])+[col*w,26];draw.line([tuple(p) for p in gt],fill='#ffdf45',width=2)
                        if arm!='GT':draw.line([tuple(p+[col*w,26]) for p in curves[arm]],fill='#32e5fa',width=2)
                    filename=f'{scene.name}_{case}.jpg';panel.save(OUT/filename,quality=94)
                    visual_rows.append(dict(file=filename,scene=scene.name,case=case,held=held))
                write_json(OUT/'partial_results.json',rows);status('fitting',scene=scene.name,case=case,held=held)
    summary=[]
    for case in sorted({r['case'] for r in rows}):
        for arm in sorted({r['arm'] for r in rows}):
            selected=[r for r in rows if r['case']==case and r['arm']==arm]
            summary.append(dict(case=case,arm=arm,n=len(selected),solver_success=sum(r['solver']['success'] for r in selected),
                **{k:dict(mean=float(np.mean([r[k] for r in selected])),median=float(np.median([r[k] for r in selected])),p90=float(np.quantile([r[k] for r in selected],.9))) for k in ['curve_3d_symmetric_mm','heldout_fullcurve_symmetric_px','heldout_endpoints_mean_px']}))
    write_json(OUT/'results.json',dict(summary=summary,rows=rows,visuals=visual_rows,inspection_pending=True,no_real_pseudolabels_created=True))
    html=['<!doctype html><meta charset="utf-8"><title>Synthetic shared3D curve stress</title><style>body{background:#202027;color:white;font:16px sans-serif}img{max-width:100%}</style><h1>Leave-one-camera-out synthetic diagnostic</h1><p>Yellow=mesh truth; cyan=shared3D curve projected to HELD OUT camera. Synthetic virtual endpoints and amodal observations assumed. This is NOT a real reconstruction or learned prediction result.</p>']
    for r in visual_rows:html.append(f'<h2>{r["scene"]}: {r["case"]}, held camera{r["held"]}</h2><img src="{r["file"]}">')
    (OUT/'index.html').write_text('\n'.join(html));status('complete')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(exist_ok=True);write_json(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()));raise
