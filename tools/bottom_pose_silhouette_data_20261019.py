"""Shared, auditable oracle-mask geometry inputs; no neural inference or training."""
import collections
import functools
import hashlib
import json
import random
from pathlib import Path
import cv2
import numpy as np
import bottom_pose_aggregation_20261019 as agg
ROOT = agg.ROOT
MANIFEST = ROOT / 'output/bottom_geometry_v3_local12h_20261018/manifest.json'
EFFECTIVE = ROOT / 'dataset/PyMouseStudio/annotation_versions/20261003_effective_v1/manifest.json'
CAL = ROOT / 'dataset/PyMouseStudio/annotation_returns/20260930_full_v2/calibration/{}/rectified_calibration.json'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def pose(E):
    E = np.asarray(E, dtype=float)
    return E[:3, :3], -E[:3, :3].T @ E[:3, 3]

def canonical_pose(R, C, Rref, Cref):
    """X_ref = Rref @ (X_world-Cref), preserving all image projections."""
    return R @ Rref.T, Rref @ (C-Cref)

def read_mask(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or not image.any():
        raise ValueError(f'Missing/empty mask: {path}')
    return image > 0

@functools.lru_cache(None)
def synthetic_packages():
    recs = collections.defaultdict(list)
    for r in json.loads(MANIFEST.read_text())['records']:
        if r.get('evaluation_role') == 'unseen_rig_gate':
            recs[r['rig']].append(r)
    gray = agg.pairs(agg.RUN / 'release_bg_gray.json')
    packages = []
    for rig, records in sorted(recs.items()):
        candidates = [p for p in gray if p['rig'] == rig and p['role'] == 'unseen_rig_gate' and p['pair_role'] == 'bottom']
        key = min(p['key'] for p in candidates)
        R0 = agg.geodesic_median([p['R'] for p in candidates if p['key'] == key])
        frames = []; truths = []; hashes = {}; ups = []
        for rec in records:
            views = []
            for v in rec['views']:
                path = ROOT / v['truth']; hashes[str(path.relative_to(ROOT))] = sha(path)
                with np.load(path) as a: mask = a['body'].copy()
                R, C = pose(v['E'])
                views.append(dict(id=v['id'], role=v['physical_role'], mask=mask, K=np.asarray(v['K'], float), R=R, C=C))
            ref = next(v for v in views if v['id'] == key[2])
            ups.append(ref['R'] @ np.array([0., 0., 1.]))
            Rref, Cref = ref['R'].copy(), ref['C'].copy()
            for v in views:
                v['R'], v['C'] = canonical_pose(v['R'], v['C'], Rref, Cref)
            bottom = next(v for v in views if v['role'] == 'bottom')
            truths.append((bottom['R'], bottom['C']))
            frames.append(dict(frame_id=dict(group=rec.get('group'), frame=rec.get('frame'), truth=[v['truth'] for v in rec['views']]), bottom=bottom, sides=[v for v in views if v['role'] != 'bottom']))
        Rgt, Cgt = truths[0]
        center_span = max(float(np.linalg.norm(C-Cgt)) for R,C in truths)
        rotation_span = max(agg.angle(R, Rgt) for R,C in truths)
        if center_span > 1e-3 or rotation_span > .1 or max(np.linalg.norm(u-ups[0]) for u in ups)>1e-4:
            raise ValueError(f'Rig is not fixed after canonicalization: {rig}')
        packages.append(dict(rig=rig, frames=frames, R0=R0, Rgt=Rgt, Cgt=Cgt, up=ups[0], metadata=dict(coordinate_frame=f'reference side camera {key[2]}', reference_view=key[2], frame_ids=[f['frame_id'] for f in frames], prior_scope='selected six frames', canonical_center_span=center_span, canonical_rotation_span_deg=rotation_span, input_sha256={str(MANIFEST.relative_to(ROOT)):sha(MANIFEST), 'release_bg_gray.json':sha(agg.RUN/'release_bg_gray.json'), **hashes})))
    return packages

@functools.lru_cache(None)
def real_inputs():
    em = json.loads(EFFECTIVE.read_text())
    prov = {(v['session'], v['frame'], v['camera']):v for v in em['provenance'] if v.get('mask') and v.get('mask_valid')}
    pairs = [p for p in agg.pairs(agg.RUN/'release_bg_gray.json') if p['role']=='known_rig_diagnostic' and p['pair_role']=='bottom']
    frames = []; hashes = {}
    calpath = Path(str(CAL).format('FR3')); cal = json.loads(calpath.read_text())
    cameras = {c:dict(K=np.asarray(v['K'],float), **dict(zip(('R','C'),pose(v['world_to_camera_mm'])))) for c,v in cal.items() if isinstance(v,dict) and 'K' in v}
    for session, frame in sorted({(p['group'],p['frame']) for p in pairs}):
        views = {}
        for c in 'ABCDEFG':
            if (session,frame,c) not in prov:continue
            path = ROOT/prov[session,frame,c]['mask']; hashes[str(path.relative_to(ROOT))] = sha(path)
            views[c] = dict(id=c, mask=read_mask(path), **cameras[c])
        if 'A' not in views or len(views)<3:
            raise ValueError(f'Insufficient masks for registered frame {(session,frame)}')
        frames.append(dict(frame_id=[session,frame],bottom=views['A'],sides=[views[c] for c in sorted(views) if c!='A']))
    hashes.update({str(EFFECTIVE.relative_to(ROOT)):sha(EFFECTIVE),str(calpath.relative_to(ROOT)):sha(calpath),'release_bg_gray.json':sha(agg.RUN/'release_bg_gray.json')})
    return frames,pairs,cameras,hashes

def real_package(max_frames=36, selection='uniform', seed=20261019, session=None, prior_scope='selected'):
    frames,pairs,cameras,hashes = real_inputs()
    available = [f for f in frames if session is None or f['frame_id'][0]==session]
    n = len(available) if max_frames is None else min(max_frames,len(available))
    if n<1: raise ValueError('No selected frames')
    if selection=='random': indices=sorted(random.Random(seed).sample(range(len(available)),n))
    elif selection=='legacy': indices=list(range(0,len(available),max(1,len(available)//n)))[:n]
    elif selection=='uniform': indices=np.linspace(0,len(available)-1,n,dtype=int).tolist()
    else:raise ValueError(selection)
    chosen = [available[i] for i in indices]; ids={tuple(f['frame_id']) for f in chosen}
    prior_pairs = [p for p in pairs if prior_scope=='all' or (p['group'],p['frame']) in ids]
    keys = sorted({p['key'] for p in prior_pairs})
    priors = [agg.geodesic_median([p['R'] for p in prior_pairs if p['key']==k]) @ cameras[chr(65+k[2])]['R'] for k in keys]
    R0 = agg.geodesic_median(priors)
    return dict(rig='pymousestudio_real',frames=chosen,R0=R0,Rgt=cameras['A']['R'],Cgt=cameras['A']['C'],up=np.array([0.,1.,0.]),metadata=dict(coordinate_frame='provided calibration world (mm)', selection=selection, seed=seed if selection=='random' else None, session=session, requested_frames=max_frames, frame_ids=[f['frame_id'] for f in chosen], side_view_ids=[dict(frame_id=f['frame_id'],cameras=[s['id'] for s in f['sides']]) for f in chosen], side_pairs=sum(len(f['sides']) for f in chosen), prior_scope=prior_scope, prior_frame_ids=sorted({(p['group'],p['frame']) for p in prior_pairs}), input_sha256=hashes))

def envelope(method, rows):
    import platform
    import scipy
    sources = ['bottom_pose_silhouette_data_20261019.py','bottom_pose_frontier_20261019.py','bottom_pose_from_hull_20261019.py','bottom_pose_aggregation_20261019.py']
    return dict(schema_version=2,method=method,neural_updates=0,oracle_inputs='Known side K/E; rendered or manual silhouettes; cached release gray-background predictions',source_sha256={p:sha(ROOT/'tools'/p) for p in sources},runtime=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,opencv=cv2.__version__),rows=rows)

def write_new(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
