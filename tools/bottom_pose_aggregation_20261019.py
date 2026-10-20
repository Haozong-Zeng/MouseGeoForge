"""Zero-training probes on saved v4.2 relative poses (release model).

1. Multi-frame robust rotation averaging for fixed rigs: the same camera pair
   observed in N synchronized instants; error of the averaged estimate.
2. Prior-constrained combination: use the full-image pose only when it agrees
   with the mouse-only (background-gray) pose within tau degrees, else use the
   mouse-only pose. tau = 10 deg is the primary value fixed before analysis;
   5/15/20 are reported as a sensitivity curve, not used for selection.

Inputs: eval_v4_2_20261019/release_original.json and release_bg_gray.json.
The bg_gray input uses GT/human-confirmed body masks: oracle diagnostic.
"""
import collections
import json
import math
import random
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'output/bottom_geometry_v3_local12h_20261018/eval_v4_2_20261019'
OUT = RUN / 'pose_aggregation_v1.json'
NS = (1, 3, 5, 10, 20, 'all')
REPEATS = 50
TAUS = (5., 10., 15., 20.)
PRIMARY_TAU = 10.


def angle(Ra, Rb):
    return math.degrees(math.acos(max(-1., min(1., (np.trace(Ra @ Rb.T) - 1) / 2))))


def project_so3(M):
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def log_so3(R):
    return Rotation.from_matrix(R).as_rotvec()


def exp_so3(v):
    return Rotation.from_rotvec(v).as_matrix()


def chordal_mean(Rs):
    return project_so3(sum(Rs))


def geodesic_median(Rs, iters=50):
    """Weiszfeld on SO(3), initialised at the medoid (robust to flipped outliers)."""
    if len(Rs) == 1:
        return Rs[0]
    D = np.array([[angle(a, b) for b in Rs] for a in Rs])
    R = Rs[int(D.sum(1).argmin())]
    for _ in range(iters):
        vs = [log_so3(R.T @ Ri) for Ri in Rs]
        ws = [1. / max(np.linalg.norm(v), 1e-6) for v in vs]
        step = sum(w * v for w, v in zip(ws, vs)) / sum(ws)
        R = R @ exp_so3(step)
        if np.linalg.norm(step) < 1e-9:
            break
    return R


def direction_median(ts):
    u = [t / max(np.linalg.norm(t), 1e-12) for t in ts]
    m = sum(u); m = m / max(np.linalg.norm(m), 1e-12)
    for _ in range(50):  # Weiszfeld on the sphere (chordal)
        w = [1. / max(np.linalg.norm(x - m), 1e-6) for x in u]
        n = sum(wi * x for wi, x in zip(w, u)); n = n / max(np.linalg.norm(n), 1e-12)
        if np.linalg.norm(n - m) < 1e-10:
            break
        m = n
    return m


def dir_err(t, gt):
    a = t / max(np.linalg.norm(t), 1e-12); b = gt / max(np.linalg.norm(gt), 1e-12)
    return math.degrees(math.acos(max(-1., min(1., float(a @ b)))))


def pairs(path):
    rows = json.loads(Path(path).read_text())['rows']
    out = []
    for i, r in enumerate(rows):
        for c in r['cameras']:
            if 'pred_rel_R' not in c:
                continue
            rig = r['rig'] or ('pymousestudio_real' if r['evaluation_role'] == 'known_rig_diagnostic' else r['group'])
            out.append(dict(row=i, role=r['evaluation_role'], tier=r.get('dev_tier'), rig=rig, key=(r['evaluation_role'], rig, c['ref_view'], c['view']),
                            pair_role=c['pair_role'], R=np.array(c['pred_rel_R']), gtR=np.array(c['gt_rel_R']),
                            t=np.array(c['pred_rel_t']), gtt=np.array(c['gt_rel_t']), frame=r.get('frame'), group=r['group']))
    return out


def summarize(errs):
    e = np.array(errs)
    return dict(n=len(e), lt5=float((e < 5).mean()), lt15=float((e < 15).mean()), median=float(np.median(e)))


def main():
    orig = pairs(RUN / 'release_original.json'); gray = pairs(RUN / 'release_bg_gray.json')
    assert len(orig) == len(gray) and all(a['key'] == b['key'] and a['row'] == b['row'] for a, b in zip(orig, gray))
    sources = dict(original=orig, mouse_only=gray)
    for tau in TAUS:
        comb = []
        for a, b in zip(orig, gray):
            use = a if angle(a['R'], b['R']) < tau else b
            comb.append(dict(b, R=use['R'], t=use['t'], used_original=use is a))
        sources[f'combined_tau{int(tau)}'] = comb
    rng = random.Random(20261019)
    result = dict(primary_tau=PRIMARY_TAU, oracle_note='mouse_only uses GT/human body masks; not deployable',
                  groups={}, combination_usage={})
    for name in sources:
        if name.startswith('combined'):
            c = sources[name]
            result['combination_usage'][name] = {f'{role}/{pr}': float(np.mean([x['used_original'] for x in c if x['role'] == role and x['pair_role'] == pr]))
                                                 for role in ('known_rig_diagnostic', 'unseen_rig_gate') for pr in ('other', 'bottom')}
    for role in ('known_rig_diagnostic', 'unseen_rig_gate'):
        for pr in ('other', 'bottom'):
            block = {}
            for name, src in sources.items():
                by = collections.defaultdict(list)
                for x in src:
                    if x['role'] == role and x['pair_role'] == pr:
                        by[x['key']].append(x)
                for k, xs in by.items():  # fixed rig: GT identical across instants
                    assert max(angle(x['gtR'], xs[0]['gtR']) for x in xs) < .1, k  # float32 normalisation noise <=0.031 deg observed
                res = {}
                for N in NS:
                    rerr = {'single': [], 'chordal_mean': [], 'geodesic_median': []}; derr = []
                    for k, xs in by.items():
                        size = len(xs) if N == 'all' else N
                        if size > len(xs):
                            continue
                        reps = 1 if (N == 'all' or N == 1) else REPEATS
                        subsets = [[x] for x in xs] if N == 1 else [rng.sample(xs, size) for _ in range(reps)]
                        for sub in subsets:
                            Rs = [x['R'] for x in sub]; gt = sub[0]['gtR']
                            if N == 1:
                                rerr['single'].append(angle(Rs[0], gt)); derr.append(dir_err(sub[0]['t'], sub[0]['gtt']))
                            else:
                                rerr['chordal_mean'].append(angle(chordal_mean(Rs), gt))
                                rerr['geodesic_median'].append(angle(geodesic_median(Rs), gt))
                                derr.append(dir_err(direction_median([x['t'] for x in sub]), sub[0]['gtt']))
                    entry = {m: summarize(v) for m, v in rerr.items() if v}
                    if derr:
                        entry['direction'] = summarize(derr)
                    entry['camera_pairs'] = len(by); entry['instants_per_pair'] = sorted({len(v) for v in by.values()})
                    if entry.get('single') or entry.get('geodesic_median'):
                        res[str(N)] = entry
                block[name] = res
            result['groups'][f'{role}/{pr}'] = block
    OUT.write_text(json.dumps(result, indent=1))
    print('WROTE', OUT)


if __name__ == '__main__':
    main()
