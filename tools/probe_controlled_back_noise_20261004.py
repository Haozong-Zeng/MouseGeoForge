"""Controlled synthetic virtual-anchor noise; no real label or GT-forward use."""
import argparse, collections, datetime as dt, fcntl, json, math, os, random, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
import numpy as np
import torch
from torch.nn import functional as F
from mousegeoforge.models.back_point_integral import BackPointIntegralProbe
from mousegeoforge.uncertain_back_points import uncertain_back_point_loss
from run_depth_pose_ablation import fingerprint, write_json
CYCLE = ROOT / 'output/surface_curve_cycle_20261003'
OUT = CYCLE / 'controlled_back_noise_probe_v1'


def nearest_curve(points, curve):
    """Pixel-space exact nearest segment distance and material parameter."""
    start, segment = curve[:, :-1], curve[:, 1:] - curve[:, :-1]
    delta = points[:, :, None] - start[:, None]
    alpha = ((delta * segment[:, None]).sum(-1) /
             segment.square().sum(-1)[:, None].clamp_min(1e-12)).clamp(0, 1)
    distances = (delta - alpha[..., None] * segment[:, None]).norm(dim=-1)
    distance, index = distances.min(-1)
    parameter = (index + alpha.gather(-1, index[..., None]).squeeze(-1)) / segment.shape[1]
    return distance, parameter


def point_curve_geometry_loss(predicted_px, curve_px, valid):
    """Auxiliary known synthetic surface geometry, never a fabricated real line."""
    distance, _ = nearest_curve(predicted_px, curve_px.detach())
    if not valid.any(): return predicted_px.sum()*0
    return F.smooth_l1_loss(distance[valid]/768,torch.zeros_like(distance[valid]),beta=.01)/2


def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--curve-aux',action='store_true');args=parser.parse_args()
    if args.curve_aux: OUT=CYCLE/'controlled_back_curve_aux_v1'
    assert not OUT.exists()
    lock = (ROOT / 'output/teacher_temporal_cycle_20261001/gpu_job.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert torch.cuda.is_available()
    OUT.mkdir(); torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False
    deadline = dt.datetime(2026, 10, 4, 8, 5, 16, tzinfo=dt.timezone.utc)
    def status(stage, **kw):
        write_json(OUT / 'status.json', dict(stage=stage, pid=os.getpid(), utc=dt.datetime.now(dt.timezone.utc).isoformat(), **kw))
    status('loading')
    path = CYCLE / 'controlled_back_noise_inputs_v1/inputs.pt'
    saved = torch.load(path, map_location='cpu', weights_only=False)
    meta = saved['metadata']; d = {k: v.cuda() for k, v in saved['tensors'].items()}
    source = torch.load(CYCLE / 'expanded_curve_features_v1/frozen_inputs.pt', map_location='cpu', weights_only=False)
    assert source['metadata'] == meta
    manifest_path = CYCLE / 'curve_spatial_memory_v2/manifest.json'
    manifest = json.loads(manifest_path.read_text()); maps = [None] * len(meta)
    for record in manifest['records']:
        assert fingerprint(ROOT / record['path']) == record['sha256']
        cached = torch.load(ROOT / record['path'], map_location='cpu', weights_only=False)
        memory = F.interpolate(cached['memory'].float(), (48, 48), mode='bilinear', align_corners=False)
        for bind in record['bindings']:
            i = bind['observation']; assert maps[i] is None
            maps[i] = memory[bind['view_index']]
    assert all(m is not None for m in maps)
    spatial = torch.stack(maps).cuda()
    splits = {role: [i for i, m in enumerate(meta) if m['role'] == role and d['valid'][i].any()]
              for role in ('train', 'validation')}
    v = splits['validation']; assert torch.equal(d['clean_xy'][v], d['noisy_xy'][v])
    buckets = collections.defaultdict(list)
    for i in splits['train']: buckets[meta[i]['dataset']].append(i)
    keys = sorted(buckets)
    modes = (['noisy_strict', 'noisy_strict_curve', 'noisy_weak_chord_curve'] if args.curve_aux else
             ['clean_strict_reference', 'noisy_strict', 'noisy_weak_chord', 'noisy_weak_oracle'])
    seeds = [17, 29, 43]; steps = 2400
    write_json(OUT / 'protocol.json', dict(modes=modes, seeds=seeds, steps=steps, batch_size=8,
        input_sha256=fingerprint(path), memory_manifest_sha256=fingerprint(manifest_path),
        initialization='Cold spatial integral point probe, identical per seed across all arms.',
        schedule='Identical dataset-uniform TRAIN batches; cosine 3e-4 to 3e-5; AdamW decay .001.',
        targets='Virtual material anchors u=.30/.70, NOT Core23 redefinitions. No real data.',
        forward='Frozen pose, original predicted coarse06/07 prior, frozen image memory ONLY.',
        oracle='True tangent only in explicitly oracle weak loss, not deployable. Chord uses noisy labels only.',
        weak_loss='Existing along-tolerance4 normal1; chord reliability .5, oracle1. Loss scale differs by design; not a pure geometry-only ablation.',
        curve_auxiliary='If mode ends _curve: add 0.5 times point-to-known-synthetic-curve Huber. Surface GT used ONLY as loss, not forward; this is legitimate synthetic-only auxiliary, unavailable for untraced real data. Original noisy material point loss retained to discourage along-curve sliding. Does not train a full-curve prediction head.' if args.curve_aux else None,
        evaluation='Fixed final checkpoint; clean anchor error, nearest-curve error AND closest material-u error. Closest-u ambiguous for self-overlap; not independent 3D evidence.',
        deadline=deadline.isoformat(), no_production_promotion=True))

    def predict(model, ids):
        return model(d['pose'][ids], d['coarse_xy'][ids], spatial[ids])

    def evaluate(model, ids):
        values = collections.defaultdict(list)
        with torch.no_grad():
            for offset in range(0, len(ids), 32):
                ix = ids[offset:offset+32]; valid = d['valid'][ix]
                p = predict(model, ix); pixels = d['pixels'][ix, None]
                error = ((p - d['clean_xy'][ix]) * pixels).norm(dim=-1)
                distance, u = nearest_curve(p * pixels, d['curve_gt'][ix] * pixels)
                values['clean_anchor_px'].extend(error[valid].tolist())
                values['point_to_curve_px'].extend(distance[valid].tolist())
                values['nearest_material_u_error'].extend((u-d['clean_u'][ix]).abs()[valid].tolist())
                values['noisy_label_px'].extend(((p-d['noisy_xy'][ix])*pixels).norm(dim=-1)[valid].tolist())
        return {k: dict(n=len(v), mean=float(np.mean(v)), median=float(np.median(v)), p90=float(np.quantile(v,.9))) for k,v in values.items()}

    def evaluate_all(model):
        return {role: evaluate(model, ids) for role, ids in splits.items()}

    results = []; predictions = {}
    for seed in seeds:
        rng = random.Random(seed+6000)
        schedule = [[rng.choice(buckets[rng.choice(keys)]) for _ in range(8)] for _ in range(steps)]
        initial = None
        for mode in modes:
            arm = f'{mode}_seed{seed}'; torch.manual_seed(seed)
            model = BackPointIntegralProbe('spatial').cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.001)
            history = [dict(step=0, **evaluate_all(model))]
            if initial is None: initial = history[0]
            else: assert initial == history[0]
            target = d['clean_xy'] if mode == 'clean_strict_reference' else d['noisy_xy']
            status('training', arm=arm, step=0)
            for step, ids in enumerate(schedule, 1):
                assert dt.datetime.now(dt.timezone.utc) < deadline
                lr = 3e-5 + 2.7e-4 * .5 * (1+math.cos(math.pi*(step-1)/steps))
                for group in optimizer.param_groups: group['lr'] = lr
                optimizer.zero_grad(set_to_none=True)
                p = predict(model, ids)*d['pixels'][ids,None]
                gt = target[ids]*d['pixels'][ids,None]; losses = []
                for j, i in enumerate(ids):
                    valid = d['valid'][i]
                    if mode in ('clean_strict_reference', 'noisy_strict', 'noisy_strict_curve'):
                        losses.append(F.smooth_l1_loss(p[j,valid]/768,gt[j,valid]/768,beta=.01))
                    else:
                        prefix = 'chord' if mode.startswith('noisy_weak_chord') else 'oracle'
                        losses.append(uncertain_back_point_loss(p[j],gt[j],valid,
                            tangent=d[prefix+'_tangent'][i], reliability=d[prefix+'_reliability'][i]))
                point_loss = torch.stack(losses).mean()
                if mode.endswith('_curve'):
                    geometry=point_curve_geometry_loss(p,d['curve_gt'][ids]*d['pixels'][ids,None],d['valid'][ids])
                    if step==1:
                        params=list(model.parameters());pg=torch.autograd.grad(point_loss,params,retain_graph=True);cg=torch.autograd.grad(geometry,params,retain_graph=True)
                        pn=float(torch.sqrt(sum(g.square().sum() for g in pg)));cn=float(torch.sqrt(sum(g.square().sum() for g in cg)))
                        write_json(OUT/f'{arm}_gradient.json',dict(point_norm=pn,curve_norm=cn,weight=.5,weighted_ratio=.5*cn/max(pn,1e-12)))
                    loss=point_loss+.5*geometry
                else: loss=point_loss
                assert torch.isfinite(loss)
                loss.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                assert torch.isfinite(norm); optimizer.step()
                if step % 400 == 0:
                    row = dict(step=step, loss=float(loss.detach()), lr=lr, **evaluate_all(model)); history.append(row)
                    write_json(OUT/f'{arm}_history.json',history)
                    torch.save(dict(state=model.state_dict(),optimizer=optimizer.state_dict(),seed=seed,mode=mode,step=step),OUT/f'{arm}_latest.pt')
                    status('training',arm=arm,step=step)
                    print(arm,step,row['validation'],flush=True)
            subgroup = {name: evaluate(model,[i for i in splits['validation'] if meta[i]['dataset']==name])
                        for name in sorted({meta[i]['dataset'] for i in splits['validation']})}
            if args.curve_aux and mode=='noisy_strict':
                prior=json.loads((CYCLE/'controlled_back_noise_probe_v1/results.json').read_text())['results']
                prior=next(r for r in prior if r['arm']==arm)
                difference=abs(prior['validation']['clean_anchor_px']['mean']-history[-1]['validation']['clean_anchor_px']['mean'])
                assert difference<1e-4
                write_json(OUT/f'{arm}_reproduction.json',dict(validation_metric_delta=difference,passed=True))
            results.append(dict(arm=arm,seed=seed,mode=mode,**history[-1],validation_by_dataset=subgroup))
            write_json(OUT/'partial_results.json',results)
            with torch.no_grad():
                predictions[arm] = torch.cat([predict(model,list(range(i,min(i+32,len(meta))))).cpu() for i in range(0,len(meta),32)]).numpy()
            np.savez_compressed(OUT/'predictions.npz',**predictions)
    write_json(OUT/'results.json',dict(results=results,inspection_pending=True,no_production_promotion=True));status('complete')


if __name__ == '__main__':
    try: main()
    except Exception:
        OUT.mkdir(exist_ok=True);write_json(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()));raise
