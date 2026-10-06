"""Matched all-point readout pilot. Conditional association, not detector AP."""
import argparse
import collections
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
import numpy as np
import torch
from torch.nn import functional as F
from mousegeoforge.models.core23_spatial_readout import Core23SpatialReadout
from mousegeoforge.point_heatmap_supervision import point_heatmap_kl
from run_depth_pose_ablation import write_json, fingerprint, state_hash

CYCLE = ROOT / 'output/core23_spatial_cycle_20261004'
SOURCE = CYCLE / 'features_v1'
OUT = CYCLE / 'readout_pilot_v1'


def now():
    return dt.datetime.now(dt.timezone.utc)


def summarize(a):
    return dict(n=len(a), mean=float(np.mean(a)), median=float(np.median(a)),
                p90=float(np.quantile(a, .9))) if len(a) else dict(n=0, mean=None, median=None, p90=None)


def main():
    global OUT,SOURCE
    parser=argparse.ArgumentParser()
    parser.add_argument('--instance-prior', choices=['none','soft_box'], default='none')
    parser.add_argument('--feature-set', choices=['original','expanded_synthetic','expanded_all'], default='original')
    parser.add_argument('--extend-to-4800', action='store_true')
    parser.add_argument('--post4800',choices=['control','reliable_heatmap'])
    args=parser.parse_args()
    if args.instance_prior == 'soft_box':
        OUT=CYCLE/'readout_softbox_v1'
    if args.feature_set!='original':
        SOURCE=CYCLE/('features_'+args.feature_set+'_v1')
        OUT=CYCLE/('readout_'+args.feature_set+'_'+args.instance_prior+'_v1')
    resume_dir=None
    if args.extend_to_4800:
        assert args.feature_set=='expanded_all' and args.instance_prior=='soft_box'
        resume_dir=OUT
        assert (resume_dir/'results.json').exists()
        OUT=CYCLE/'readout_expanded_all_soft_box_extended4800_v1'
    if args.post4800:
        assert not args.extend_to_4800 and args.feature_set=='expanded_all' and args.instance_prior=='soft_box'
        resume_dir=CYCLE/'readout_expanded_all_soft_box_extended4800_v1'
        assert (resume_dir/'results.json').exists()
        OUT=CYCLE/f'readout_post4800_{args.post4800}_v1'
    plan = json.loads((CYCLE / 'plan.json').read_text())
    deadline = dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z', '+00:00'))
    assert now() < deadline
    lock = (ROOT / plan['gpu_lock']).open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not OUT.exists(), 'Inspect existing pilot before explicit resume or new version.'
    manifest = json.loads((SOURCE / 'manifest.json').read_text())
    OUT.mkdir()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    state = json.loads((CYCLE / 'state.json').read_text())
    state.update(stage='training_matched_full23_readouts', active_job=OUT.name, active_pid=os.getpid(), updated_utc=now().isoformat())
    write_json(CYCLE / 'state.json', state)

    def status(stage, **extra):
        write_json(OUT / 'status.json', dict(stage=stage, pid=os.getpid(), utc=now().isoformat(), **extra))

    status('loading')
    rows, maps = [], []
    for r in manifest['records']:
        assert fingerprint(ROOT / r['path']) == r['sha256']
        p = torch.load(ROOT / r['path'], map_location='cpu', weights_only=False)
        memory = F.interpolate(p['memory'].float(), (48, 48), mode='bilinear', align_corners=False)
        for obs in p['observations']:
            m = obs['metadata']
            assert m['record'].get('session') not in ('GR2', 'UR2', 'UR3')
            rows.append(dict(obs, pixels=torch.tensor(p['image_hw'][::-1], dtype=torch.float32)))
            maps.append(memory[m['view_index']])
    memory = torch.stack(maps).cuda()
    del maps
    data = {k: torch.stack([r[k] for r in rows]).cuda() for k in
            ('pose', 'coarse_xy', 'original_xy', 'gt_xy', 'valid', 'pixels', 'visible', 'visibility_valid')}
    meta = [r['metadata'] for r in rows]
    predicted_boxes=None
    if args.instance_prior=='soft_box':
        prior_file=SOURCE/'prior_manifest.json' if args.feature_set!='original' else CYCLE/'instance_priors_v1/manifest.json'
        prior_manifest=json.loads(prior_file.read_text())
        assert prior_manifest['source_manifest_sha256']==fingerprint(SOURCE/'manifest.json')
        assert fingerprint(ROOT/prior_manifest['path'])==prior_manifest['sha256']
        priors=torch.load(ROOT/prior_manifest['path'],map_location='cpu',weights_only=False)
        predicted_boxes=priors['boxes_cxcywh'].cuda()
        assert predicted_boxes.shape==(len(meta),4)
    # Keep fixed prior user policy: multi-mouse tails disabled, all arms alike.
    for i, m in enumerate(meta):
        if m['instance_count'] > 1:
            data['valid'][i, 7:14] = False
    def kind(i):
        return 'real' if meta[i]['record']['kind'] == 'real' else 'synthetic'
    sets = {(k, role): [i for i, m in enumerate(meta) if kind(i) == k and m['record']['role'] == role and data['valid'][i].any()]
            for k in ('real', 'synthetic') for role in ('train', 'validation')}
    assert all(sets.values())
    buckets = {k: collections.defaultdict(list) for k in ('real', 'synthetic')}
    for k in buckets:
        for i in sets[k, 'train']:
            record = meta[i]['record']
            source = record.get('session') or record['scene'].split('/')[2]
            buckets[k][source].append(i)

    def predict(model, ids, log_probability=False):
        return model(data['pose'][ids], data['coarse_xy'][ids], memory[ids], predicted_boxes=None if predicted_boxes is None else predicted_boxes[ids],return_log_probability=log_probability)

    @torch.no_grad()
    def predictions(model):
        return torch.cat([predict(model, list(range(s, min(s+16, len(meta))))) for s in range(0, len(meta), 16)])

    def evaluate(p):
        error = ((p-data['gt_xy']) * data['pixels'][:, None]).norm(dim=-1).cpu()
        valid = data['valid'].cpu()
        metrics = {}
        for (k, role), ids in sets.items():
            prefix = k + '_' + role
            accum = collections.defaultdict(list)
            for i in ids:
                e, good = error[i], valid[i]
                accum['all_supported'].extend(e[good].tolist())
                reliable = good.clone()
                reliable[5:7] = False
                if k == 'real':
                    select = torch.zeros(23, dtype=torch.bool)
                    select[[0, 14, 15, 16, 17]] = True
                    reliable &= select
                accum['reliable'].extend(e[reliable].tolist())
                accum['back06_07'].extend(e[5:7][good[5:7]].tolist())
                accum['multi_instance' if meta[i]['instance_count'] > 1 else 'single_instance'].extend(e[good].tolist())
                for j in range(23):
                    if good[j]:
                        accum[f'point{j+1:02d}'].append(float(e[j]))
                source = meta[i]['record'].get('session') or meta[i]['record']['scene'].split('/')[2]
                accum['source/' + source].extend(e[good].tolist())
                known = data['visibility_valid'][i].cpu() & good
                visible = data['visible'][i].cpu().bool()
                accum['known_visible'].extend(e[known & visible].tolist())
                accum['known_occluded'].extend(e[known & ~visible].tolist())
            metrics[prefix] = {key: summarize(values) for key, values in accum.items()}
        return metrics

    def loss_for(model, ids):
        if args.post4800=='reliable_heatmap':
            p,prob,logp=predict(model,ids,log_probability=True)
        else:p = predict(model, ids)
        scale = data['pixels'][ids, None] / 768
        safe_gt = torch.where(data['valid'][ids, :, None], data['gt_xy'][ids], p.detach())
        point_loss = F.smooth_l1_loss(p*scale, safe_gt*scale, beta=.01, reduction='none').mean(-1)
        mask = data['valid'][ids]
        coordinate=((point_loss*mask).sum(-1)/mask.sum(-1).clamp_min(1)).mean()
        if args.post4800=='reliable_heatmap':
            selected=torch.zeros_like(mask);selected[:,[0,14,15,16,17]]=True
            heatmap=point_heatmap_kl(prob,data['gt_xy'][ids],mask&selected,data['pixels'][ids],sigma_px=12.,log_probability=logp)
            return coordinate+.002*heatmap
        return coordinate

    seeds, steps = [17, 29, 43], 6400 if args.post4800 else 4800 if args.extend_to_4800 else 1600
    start_step=4800 if args.post4800 else 1600 if resume_dir is not None else 0
    write_json(OUT / 'protocol.json', dict(
        seeds=seeds, steps=steps, modes=['global', 'spatial'], batch_per_kind=8,
        learning_rate=f'resumeAdamW at{start_step}; extra{steps-start_step} cosine1e-4to1e-5' if resume_dir else 'cosine3e-4to1e-5', real_loss_weight=.25, weight_decay=.001,
        start_step=start_step,
        resume_source=None if resume_dir is None else str(resume_dir.relative_to(ROOT)),
        post4800=args.post4800,
        heatmap_policy='If reliable_heatmap: coordinate loss unchanged plus.002KL, sigma12pixels at768scale, IDs1/15/16/17/18 only in BOTH domains, existing valid masks. Logsoftmax stable gradients. Real06/07 untouched. Kernel is not calibrated label uncertainty.',
        budget_caveat='Continuation with a prespecified lower-LR restart, not an identical-LR pure step-count causal test; source checkpoints remain untouched.' if resume_dir else None,
        initialization='Each point uses own256d decoder token; same parameter count and exact seed init. Zero-query broad predicted Gaussian prior sigma.25; not identical to original predictions near boundaries.',
        target_policy='Immutable original validity, real only1/6/7/15/16/17/18. Multi-mouse tails8-14 disabled in BOTH training and evaluation. Unknown visibility not assigned.',
        evaluation=f'Fixed {steps}-step final, curves every200. Box/objectness-assisted matching, NOT detector AP. All points per supported mask, no fabricated real spine/curves. FR2/AZR2L2 are development only.',
        features_manifest_sha256=fingerprint(SOURCE / 'manifest.json'),
        script_sha256=fingerprint(Path(__file__)),
        readout_module_sha256=fingerprint(ROOT/'mousegeoforge/models/core23_spatial_readout.py'),
        feature_set=args.feature_set,
        instance_prior=args.instance_prior,
        instance_prior_rule='If enabled: frozen predicted box expanded25percent; smooth outside penalty capped6; no GT crop, confidence or instance-count gating.',
        inspiration='Prediction-specific foreground restriction in Mask2Former (https://bowenc0221.github.io/mask2former/); this soft-box integral probe is NOT a reproduction of masked transformer attention.',
        source_counts={str(k): len(v) for k, v in sets.items()},
        limitations=['Frozen-head feature probe, not production integration.', 'Controlled frozen-feature sample; data size is specified by feature_set/source_counts.', 'Two-view feature contexts only; 1/multiple-view integration remains unverified.'],
        training_stop_utc=plan['training_stop_utc']))
    write_json(OUT / 'baseline.json', dict(original_refined=evaluate(data['original_xy']), original_coarse=evaluate(data['coarse_xy'])))
    write_json(OUT / 'observations.json', meta)
    results = []
    all_predictions = {}
    for seed in seeds:
        rng = random.Random(seed+3000)
        schedule = [{k: [rng.choice(buckets[k][rng.choice(sorted(buckets[k]))]) for _ in range(8)] for k in buckets} for _ in range(steps)]
        if resume_dir is not None:
            old_schedule=json.loads((resume_dir/f'schedule_seed{seed}.json').read_text())
            assert schedule[:start_step]==old_schedule['steps']
        write_json(OUT / f'schedule_seed{seed}.json', dict(steps=schedule, sha256=hashlib.sha256(json.dumps(schedule, sort_keys=True).encode()).hexdigest()))
        init_hash = None
        for mode in ('global', 'spatial'):
            arm = f'{mode}_seed{seed}'
            torch.manual_seed(seed)
            model = Core23SpatialReadout(mode).cuda()
            this_hash = state_hash({k: v.detach().cpu() for k, v in model.state_dict().items()})
            if init_hash is None:
                init_hash = this_hash
            else:
                assert this_hash == init_hash
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.001)
            if resume_dir is not None:
                cp=resume_dir/f'{arm}_latest.pt'
                previous=torch.load(cp,map_location='cuda',weights_only=False)
                assert previous['step']==start_step and previous['initial_hash']==init_hash
                model.load_state_dict(previous['state'],strict=True);optimizer.load_state_dict(previous['optimizer'])
                write_json(OUT/f'{arm}_resume.json',dict(checkpoint_sha256=fingerprint(cp),step=start_step,initial_hash=init_hash))
            history = [dict(step=start_step, metrics=evaluate(predictions(model)))]
            if resume_dir is not None:
                old_history=json.loads((resume_dir/f'{arm}_history.json').read_text())
                assert history[0]['metrics']==old_history[-1]['metrics'], 'Resume prediction/metric mismatch'
            for step, ids in enumerate(schedule[start_step:], start_step+1):
                if now() >= deadline:
                    torch.save(dict(state=model.state_dict(), optimizer=optimizer.state_dict(), step=step-1, seed=seed, mode=mode), OUT / f'{arm}_deadline.pt')
                    status('deadline_stopped', arm=arm, step=step-1)
                    state.update(stage='training_deadline_reached_wrapup_required', active_pid=None, active_job=None, updated_utc=now().isoformat())
                    write_json(CYCLE / 'state.json', state)
                    return
                peak_lr=1e-4 if resume_dir else 3e-4
                lr = 1e-5 + (peak_lr-1e-5)*.5*(1+math.cos(math.pi*(step-1-start_step)/(steps-start_step)))
                for g in optimizer.param_groups:
                    g['lr'] = lr
                optimizer.zero_grad(set_to_none=True)
                synthetic = loss_for(model, ids['synthetic'])
                real = loss_for(model, ids['real'])
                loss = synthetic + .25*real
                assert torch.isfinite(loss)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                assert torch.isfinite(norm)
                optimizer.step()
                if step % 200 == 0:
                    row = dict(step=step, lr=lr, synthetic_loss=float(synthetic.detach()), real_loss=float(real.detach()), metrics=evaluate(predictions(model)))
                    history.append(row)
                    write_json(OUT / f'{arm}_history.json', history)
                    torch.save(dict(state=model.state_dict(), optimizer=optimizer.state_dict(), step=step, seed=seed, mode=mode, initial_hash=init_hash), OUT / f'{arm}_latest.pt')
                    status('training', arm=arm, step=step, steps=steps)
                    print(arm, step, 'real_reliable', row['metrics']['real_validation']['reliable']['mean'], 'syn_all', row['metrics']['synthetic_validation']['all_supported']['mean'], flush=True)
            results.append(dict(arm=arm, initial_hash=init_hash, **history[-1]))
            write_json(OUT / 'partial_results.json', results)
            all_predictions[arm] = predictions(model).cpu().numpy()
    np.savez_compressed(OUT / 'predictions.npz', **all_predictions)
    write_json(OUT / 'results.json', dict(results=results, visual_inspection_pending=True, no_production_promotion=True))
    status('complete')
    state.update(stage='readout_pilot_complete_review_required', active_pid=None, active_job=None, updated_utc=now().isoformat(),
                 next='Review full per-point curves versus original refined baseline, inspect real feet/back and synthetic single/dual failures visually. Check whether gains generalize before feature expansion or head integration; no waiting for human tracing.')
    state['completed'].append(OUT.name+f'_matched_global_spatial_3seeds{steps}steps')
    write_json(CYCLE / 'state.json', state)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        OUT.mkdir(exist_ok=True)
        write_json(OUT / 'failure.json', dict(utc=now().isoformat(), traceback=__import__('traceback').format_exc()))
        raise
