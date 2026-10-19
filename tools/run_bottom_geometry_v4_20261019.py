"""Explicit v4 prepare/preflight/train commands. No scheduler or automatic next arm.

Frozen historical v3 files are dependencies, not patched entry points. Preparing
or preflighting never updates an optimizer. Training needs a separately enabled
configuration, deadlines, a passed COMPLETE-objective preflight and --execute.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import fcntl
import json
import math
from pathlib import Path
import signal
import time

import torch

import batched_geometry_20261018 as batched
import bottom_geometry_v3_data_20261018 as data
import bottom_geometry_v4_core_20261019 as core
import bottom_geometry_v4_runtime_20261019 as rt
import resume_geometry_workers_20261018 as state

STOP = False


def weighted(terms, weights):
    if set(terms) - set(weights):
        raise ValueError('Unregistered loss weights: ' + str(set(terms)-set(weights)))
    return sum(weights[k]*v for k, v in terms.items()) / 8


def backward(model, teacher, feed, items, weights, cap, anchor=True):
    """Exactly eight ordered sample means; complete objective shared by all modes."""
    if len(items) != 8:
        raise ValueError('An update must contain exactly eight samples')
    model.zero_grad(set_to_none=True); total = 0.; losses = collections.Counter()
    for group in batched.groups(items, cap):
        batches = [feed.next(item) for item in group]
        pred = batched.geometry_forward(model, batches)
        target = None
        if teacher is not None:
            # Same sample-wise encoder and physical head grouping as the student.
            with torch.no_grad():
                target = batched.geometry_forward(teacher, batches)
        objective = 0.
        for i, batch in enumerate(batches):
            pp = batched.sample_prediction(pred, i, len(group))
            tt = batched.sample_prediction(target, i, len(group)) if target is not None else None
            terms = core.complete_terms(pp, batch, teacher=tt, real_anchor=anchor)
            objective = objective + weighted(terms, weights)
            for name, value in terms.items():
                losses[name] += float(value.detach())/8
        if not torch.isfinite(objective):
            raise FloatingPointError('Nonfinite complete objective')
        total += float(objective.detach()); objective.backward()
    return total, dict(losses)


def snapshot(model):
    return {n: (p.grad.detach().cpu().clone() if p.grad is not None else torch.zeros_like(p, device='cpu'))
            for n, p in model.named_parameters() if p.requires_grad}


def preflight(model, teacher, manifest, items, weights, destination, anchor=True):
    rng = rt.geo.old.rng_state(); reference = None; rows = []
    try:
        for cap in (1, 2, 4):
            state.restore_rng(rng); feed = data.Feed(manifest, items, 0)
            try:
                loss, terms = backward(model, teacher, feed, items, weights, cap, anchor)
                grads = snapshot(model)
            finally:
                feed.close()
            if reference is None:
                reference = loss, grads
                rows.append(dict(cap=cap, loss=loss, terms=terms))
            else:
                comparison = core.gradient_comparison(reference[1], grads)
                relative_loss = abs(loss-reference[0])/max(abs(reference[0]), 1e-8)
                rows.append(dict(cap=cap, loss=loss, terms=terms, relative_loss=relative_loss, **comparison))
                if relative_loss > .002 or comparison['relative_l2'] > .003 or comparison['cosine'] < .9999:
                    raise ValueError('Full-objective physical-batch parity failed')
        rt.write_new(destination, dict(passed=True, rows=rows, optimizer_steps=0,
                                      teacher_included=teacher is not None, real_anchor=anchor,
                                      teacher_weight=weights.get('teacher', 0), complete_objective=True))
    except BaseException as error:
        rt.write_new(destination, dict(passed=False, rows=rows, optimizer_steps=0, error=str(error)))
        raise
    finally:
        model.zero_grad(set_to_none=True); state.restore_rng(rng)


def source_files():
    names = ['bottom_geometry_v4_core_20261019', 'bottom_geometry_v4_runtime_20261019',
             'bottom_geometry_v4_eval_20261019', 'run_bottom_geometry_v4_20261019',
             'bottom_geometry_v3_losses_20261018', 'bottom_geometry_v3_data_20261018',
             'batched_geometry_20261018', 'geometry_prefetch_20261018',
             'resume_geometry_workers_20261018', 'pure_geometry_bottom_20261018',
             'bottom_domain_adaptation_20261018', 'audit_bottom_view_coverage']
    names += ['calibrate_bottom_geometry_v4_20261019', 'diagnose_bottom_geometry_v4_20261019']
    files = [rt.geo.ROOT/'tools'/(n+'.py') for n in names]
    upstream = sorted((rt.geo.ROOT/'vggt-omega/vggt_omega').rglob('*.py'))
    upstream += sorted((rt.geo.ROOT/'vggt-omega/training').rglob('*.py'))
    if not upstream:
        raise FileNotFoundError('Provide the pinned upstream source checkout under vggt-omega/')
    files += upstream
    return {str(f.relative_to(rt.geo.ROOT)): rt.sha(f) for f in files}


def resolve(config):
    result = dict(config)
    for name in ('manifest', 'release', 'output_dir', 'data_file_pins'):
        result[name] = (rt.geo.ROOT/config[name]).resolve()
    return result


def verify_payload(pin_path):
    # Content identity survives cluster copies whose mtimes differ from WSL.
    for name, pin in json.loads(Path(pin_path).read_text()).items():
        path = rt.geo.ROOT/name
        if path.stat().st_size != pin['bytes'] or rt.sha(path) != pin['sha256']:
            raise ValueError('Dataset content pin mismatch: ' + name)


def prepare(config_path):
    cfg = json.loads(config_path.read_text()); c = resolve(cfg)
    if cfg.get('effective_batch') != 8 or cfg.get('head_batch_cap') not in (1, 2, 4):
        raise ValueError('Invalid batch configuration')
    if cfg.get('encoder_batch') != 1 or cfg.get('workers') != 4:
        raise ValueError('Registered implementation uses sample-wise encoder and four workers')
    if not 0 < cfg['warmup_updates'] < cfg['updates']:
        raise ValueError('Warmup must leave a positive decay interval')
    if cfg['updates'] == 3000 and cfg['warmup_updates'] != 500:
        raise ValueError('Formal pair uses the registered 500-update warmup')
    needed = {'camera', 'depth', 'shape', 'transfer', 'real_sparse', 'quaternion_radius', 'real_camera_anchor', 'teacher'}
    if set(cfg['weights']) != needed or any(not math.isfinite(x) or x < 0 for x in cfg['weights'].values()):
        raise ValueError('Explicit finite nonnegative weights for every objective term required')
    if cfg.get('arm_names') != ['P0', 'P1'] or cfg.get('real_anchor') is not True:
        raise ValueError('Both paired arms require the SAME real anchor')
    if not 0 < cfg['weights'].get('teacher', 0) <= .05:
        raise ValueError('Teacher weight must be positive and <= .05, calibrated on TRAIN')
    if cfg['updates'] not in (300, 3000) or cfg['seed'] not in (17, 29, 43):
        raise ValueError('Unregistered endpoint or seed')
    if cfg['updates'] == 3000 and not cfg.get('learnability_receipt'):
        raise ValueError('Formal pair needs a separately reviewed learnability receipt')
    if c['output_dir'].exists():
        raise FileExistsError('Use a fresh namespace; preparation is immutable')
    manifest = rt.read_manifest(c['manifest'])
    verify_payload(c['data_file_pins'])
    schedule = data.schedule(manifest, cfg['updates'])
    receipts = {}
    for name in ('calibration_receipt', 'learnability_receipt'):
        if cfg.get(name):
            receipt_path = rt.geo.ROOT/cfg[name]
            receipt = json.loads(receipt_path.read_text())
            if not receipt.get('passed'):
                raise ValueError('Receipt is not passed: ' + name)
            if name == 'calibration_receipt' and receipt.get('weights') != cfg['weights']:
                raise ValueError('Calibration weights differ from config')
            if receipt.get('manifest_sha256') != rt.sha(c['manifest']) or receipt.get('release_sha256') != rt.sha(c['release']):
                raise ValueError('Receipt is from different data or release: ' + name)
            if name == 'learnability_receipt':
                if receipt.get('optimizer_steps') != 300 or receipt.get('r_component_reduction', 0) < .30 or receipt.get('train_bottom_median_reduction', 0) < .20:
                    raise ValueError('Learnability receipt does not meet registered TRAIN criteria')
            receipts[cfg[name]] = rt.sha(receipt_path)
    pins = dict(config=cfg, manifest_sha256=rt.sha(c['manifest']), release_sha256=rt.sha(c['release']),
                data_pins_sha256=rt.sha(c['data_file_pins']), receipt_pins=receipts,
                source_pins=source_files(), schedule_sha256=state.digest(schedule),
                scope='free-moving geometry only; no sealed inference or automatic follow-up')
    rt.write_new(c['output_dir']/'protocol.json', pins)
    rt.write_new(c['output_dir']/'schedule.json', schedule)
    print('Prepared immutable protocol; no model loaded and no optimizer updates', flush=True)


def verified(config_path):
    cfg = json.loads(config_path.read_text()); c = resolve(cfg)
    protocol = json.loads((c['output_dir']/'protocol.json').read_text())
    if protocol['config'] != cfg or protocol['source_pins'] != source_files():
        raise ValueError('Config/source changed after prepare; create a new namespace')
    if protocol['manifest_sha256'] != rt.sha(c['manifest']) or protocol['release_sha256'] != rt.sha(c['release']):
        raise ValueError('Manifest/release pin mismatch')
    if rt.sha(c['data_file_pins']) != protocol['data_pins_sha256']:
        raise ValueError('Dataset pin list changed')
    verify_payload(c['data_file_pins'])
    for name, digest in protocol['receipt_pins'].items():
        if rt.sha(rt.geo.ROOT/name) != digest:
            raise ValueError('Prerequisite receipt changed: ' + name)
    work = json.loads((c['output_dir']/'schedule.json').read_text())
    if state.digest(work) != protocol['schedule_sha256']:
        raise ValueError('Schedule changed')
    return cfg, c, protocol, work


def save(path, model, optimizer, ema, step, work, protocol_sha, release_sha, exposures):
    tmp = path.with_suffix('.tmp')
    torch.save(dict(model={n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
                    buffers={n: b.detach().cpu() for n, b in model.named_buffers()},
                    ema={n: v.detach().cpu() for n, v in ema.items()}, optimizer=optimizer.state_dict(),
                    rng=rt.geo.old.rng_state(), step=step, next_microbatch_offset=step*8,
                    schedule_sha256=state.digest(work), protocol_sha256=protocol_sha,
                    base_sha256=release_sha, exposures=dict(exposures)), tmp)
    tmp.replace(path)


def restore(path, model, optimizer, protocol_sha, release_sha, work):
    ck = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    if (ck['protocol_sha256'], ck['base_sha256'], ck['schedule_sha256']) != (protocol_sha, release_sha, state.digest(work)):
        raise ValueError('Complete-state identity mismatch')
    named = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if set(named) != set(ck['model']) or set(named) != set(ck['ema']):
        raise ValueError('Trainable/EMA keys differ')
    with torch.no_grad():
        for n, p in named.items():
            p.copy_(ck['model'][n])
        for n, b in model.named_buffers():
            b.copy_(ck['buffers'][n])
    optimizer.load_state_dict(ck['optimizer'])
    ema = {n: v.to(named[n].device).clone() for n, v in ck['ema'].items()}
    expected = collections.Counter(i['sampling_role'] for micros in work[:ck['step']] for i in micros)
    if dict(expected) != ck['exposures'] or ck['next_microbatch_offset'] != 8*ck['step']:
        raise ValueError('Sampler progress mismatch')
    checks = [state.digest(named) == state.digest(ck['model']), state.digest(ema) == state.digest(ck['ema']),
              state.digest(optimizer.state_dict()) == state.digest(ck['optimizer']),
              state.digest(dict(model.named_buffers())) == state.digest(ck['buffers'])]
    state.restore_rng(ck['rng']); checks.append(state.digest(rt.geo.old.rng_state()) == state.digest(ck['rng']))
    if not all(checks):
        raise ValueError('Serialized state did not restore exactly')
    return ema, ck['step'], expected


def run(config_path, arm, command, execute=False, resume=False):
    if command == 'train' and (not execute or not json.loads(config_path.read_text()).get('execution_authorized')):
        raise ValueError('Training disabled: explicit --execute AND separately enabled config required')
    cfg, c, protocol, work = verified(config_path)
    manifest = rt.read_manifest(c['manifest']); dest = c['output_dir']/arm
    if command == 'train':
        if cfg['updates'] == 300 and arm != 'P0':
            raise ValueError('Learnability probe is P0 only; teacher is a later contrast')
        if not execute or not cfg.get('execution_authorized'):
            raise ValueError('Training disabled: explicit --execute AND separately enabled config required')
        for key in ('optimization_deadline_utc', 'hard_deadline_utc'):
            if not cfg.get(key) or dt.datetime.fromisoformat(cfg[key]).utcoffset() is None:
                raise ValueError('Timezone-aware absolute deadlines required')
        if dt.datetime.fromisoformat(cfg['optimization_deadline_utc']) >= dt.datetime.fromisoformat(cfg['hard_deadline_utc']):
            raise ValueError('Reserve evaluation/closure time after optimization')
        if dt.datetime.now(dt.timezone.utc) >= dt.datetime.fromisoformat(cfg['optimization_deadline_utc']):
            raise ValueError('Optimization deadline already passed')
        receipt = json.loads((dest/'preflight.json').read_text())
        if not receipt.get('passed') or not receipt.get('complete_objective') or receipt.get('teacher_included') != (arm == 'P1'):
            raise ValueError('This arm has no complete passed preflight')
        if not cfg.get('calibration_receipt'):
            raise ValueError('TRAIN-only module calibration receipt required before training')
        calibration = json.loads((rt.geo.ROOT/cfg['calibration_receipt']).read_text())
        if calibration.get('weights') != cfg['weights'] or not calibration.get('passed'):
            raise ValueError('Calibrated weights must equal frozen protocol weights')
        if cfg['updates'] == 3000:
            for peer in ('P0', 'P1'):
                other = json.loads((c['output_dir']/peer/'preflight.json').read_text())
                if not other.get('passed') or not other.get('complete_objective'):
                    raise ValueError('Both arms must pass complete preflight before the formal pair')
    else:
        dest.mkdir(parents=True, exist_ok=True)
        if (dest/'preflight.json').exists():
            raise FileExistsError('Preserve earlier preflight; use a new attempt namespace')
    model, optimizer = rt.model(c['release'], trainable=True, seed=cfg['seed'],
                                camera_lr=cfg['camera_lr'], encoder_lr=cfg['encoder_lr'], depth_lr=cfg['depth_lr'])
    rng = rt.geo.old.rng_state()
    teacher = rt.model(c['release'], seed=cfg['seed']) if arm == 'P1' else None
    state.restore_rng(rng)
    if command == 'preflight':
        preflight(model, teacher, manifest, work[0], cfg['weights'], dest/'preflight.json', cfg['real_anchor'])
        return
    names = {n: p for n, p in model.named_parameters() if p.requires_grad}
    protocol_sha = rt.sha(c['output_dir']/'protocol.json'); step = 0; exposures = collections.Counter()
    ema = {n: p.detach().clone() for n, p in names.items()}
    if resume:
        ema, step, exposures = restore(dest/'latest.pt', model, optimizer, protocol_sha, protocol['release_sha256'], work)
    elif (dest/'latest.pt').exists():
        raise FileExistsError('Existing arm needs --resume; never overwrite a run')
    else:
        save(dest/'latest.pt', model, optimizer, ema, step, work, protocol_sha, protocol['release_sha256'], exposures)
        ema, step, exposures = restore(dest/'latest.pt', model, optimizer, protocol_sha, protocol['release_sha256'], work)
    feed = data.Feed(manifest, [x for micros in work[step:] for x in micros], cfg['workers'])
    status = 'stopped'; in_optimizer = False; last_saved = step
    try:
        def diagnostic_halt():
            if STOP or dt.datetime.now(dt.timezone.utc) >= dt.datetime.fromisoformat(cfg['optimization_deadline_utc']):
                raise InterruptedError('Diagnostic reached optimization cutoff')
        import diagnose_bottom_geometry_v4_20261019 as diagnostic
        if step == 0:
            diagnostic.diagnose(model, teacher, manifest, cfg['weights'], dest/'gradients_step000000.json', diagnostic_halt)
        with (dest/'train.jsonl').open('a' if resume else 'x', buffering=1) as log:
            while step < len(work):
                now = dt.datetime.now(dt.timezone.utc)
                if STOP or (c['output_dir']/'STOP').exists() or now >= dt.datetime.fromisoformat(cfg['optimization_deadline_utc']):
                    break
                start = time.perf_counter(); boundary_rng = rt.geo.old.rng_state()
                warmup = cfg['warmup_updates']
                progress = max(0., (step-warmup)/max(1, len(work)-warmup))
                factor = min(1., (step+1)/warmup)*(.1+.9*(1+math.cos(math.pi*min(progress, 1)))/2)
                for pg in optimizer.param_groups:
                    pg['lr'] = pg['base_lr']*factor
                try:
                    loss, terms = backward(model, teacher, feed, work[step], cfg['weights'], cfg['head_batch_cap'], cfg['real_anchor'])
                except BaseException:
                    state.restore_rng(boundary_rng); raise
                norm = torch.nn.utils.clip_grad_norm_(list(names.values()), cfg['clip_norm'])
                if not torch.isfinite(norm):
                    state.restore_rng(boundary_rng); raise FloatingPointError('Nonfinite complete gradient')
                in_optimizer = True; optimizer.step()
                next_step = step+1
                with torch.no_grad():
                    decay = min(.999, (next_step+1)/(next_step+10))
                    for n, v in ema.items():
                        v.lerp_(names[n].detach(), 1-decay)
                exposures.update(i['sampling_role'] for i in work[step]); step = next_step
                torch.cuda.synchronize()
                in_optimizer = False
                log.write(json.dumps(dict(step=step, loss=loss, terms=terms, gradient_norm=float(norm),
                                          clipped=bool(norm > cfg['clip_norm']), lr_factor=factor,
                                          update_seconds=time.perf_counter()-start))+'\n')
                if step % 100 == 0:
                    save(dest/'latest.pt', model, optimizer, ema, step, work, protocol_sha, protocol['release_sha256'], exposures)
                    last_saved = step
                if step % 250 == 0 or step == len(work):
                    diagnostic.diagnose(model, teacher, manifest, cfg['weights'],
                                        dest/f'gradients_step{step:06d}.json', diagnostic_halt)
            status = 'completed' if step == len(work) else 'stopped'
    except InterruptedError:
        status = 'stopped'
    except BaseException:
        status = 'failed'
        raise
    finally:
        feed.close()
        if not in_optimizer:
            save(dest/'latest.pt', model, optimizer, ema, step, work, protocol_sha, protocol['release_sha256'], exposures)
        rt.write_new(dest/f'closeout_step{step:06d}_{time.time_ns()}.json',
                     dict(status=status, step=step, last_periodic_save=last_saved,
                          checkpoint_safe=not in_optimizer, no_automatic_next_arm=True, no_sealed_evaluation=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'preflight', 'train'])
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--arm', choices=['P0', 'P1'])
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.config); return
    if not args.arm:
        parser.error('--arm is required')
    lock_path = rt.gpu_lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg = resolve(json.loads(args.config.read_text()))
        if not (cfg['output_dir']/'protocol.json').is_file():
            parser.error('Run prepare in a fresh namespace first')
        with (cfg['output_dir']/f'{args.arm}.owner.lock').open('a') as owner:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run(args.config, args.arm, args.command, args.execute, args.resume)


if __name__ == '__main__':
    def stop(*_):
        global STOP
        STOP = True
    signal.signal(signal.SIGINT, stop); signal.signal(signal.SIGTERM, stop)
    main()
