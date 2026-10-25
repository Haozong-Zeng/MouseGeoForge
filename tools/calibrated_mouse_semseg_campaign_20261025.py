"""Matched L/M common TRAIN endpoint; optional F/U only after convergence."""
from __future__ import annotations
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mousegeoforge.semseg_training import digest, write_json

OUT = ROOT / 'output/calibrated_mouse_10h_20261025'
RUN = OUT / 'semseg_campaign_s17_v1'
ENGINE = 'tools/train_mouse_semseg_cycle_20261025.py'


def monitor_summary(path):
    rows = json.loads(path.read_text())
    result = {}
    for name, subset in [('all', rows), *[(f'{d}/{b}', [r for r in rows if (r['domain'], r['bottom']) == (d,b)])
                                          for d in ['real','synthetic'] for b in [True,False]]]:
        result[name] = dict(loss=float(np.mean([r['loss'] for r in subset])),
            iou=float(np.mean([r['ref512_tol2']['iou'] for r in subset])),
            boundary_f1=float(np.mean([r['ref512_tol2']['boundary_f1'] for r in subset])))
    return result


def plateau(history):
    if len(history) < 5:
        return False
    last = history[-5:]
    for group in last[0]['metrics']:
        for metric in ['loss','iou','boundary_f1']:
            values = [row['metrics'][group][metric] for row in last]
            tolerance = .01*max(1e-8, np.mean(values)) if metric=='loss' else .002
            if max(values)-min(values) > tolerance:
                return False
    return True


def run_child(args, log_name, pins):
    for path, expected in pins.items():
        if digest(ROOT/path) != expected:
            raise RuntimeError('Campaign source changed: '+path)
    with (RUN / (log_name+'.log')).open('x') as stream:
        # Inherit the campaign process group: outer watchdog owns every child.
        result = subprocess.run([sys.executable, ENGINE, *args], cwd=ROOT,
                                stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError('Failed child: '+log_name)


def main():
    if RUN.exists():
        raise FileExistsError('Do not duplicate/restart the registered campaign')
    RUN.mkdir()
    plan = json.loads((OUT/'plan.json').read_text())
    cutoff = dt.datetime.fromisoformat(plan['optimization_cutoff_utc'])
    sources = [ENGINE, 'mousegeoforge/models/mouse_semseg.py', 'mousegeoforge/semseg_training.py',
               Path(__file__).relative_to(ROOT).as_posix()]
    pins = {p:digest(ROOT/p) for p in sources}
    protocol = dict(seed=17, side=512, global_batch=4, micro_batch=1, min_head_steps=6000,
        max_head_steps=18000, monitor_interval=500, common_lr_schedule_steps=18000,
        plateau_checks=5, loss_relative_span=.01, iou_boundary_span=.002,
        convergence_all_and_each_stratum=True, training_only_stopping=True,
        same_initialization_and_exposure=True, primary='common endpoint EMA',
        optional_FU_requires_both_heads_plateau=True, optional_FU_steps=6000,
        no_dev_checkpoint_selection=True, source_sha256=pins,
        items_sha256=digest(OUT/'semseg_items.json'),
        source_limited='synthetic six-pose V5, real manual sessions; no sealed; no actual DA encoder')
    write_json(RUN/'protocol.json', protocol)
    base = ['--items', str(OUT/'semseg_items.json'), '--side','512','--global-batch','4',
            '--micro-batch','1','--workers','2','--seed','17','--device','cuda','--precision','bf16']
    history = {'L': [], 'M': []}; parents = {}; endpoint = 0; converged = False
    pair_seconds = None
    for stop in range(500,18001,500):
        remaining = (cutoff-dt.datetime.now(dt.timezone.utc)).total_seconds()
        # Keep time for both arms and evaluation; never call an unmatched prefix a pair.
        if remaining < max(1800, (pair_seconds or 300)*1.4+900):
            break
        began = time.monotonic()
        for arm in ['L','M']:
            folder = RUN/f'{arm}_{stop:06d}'
            args = ['train',*base,'--arm',arm,'--steps','18000','--stop-after',str(stop),
                    '--save-every','500','--eval-every','500','--run-dir',str(folder),
                    '--allow-unregistered-fixed-budget']
            if arm in parents:
                args += ['--resume', str(parents[arm])]
            run_child(args, f'{arm}_{stop:06d}', pins)
            parents[arm] = folder/f'step_{stop:06d}.pt'
            history[arm].append(dict(step=stop, metrics=monitor_summary(folder/f'train_monitor_{stop:06d}.json')))
        pair_seconds = time.monotonic()-began
        endpoint = stop
        converged = stop>=6000 and all(plateau(history[a]) for a in ['L','M'])
        write_json(RUN/'progress.json',dict(common_endpoint=endpoint, converged=converged,
            last_pair_seconds=pair_seconds, history=history, parents={a:str(p) for a,p in parents.items()}))
        print('COMMON_HEAD',endpoint,'converged',converged,'pair_seconds',pair_seconds,flush=True)
        if converged:
            break
    if not endpoint:
        write_json(RUN/'completion.json',dict(status='insufficient_budget',head_converged=False))
        return
    for arm in ['L','M']:
        run_child(['evaluate',*base,'--arm',arm,'--steps','18000','--resume',str(parents[arm]),
                   '--run-dir',str(RUN/f'{arm}_dev')],arm+'_dev',pins)
    fu_completed = False
    remaining = (cutoff-dt.datetime.now(dt.timezone.utc)).total_seconds()
    # Adapted encoder is substantially more expensive: admission is conservative.
    if converged and remaining > max(3600, (pair_seconds or 300)*24):
        for arm in ['F','U']:
            folder = RUN/f'{arm}_006000'
            run_child(['train',*base,'--arm',arm,'--steps','6000','--parent',str(parents['M']),
                       '--run-dir',str(folder),'--allow-unregistered-fixed-budget'],arm+'_006000',pins)
            run_child(['evaluate',*base,'--arm',arm,'--steps','6000','--resume',str(folder/'step_006000.pt'),
                       '--run-dir',str(RUN/f'{arm}_dev')],arm+'_dev',pins)
        fu_completed = True
    write_json(RUN/'completion.json',dict(status='completed',common_head_endpoint=endpoint,
        head_converged=converged, stopped_at_max=endpoint==18000, F_U_completed=fu_completed,
        primary_segmentation_masks=str(RUN/'M_dev/masks_ema'),
        depth_model_selected_by_segmentation=False,
        checkpoint_paths={a:str(p) for a,p in parents.items()}, history=history))


if __name__ == '__main__':
    main()
