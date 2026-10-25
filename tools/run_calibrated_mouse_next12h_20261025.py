"""Finite local queue with independent wall-clock watchdog; never resets the budget."""
from __future__ import annotations
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/calibrated_mouse_next12h_20261025'


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def now():
    return dt.datetime.now(dt.timezone.utc)


def stamp():
    return now().isoformat()


def identity(pid):
    stat = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()
    return dict(pid=pid, start_ticks=stat[19], pgid=os.getpgid(pid),
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                command=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode().strip())


def matches(saved):
    try:
        return identity(saved['pid']) == saved
    except (FileNotFoundError, ProcessLookupError):
        return False


def stop(saved):
    if saved and matches(saved):
        os.killpg(saved['pgid'], signal.SIGTERM)
        for _ in range(15):
            if not matches(saved):
                return
            time.sleep(1)
        if matches(saved):
            os.killpg(saved['pgid'], signal.SIGKILL)


def watchdog(plan):
    cutoff = dt.datetime.fromisoformat(plan['optimization_cutoff_utc'])
    hard = dt.datetime.fromisoformat(plan['hard_finish_utc'])
    while now() < hard:
        child = read(OUT / 'active_child.json', {})
        if now() >= cutoff and child.get('optimization'):
            stop(child.get('identity'))
        if (OUT / 'completion.json').exists():
            return
        time.sleep(3)
    child = read(OUT / 'active_child.json', {})
    stop(child.get('identity'))
    stop(read(OUT / 'controller_process.json'))
    write(OUT / 'watchdog_closeout.json', dict(time=stamp(), hard_finish_reached=True,
        child_still_matches=bool(child.get('identity') and matches(child['identity']))))


def controller(plan):
    singleton = (OUT / 'controller.lock').open('a+')
    fcntl.flock(singleton, fcntl.LOCK_EX | fcntl.LOCK_NB)
    write(OUT / 'controller_process.json', identity(os.getpid()))
    log = (OUT / 'logs/watchdog.log').open('a')
    watcher = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--watch', '--output', str(OUT)],
        cwd=ROOT, start_new_session=True, stdout=log, stderr=subprocess.STDOUT)
    write(OUT / 'watchdog_process.json', identity(watcher.pid))
    hard = dt.datetime.fromisoformat(plan['hard_finish_utc'])
    cutoff = dt.datetime.fromisoformat(plan['optimization_cutoff_utc'])
    state = read(OUT / 'state.json', dict(jobs={}, started=stamp()))
    while now() < hard:
        if (OUT / 'STOP').exists():
            break
        queue = read(OUT / 'queue.json', [])
        pending = [job for job in queue if job['id'] not in state['jobs']]
        if not pending:
            state.update(time=stamp(), status='waiting_for_next_registered_stage')
            write(OUT / 'state.json', state)
            if (OUT / 'QUEUE_CLOSED').exists():
                break
            time.sleep(5)
            continue
        job = pending[0]
        depends = job.get('depends', [])
        failed = [key for key in depends if state['jobs'].get(key, {}).get('status') != 'completed']
        if failed:
            state['jobs'][job['id']] = dict(status='skipped_dependency', dependencies=failed, time=stamp())
            write(OUT / 'state.json', state)
            continue
        end = min(hard, cutoff if job.get('optimization', True) else hard)
        remaining = (end - now()).total_seconds()
        if remaining < job.get('minimum_seconds', 120):
            state['jobs'][job['id']] = dict(status='skipped_budget', time=stamp())
            write(OUT / 'state.json', state)
            continue
        if (OUT / 'STOP').exists():
            break
        for relative, expected in job['source_sha256'].items():
            actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            if actual != expected:
                raise RuntimeError('Pinned job source changed: ' + relative)
        gpu_handle = None
        if job.get('gpu_lock_external', False):
            gpu_handle = (ROOT / plan['gpu_lock']).open('a+')
            while True:
                try:
                    fcntl.flock(gpu_handle, fcntl.LOCK_EX | fcntl.LOCK_NB); break
                except BlockingIOError:
                    if now() >= end or (OUT / 'STOP').exists():
                        gpu_handle.close(); gpu_handle = None; break
                    state.update(time=stamp(),status='waiting_for_shared_GPU',active_job=job['id'])
                    write(OUT / 'state.json',state); time.sleep(3)
            if gpu_handle is None:
                state['jobs'][job['id']] = dict(status='skipped_gpu_budget',time=stamp())
                write(OUT / 'state.json',state); continue
        cmd = [sys.executable, *job['args']]
        write(OUT / 'jobs' / (job['id'] + '.json'), dict(**job, registered=stamp()))
        with (OUT / 'logs' / (job['id'] + '.log')).open('x') as stream:
            child = subprocess.Popen(cmd, cwd=ROOT, start_new_session=True,
                stdout=stream, stderr=subprocess.STDOUT,
                env={**os.environ, 'OMP_NUM_THREADS': '4', 'PYTHONUNBUFFERED': '1'})
            child_id = identity(child.pid)
            write(OUT / 'active_child.json', dict(job=job['id'], identity=child_id,
                optimization=job.get('optimization', True)))
            state.update(time=stamp(), status='running', active_job=job['id'])
            write(OUT / 'state.json', state)
            began = time.monotonic()
            reason = None
            while child.poll() is None:
                if now() >= end or time.monotonic() - began >= job.get('timeout_seconds', remaining):
                    reason = 'cutoff_or_timeout'; stop(child_id)
                if (OUT / 'STOP').exists():
                    reason = 'stop_requested'; stop(child_id)
                time.sleep(1)
            result = dict(status='completed' if child.returncode == 0 and not reason else 'failed',
                          returncode=child.returncode, reason=reason, time=stamp(),
                          seconds=time.monotonic() - began)
            state['jobs'][job['id']] = result
            state.pop('active_job', None)
            write(OUT / 'jobs' / (job['id'] + '_result.json'), result)
            write(OUT / 'state.json', state)
            write(OUT / 'active_child.json', {})
        if gpu_handle is not None:
            fcntl.flock(gpu_handle, fcntl.LOCK_UN); gpu_handle.close()
    write(OUT / 'completion.json', dict(time=stamp(), state=state,
        active_child=read(OUT / 'active_child.json', {}),
        reason='queue_closed_or_stop' if now() < hard else 'hard_finish'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output.resolve()
    plan = read(OUT / 'plan.json')
    if args.watch:
        watchdog(plan)
    else:
        controller(plan)
