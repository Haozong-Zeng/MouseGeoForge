"""Observe owned process groups, enforce cutoffs even if a wrapper disappears."""
from __future__ import annotations
import datetime as dt
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from run_calibrated_mouse_resume8h_20261025 import OUT, ROOT, identity, matches, read, write


def remember_group(leader):
    if not leader or not matches(leader):
        return []
    result=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:
            member=identity(int(path.name))
            if member['pgid']==leader['pgid']:result.append(member)
        except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    return result


def stop_owned_group(members):
    alive=[member for member in members if matches(member)]
    if not alive:return
    groups={member['pgid'] for member in alive}
    if os.getpgrp() in groups:raise RuntimeError('Guardian must never stop its own group')
    for group in groups:
        try:os.killpg(group,signal.SIGTERM)
        except ProcessLookupError:pass
    for _ in range(15):
        if not any(matches(member) for member in alive):return
        time.sleep(.2)
    for member in alive:
        if matches(member):
            try:os.killpg(member['pgid'],signal.SIGKILL)
            except ProcessLookupError:pass


def main():
    import fcntl
    singleton=(OUT/'guardian.lock').open('a+')
    fcntl.flock(singleton,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=read(OUT/'plan.json');cutoff=dt.datetime.fromisoformat(plan['optimization_cutoff_utc']);hard=dt.datetime.fromisoformat(plan['hard_finish_utc'])
    write(OUT/'guardian_process.json',identity(os.getpid()))
    tracked={};last_report=0
    while True:
        now=dt.datetime.now(dt.timezone.utc);child=read(OUT/'active_child.json',{})
        if child.get('identity'):
            job=child['job'];old=tracked.setdefault(job,dict(optimization=child.get('optimization',True),members={}))
            for member in remember_group(child['identity']):old['members'][str(member['pid'])]=member
        if now>=cutoff:
            for job,value in tracked.items():
                if value['optimization'] or now>=hard:stop_owned_group(list(value['members'].values()))
        write(OUT/'guardian_status.json',dict(time=now.isoformat(),tracked=tracked,optimization_closed=now>=cutoff))
        if time.monotonic()-last_report>60 or now>=hard:
            with (OUT/'logs/guardian_reports.log').open('a') as stream:
                try:subprocess.run([sys.executable,'tools/report_calibrated_mouse_resume8h_20261025.py'],cwd=ROOT,
                        stdout=stream,stderr=subprocess.STDOUT,timeout=60)
                except subprocess.TimeoutExpired:pass
            last_report=time.monotonic()
        if now>=hard:
            controller=read(OUT/'controller_process.json')
            if controller:stop_owned_group([controller])
            write(OUT/'guardian_completion.json',dict(time=dt.datetime.now(dt.timezone.utc).isoformat(),
                hard_finish=True,remaining_owned_members=[m for value in tracked.values() for m in value['members'].values() if matches(m)]))
            break
        if (OUT/'completion.json').exists() and not read(OUT/'active_child.json',{}):break
        time.sleep(3)


if __name__=='__main__':main()
