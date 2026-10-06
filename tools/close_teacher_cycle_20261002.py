"""Deadline closeout only; verify idle/paused and update derived handoff status."""
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import tomllib
from run_teacher_cycle_dlc_20261001 import OUT, write_state

plan=json.loads((OUT/'plan.json').read_text())
now=dt.datetime.now(dt.timezone.utc)
assert now>=dt.datetime.fromisoformat(plan['deadline_utc'])
automation=tomllib.loads(Path('/mnt/c/Users/55395/.codex/automations/mousegeoforge/automation.toml').read_text())
assert automation['status']=='PAUSED' and automation['id']=='mousegeoforge'
lock=(OUT/'gpu_job.lock').open('a+')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
assert not gpu, 'Unexpected GPU task: inspect ownership, do not kill arbitrary user work'
owned=[]
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
    try:args=(proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
    except (FileNotFoundError,PermissionError,ProcessLookupError):continue
    if not args or 'python' not in Path(args[0]).name:continue
    if any('/tools/' in a and ('20261001' in a or '20261002' in a) for a in args):
        owned.append(dict(pid=int(proc.name),args=args))
assert not owned, owned
record=dict(closed_utc=now.isoformat(),authorized_deadline_utc=plan['deadline_utc'],automation_id='mousegeoforge',automation_status='PAUSED',
    own_jobs_remaining=owned,gpu_jobs=gpu,shared_gpu_lock_free=True,stop_sentinel=True,
    new_experiments_after_deadline=False,processes_killed=0,original_files_deleted=0,pseudo_gt_released=False,
    outcome='Cycle ended with saved diagnostics, not a validated production3Dlabeling pipeline. No furtherautomaticwork.',
    unresolved=['Persistentpawidentityerrors','Highabsoluteheldviewreprojectionerror','Single-seedtemporalablation onreuseddevelopmentvideos','No densemotion3Dtruth','No downstreamfoundationmodelablation','YOLO/tailcurve route notimplementedthiscycle'])
(OUT/'closeout.json').write_text(json.dumps(record,indent=2))
handoff=OUT/'cycle_handoff'
summary=json.loads((handoff/'summary.json').read_text())
summary['status']='closed; automation PAUSED; GPUidle; no newexperiments';summary['closeout']=record
(handoff/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
page=(handoff/'index.html').read_text()
page=page.replace('此页面是交付汇总，不代表定时器已暂停。','本周期已结束：自动化已暂停，GPU空闲，不再继续实验。')
(handoff/'index.html').write_text(page)
export=Path('/mnt/c/Users/55395/Downloads/MouseGeoForge-TeacherTemporal-20261001/cycle_handoff')
for name in ['summary.json','index.html']:shutil.copy2(handoff/name,export/name)
shutil.copy2(OUT/'closeout.json',export/'closeout.json')
write_state(stage='cycle_closed',pid=None,active_run=None,mode='stopped',job_deadline_utc=None,
    closed_utc=now.isoformat(),automation_status='PAUSED',latest_result=str(handoff/'summary.json'),
    latest_decision=str(OUT/'closeout.json'),next_action='Wait for explicit user direction. Do not resume expired cycle.',pseudo_gt_released=False)
print(json.dumps(record,indent=2))
