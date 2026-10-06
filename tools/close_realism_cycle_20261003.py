"""Record bounded-cycle handoff after the app has paused its heartbeat."""
import datetime as dt
import fcntl
import hashlib
import json
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out = root / 'output/pymousestudio_realism_cycle_20261002'
plan = json.loads((out / 'plan.json').read_text())
state = json.loads((out / 'state.json').read_text(encoding='utf-8-sig'))
cfg = tomllib.loads(Path('/mnt/c/Users/55395/.codex/automations/pymousestudio/automation.toml').read_text())
assert cfg['id'] == 'pymousestudio' and cfg['status'] == 'PAUSED'
assert state['active_pid'] is None
now = dt.datetime.now(dt.timezone.utc)
assert now < dt.datetime.fromisoformat(plan['deadline_utc'])
with (root / 'output/teacher_temporal_cycle_20261001/gpu_job.lock').open('a+') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    fcntl.flock(lock, fcntl.LOCK_UN)
proc = subprocess.run(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe', '-NoProfile', '-Command', '@(Get-Process blender -ErrorAction SilentlyContinue).Count'], check=True, capture_output=True, text=True)
assert proc.stdout.strip() == '0', proc.stdout

def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

assert sha(Path(plan['source_blend'])) == plan['source_sha256']
delivery = Path(plan['project_directory']) / 'DELIVERY'
windows = Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-RealCapture-Replica-20261002/DELIVERY')
status = dict(
    cycle_status='CLOSED_WITH_DOCUMENTED_LIMITATIONS',
    closed_utc=now.isoformat(), start_utc=plan['start_utc'], planned_deadline_utc=plan['deadline_utc'],
    elapsed_hours=(now-dt.datetime.fromisoformat(plan['start_utc'])).total_seconds()/3600,
    closed_during_reserved_final_wrap=True, automation_id='pymousestudio', automation_status='PAUSED',
    original_source_unchanged=True, original_source_sha256=plan['source_sha256'], blender_processes=0, shared_gpu_lock_available=True,
    deliverables=dict(static_blends=3, native_views=21, cameras='ABCDEFG', resolution=[1080,1200], png_bits=16, windows_directory='C:/Users/55395/Downloads/PyMouseStudio-RealCapture-Replica-20261002/DELIVERY'),
    objective_assessment='Editable real-capture-style replica and three representative static poses delivered. Photorealism and exact real-video pose reconstruction NOT achieved.',
    remaining_limitations=['Stylized elongated muzzle', 'Fur microstructure/highlights and skin response differ from real capture', 'Some climb hind-foot contact gaps', 'Bright implant faces/endpoints and hard paw/tail highlights', 'Glass/background/hardware too clean and simplified', 'Not continuous animation, not reliable real-video 3D ground truth', 'Local-browser preview blocked by tool policy; image visual QA and programmatic link checks performed'],
    next_work_requires_new_request=True,
)
(delivery / 'evidence/cycle_final_status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2))
page = (delivery / 'index.html').read_text()
assert 'id="cycle-closed"' not in page
banner = '<p class="notice" id="cycle-closed">本轮已在预留收尾阶段结束（'+now.strftime('%Y-%m-%d %H:%M UTC')+'），定时任务已暂停，无遗留Blender渲染进程。复刻工程已交付，但照片级真实和精确姿态重建未达成；后续继续需另行安排。<a href="evidence/cycle_final_status.json">周期收尾记录</a></p>'
page = page.replace('<h1>PyMouseStudio · RealCapture Replica</h1>', '<h1>PyMouseStudio · RealCapture Replica</h1>'+banner)
(delivery / 'index.html').write_text(page, encoding='utf-8')
manifest = json.loads((delivery / 'package_manifest.json').read_text())
manifest['updated_utc'] = now.isoformat()
manifest['files'] = [dict(path=p.relative_to(delivery).as_posix(), bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(delivery.rglob('*')) if p.is_file() and p.name != 'package_manifest.json']
manifest['payload_bytes'] = sum(x['bytes'] for x in manifest['files'])
(delivery / 'package_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
for relative in ['evidence/cycle_final_status.json', 'index.html', 'package_manifest.json']:
    shutil.copy2(delivery / relative, windows / relative)
checks=[]
for folder in [delivery,windows]:
    result=subprocess.run([sys.executable,str(folder/'verify_package.py')],check=True,capture_output=True,text=True)
    checks.append(json.loads(result.stdout))
assert sha(delivery / 'package_manifest.json') == sha(windows / 'package_manifest.json')
status['final_package_checks']=checks
(out / 'cycle_final_status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2))
state.update(stage='cycle_closed_with_documented_limitations',active_pid=None,active_job=None,updated_utc=now.isoformat(),closed_utc=now.isoformat(),automation_status='PAUSED',cycle_final_status=str(out/'cycle_final_status.json'),next_action='CYCLE CLOSED. pymousestudio automation confirmed PAUSED through app tool and TOML readback. No Blender process; shared GPU lock available; original source unchanged. Three static delivery blends,21native views,report+scripts in project/DELIVERY andWindowsDownloads. All finalchecks pass. Do not auto-resume, extend, train, render, download or change artifacts without a new user request. Photorealism/exactposeGT not achieved; documentedlimitations remain.')
(out / 'state.json').write_text(json.dumps(state,ensure_ascii=False,indent=2))
print(json.dumps(status,ensure_ascii=False,indent=2))
