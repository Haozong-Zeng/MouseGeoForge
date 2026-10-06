"""Close the bounded ear study without touching its immutable source projects."""
import datetime as dt
import fcntl
import hashlib
import json
import shutil
import subprocess
import tomllib
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlparse

root = Path(__file__).resolve().parents[1]
base = root / 'output/pymousestudio_ear_cycle_20261003'
plan = json.loads((base / 'plan.json').read_text())
state = json.loads((base / 'state.json').read_text())
automation = tomllib.loads(Path('/mnt/c/Users/55395/.codex/automations/pymousestudio/automation.toml').read_text())
assert automation['id'] == 'pymousestudio' and automation['status'] == 'PAUSED'
assert state['active_pid'] is None and state['active_job'] is None
with (root / 'output/teacher_temporal_cycle_20261001/gpu_job.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    fcntl.flock(lock, fcntl.LOCK_UN)
result = subprocess.run(['powershell.exe', '-NoProfile', '-Command', '@(Get-Process blender -ErrorAction SilentlyContinue).Count'], capture_output=True, text=True, check=True)
assert result.stdout.strip() == '0', result.stdout
now = dt.datetime.now(dt.timezone.utc)
assert now < dt.datetime.fromisoformat(plan['deadline_utc'])
delivery = Path(state['delivery'])
windows = Path(state['windows_delivery'])

def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

for pose, path in plan['sources'].items():
    assert sha(Path(path)) == plan['source_sha256'][pose]
status = dict(start_utc=plan['start_utc'], deadline_utc=plan['deadline_utc'], closed_utc=now.isoformat(),
              status='COMPLETED_WITH_LIMITATIONS', automation_status='PAUSED', active_blender_processes=0,
              candidate_count=9, delivered_blends=3, native_render_count=12, resolution=[1080, 1200],
              selected_change='Ear material slot 4 neutral linear Base Color 0.12 to 0.60 only',
              original_source_sha256_unchanged=True, geometry_and_other_materials_unchanged=True,
              limitations=['Qualitative grayscale capture appearance fit, not measured spectral reflectance.',
                           'Ear shape, folding, thickness and root fur remain imperfect.',
                           'Three static poses, not animation or ground-truth reconstruction.'],
              external_assets_downloaded=False, private_data_uploaded=False)
(delivery / 'evidence/cycle_final_status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2))
page = delivery / 'index.html'
html = page.read_text()
assert 'id="cycle-closed"' not in html
section = '<section id="cycle-closed"><h2>本轮已完成并暂停</h2><p>耳部短周期实验已结束，自动任务已暂停，无遗留渲染进程。九组材质候选后选用仅提亮耳部基础色的方案；三种静态姿态、四个视角验证完毕。原始工程未覆盖。耳形、折叠和厚度仍有差异，不能视为完整真实感复刻。</p><p><a href="evidence/cycle_final_status.json">收尾记录</a></p></section>'
page.write_text(html.replace('</main>', section + '</main>'), encoding='utf-8')
manifest_path = delivery / 'package_manifest.json'
manifest = json.loads(manifest_path.read_text())
manifest['updated_utc'] = now.isoformat()
manifest['files'] = [dict(path=p.relative_to(delivery).as_posix(), bytes=p.stat().st_size, sha256=sha(p))
                     for p in sorted(delivery.rglob('*')) if p.is_file() and p != manifest_path]
manifest['payload_bytes'] = sum(item['bytes'] for item in manifest['files'])
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
for relative in ['index.html', 'package_manifest.json', 'evidence/cycle_final_status.json']:
    shutil.copy2(delivery / relative, windows / relative)
for entry in manifest['files']:
    assert sha(windows / entry['path']) == entry['sha256'], entry['path']

class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.items = []
    def handle_starttag(self, tag, attrs):
        self.items.extend(value for key, value in attrs if key in ('href', 'src'))

links = Links()
links.feed((windows / 'index.html').read_text())
for link in links.items:
    if not urlparse(link).scheme:
        assert (windows / unquote(urlparse(link).path)).is_file(), link
status.update(payload_files=len(manifest['files']), payload_bytes=manifest['payload_bytes'],
              windows_hashes_verified=True, local_links_checked=len(links.items))
(base / 'cycle_final_status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2))
state.update(stage='ear_cycle_closed', automation_status='PAUSED', updated_utc=now.isoformat(),
             active_job=None, active_pid=None, final_status=str(base / 'cycle_final_status.json'),
             next_action='Completed. Do not resume automatically; await a new user request. Original projects and datasets remain immutable.')
(base / 'state.json').write_text(json.dumps(state, ensure_ascii=False, indent=2))
print(json.dumps(status, ensure_ascii=False, indent=2))
