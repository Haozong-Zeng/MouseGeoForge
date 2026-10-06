import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out = root / 'output/pymousestudio_realism_cycle_20261002'
plan = json.loads((out / 'plan.json').read_text())
delivery = Path(plan['project_directory']) / 'DELIVERY'
windows = Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-RealCapture-Replica-20261002/DELIVERY')

def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

original_sha = sha(Path(plan['source_blend']))
assert original_sha == plan['source_sha256']
page = (delivery / 'index.html').read_text()
assert 'verify_package.py' not in page
page = page.replace('<a href="package_manifest.json">文件校验清单</a>', '<a href="package_manifest.json">文件校验清单</a> · <a href="verify_package.py">离线完整性校验脚本</a>')
(delivery / 'index.html').write_text(page, encoding='utf-8')
manifest = json.loads((delivery / 'package_manifest.json').read_text())
manifest['updated_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
manifest['files'] = [dict(path=p.relative_to(delivery).as_posix(), bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(delivery.rglob('*')) if p.is_file() and p.name != 'package_manifest.json']
manifest['payload_bytes'] = sum(x['bytes'] for x in manifest['files'])
(delivery / 'package_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
for relative in ['verify_package.py', 'README.txt', 'index.html', 'package_manifest.json']:
    shutil.copy2(delivery / relative, windows / relative)
results = []
for directory in [delivery, windows]:
    proc = subprocess.run([sys.executable, str(directory / 'verify_package.py')], check=True, capture_output=True, text=True)
    result = json.loads(proc.stdout)
    assert result['ok'] and result['native_views'] == 21
    results.append(dict(directory=str(directory), result=result))
assert sha(delivery / 'package_manifest.json') == sha(windows / 'package_manifest.json')
audit = dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(), original_source_sha256=original_sha, original_source_unchanged=True, checks=results)
(out / 'v069_final_integrity.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2))
state = json.loads((out / 'state.json').read_text(encoding='utf-8-sig'))
assert state['active_pid'] is None
state.update(stage='v069_final_integrity_passed_ready_for_shutdown', updated_utc=audit['utc'], package_verification=str(out / 'v069_final_integrity.json'), next_action='All delivery tasks complete; appearance frozen. At06:29 added portable stdlib verify_package.py and tested project+Windows copies:61payloadfiles,21PNG1080x1200RGBA16,49links,allSHApass,sourceunchanged. GPUlockavailable,noBlenderprocess. No further renders/downloads/experiments needed. NEXT heartbeat ~06:44 is scheduled finalwrap: readplan/state/log/sourceaudit summary,checkownedprocess/sourcehash; use app automation_update to PAUSE pymousestudio preservingfields(readTOML with tomllib), confirmPAUSEDreadback, save cycle_final_status.json/state cycle_closed BEFORE harddeadline06:59:16. Do not fabricatephotorealism orperfectcontact. Final Chinese reportlink +3staticprojects21views and residualmuzzle/furglass/footgap; notGTnotanimation. BrowserfileURLpolicyblocked,no workaround,reportimage/linkQAonly. OpenAI Docs skillread+official https://learn.chatgpt.com/docs/automations?surface=app fetched previousheartbeat; currenttoolsupportspause,toolviewonlyrenderscard. Windowsdelivery C:/Users/55395/Downloads/PyMouseStudio-RealCapture-Replica-20261002/DELIVERY/index.html. The current notification can remainquiet; no materialvisualchanges.')
(out / 'state.json').write_text(json.dumps(state, ensure_ascii=False, indent=2))
print(json.dumps(audit, ensure_ascii=False, indent=2))
