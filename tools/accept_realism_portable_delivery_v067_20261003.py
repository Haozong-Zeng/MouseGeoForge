import datetime as dt
import hashlib
import json
import shutil
from pathlib import Path
import cv2
import numpy as np

root = Path(__file__).resolve().parents[1]
out = root / 'output/pymousestudio_realism_cycle_20261002'
plan = json.loads((out / 'plan.json').read_text())
delivery = Path(plan['project_directory']) / 'DELIVERY'
windows = Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-RealCapture-Replica-20261002/DELIVERY')
smoke = json.loads((out / 'v067_portable_helper/result.json').read_text())
assert smoke['complete'] and smoke['overwrite_refused'] and smoke['blend_unchanged_after_render']
a = cv2.imread(str(out / 'v064_final_delivery/headraise/camF.png'), cv2.IMREAD_UNCHANGED)
b = cv2.imread(str(out / 'v067_portable_helper/actual_render/camF.png'), cv2.IMREAD_UNCHANGED)
assert a.shape == b.shape == (1200, 1080, 4) and a.dtype == b.dtype == np.uint16
diff = np.abs(a[:, :, :3].astype(np.int32) - b[:, :, :3].astype(np.int32)) / 257
metrics = dict(rgb_mean_difference_gray8=float(diff.mean()), max_difference_gray8=float(diff.max()), pixels_over_1gray=int((diff.max(axis=2)>1).sum()))
assert metrics['rgb_mean_difference_gray8'] < .02
review = dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(), smoke_test=smoke, render_difference=metrics, visual_review={
    'boards_reviewed': ['climb real/synth ABDF', 'quadruped real/synth ABDF', 'headraise real/synth ABDF (real B unavailable)'],
    'native_helper_render_reviewed': 'HeadRaise F 1080x1200',
    'assessment': 'Still recognizably synthetic: smoother/darker coat with fewer irregular bright strands; stylized elongated muzzle; simpler, cleaner apparatus and glass; poses are representative, not per-frame reconstruction.',
    'layout_qa': 'Browser tool policy blocked local file URL. No workaround attempted. HTML paths checked programmatically and all comparison images visually reviewed; actual browser layout not verified.',
}, source_unchanged=True)
for src, dest in [(out / 'v067_portable_helper/result.json', 'portable_helper_test.json')]:
    shutil.copy2(src, delivery / 'evidence' / dest)
(delivery / 'evidence/final_visual_qa.json').write_text(json.dumps(review, ensure_ascii=False, indent=2))
p = delivery / 'index.html'
page = p.read_text()
note = '''<section id="portable-final-qa"><h2>Windows副本与交付脚本检查</h2><p>Windows下载目录的三个工程全部通过脚本检查；从该副本实际重渲染抬头F视角成功，OptiX GPU已启用。工程文件SHA未变化，已有输出目录会被拒绝覆盖。</p><p>复核对照图后，仍能明确辨认合成痕迹：毛发表面偏平滑、偏暗，缺少不规则亮毛；口鼻偏长且有风格化轮廓；器材和玻璃过于干净简化，姿态并非精确重建。此版本不能视作已消除真实域差距。</p><p>报告中的图片已视觉检查，链接已程序验证；浏览器工具限制本地file URL，因此没有完成浏览器排版验证，也未绕过限制。</p><p><a href="evidence/portable_helper_test.json">脚本测试记录</a> · <a href="evidence/final_visual_qa.json">最终视觉检查</a></p></section>'''
assert 'id="portable-final-qa"' not in page
p.write_text(page.replace('</main>', note + '</main>'), encoding='utf-8')

def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

assert sha(Path(plan['source_blend'])) == plan['source_sha256']
manifest = json.loads((delivery / 'package_manifest.json').read_text())
manifest['updated_utc'] = review['utc']
manifest['files'] = [dict(path=p.relative_to(delivery).as_posix(), bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(delivery.rglob('*')) if p.is_file() and p.name != 'package_manifest.json']
manifest['payload_bytes'] = sum(x['bytes'] for x in manifest['files'])
(delivery / 'package_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
for relative in ['index.html', 'package_manifest.json', 'evidence/portable_helper_test.json', 'evidence/final_visual_qa.json']:
    shutil.copy2(delivery / relative, windows / relative)
for entry in manifest['files']:
    assert sha(windows / entry['path']) == entry['sha256'], entry['path']
state = json.loads((out / 'state.json').read_text(encoding='utf-8-sig'))
state.update(stage='v067_portable_windows_delivery_verified', active_pid=None, active_job=None, updated_utc=review['utc'], latest_review=str(delivery / 'evidence/final_visual_qa.json'), next_action='Appearance remains FROZEN. Windows/project DELIVERY now48 payload files ~932MB,3blends12nativeviews3comparisonboards+heldout+README/license/renderhelper. All SHA match. v067 actual Windows copies all3dryruns +HeadRaiseF128OptiXrender successful; no.blendchanges; overwriteguard tested. Boards+nativehelperimage visually reviewed, residuals explicitly documented. Browser file URL security blocked; do NOT workaround or claim browser layout QA. At next heartbeat do final report/evidence consistency checks or multi-camera helper safety validation only; no new appearance experiments. Keep hard deadline06:59:16UTC, last15min complete shutdown and pause pymousestudio automation via app tool. Current~06:14,~45min left. Original/source/dataset/training unchanged.')
(out / 'state.json').write_text(json.dumps(state, ensure_ascii=False, indent=2))
print(json.dumps(dict(payload_files=len(manifest['files']), payload_bytes=manifest['payload_bytes'], render_difference=metrics, source_unchanged=True, windows_sha_all_match=True), indent=2))
