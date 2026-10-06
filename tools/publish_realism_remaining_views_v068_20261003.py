import datetime as dt
import hashlib
import json
import shutil
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlparse
import cv2
import numpy as np

root = Path(__file__).resolve().parents[1]
base = root / 'output/pymousestudio_realism_cycle_20261002'
out = base / 'v068_remaining_views'
plan = json.loads((base / 'plan.json').read_text())
delivery = Path(plan['project_directory']) / 'DELIVERY'
windows = Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-RealCapture-Replica-20261002/DELIVERY')
manifest = json.loads((out / 'manifest.json').read_text())
assert manifest['complete'] and all(x['blend_unchanged'] for x in manifest['poses'])
new_files = []
checks = []
for pose in ['climb', 'quadruped', 'headraise']:
    for c in 'CEG':
        src = out / pose / f'cam{c}.png'
        im = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
        assert im.shape == (1200, 1080, 4) and im.dtype == np.uint16
        checks.append(dict(pose=pose, camera=c, shape=list(im.shape), dtype=str(im.dtype)))
        relative = f'renders/{pose}/cam{c}.png'
        assert not (delivery / relative).exists()
        shutil.copy2(src, delivery / relative)
        new_files.append(relative)
for src, relative in [(out / 'remaining_views_contact.jpg', 'comparisons/remaining_views.jpg'), (out / 'manifest.json', 'evidence/remaining_views_manifest.json')]:
    shutil.copy2(src, delivery / relative)
    new_files.append(relative)

review = dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(), appearance_changed=False, rendered_from='Unmodified Windows delivery files using shipped render_views.py', render_checks=checks, visual_checks={
    'all_nine_images': 'Reviewed 3x3 contact sheet',
    'additional_native_checks': ['climb C', 'climb G', 'quadruped C', 'headraise G'],
    'findings': [
        'No gross whole-body displacement or missing image assets in these views; this is not an exhaustive collision certificate.',
        'Climb C/G show residual hind-foot contact gap; not accepted as perfectly grounded.',
        'Elongated/stylized muzzle, broad highlights on pale paws/tail and overly clean apparatus remain.',
        'HeadRaise G exposes a bright implant face as well as the already known endpoint issue in other views.',
        'HeadRaise has a lifted forepaw; static support/contact is not equivalent to all four soles flat.',
    ],
    'decision': 'Publish additional diagnostic views, retaining all limitations. No appearance edits or claim of photorealistic acceptance.',
})
(delivery / 'evidence/remaining_views_review.json').write_text(json.dumps(review, ensure_ascii=False, indent=2))
new_files.append('evidence/remaining_views_review.json')
page_path = delivery / 'index.html'
page = page_path.read_text()
assert 'id="remaining-views"' not in page
section = '<section id="remaining-views"><h2>补充侧面 C / E / G：现在共21张原分辨率合成图</h2><p>三个未改动的交付工程各补渲染三个侧面，1080×1200 / 128采样 / OptiX；使用随附脚本，工程SHA不变。</p><a href="comparisons/remaining_views.jpg"><img src="comparisons/remaining_views.jpg" alt="三个姿态C E G视角"></a><p>'
for pose, chinese in [('climb', '攀爬'), ('quadruped', '四足'), ('headraise', '抬头')]:
    section += chinese + '：' + ' · '.join(f'<a href="renders/{pose}/cam{c}.png">{c}原图</a>' for c in 'CEG') + '　'
section += '</p><p>新增视角再次揭示残余问题：攀爬C/G中后足贴地仍有间隙，爪和尾部高光偏硬，口鼻偏长；抬头G中装置出现亮白面。抬头姿态还有抬起的前足，不能把静态支撑理解为四只脚全部贴平。未发现整体姿态错位或素材缺失，但不构成无穿模、照片级真实的验收。</p><p><a href="evidence/remaining_views_review.json">逐项复核记录</a> · <a href="evidence/remaining_views_manifest.json">未改动工程与渲染记录</a></p></section>'
page = page.replace('完成12视角GPU渲染验证', '完成21视角GPU渲染验证')
page = page.replace('A底部，B/D/F侧面', 'A底部，B–G侧面')
page = page.replace('<section><h2>预留真实帧', section + '<section><h2>预留真实帧')
page_path.write_text(page, encoding='utf-8')

def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

assert sha(Path(plan['source_blend'])) == plan['source_sha256']
package = json.loads((delivery / 'package_manifest.json').read_text())
package['updated_utc'] = review['utc']
package['files'] = [dict(path=p.relative_to(delivery).as_posix(), bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(delivery.rglob('*')) if p.is_file() and p.name != 'package_manifest.json']
package['payload_bytes'] = sum(x['bytes'] for x in package['files'])
(delivery / 'package_manifest.json').write_text(json.dumps(package, ensure_ascii=False, indent=2))
for relative in new_files + ['index.html', 'package_manifest.json']:
    shutil.copy2(delivery / relative, windows / relative)
for entry in package['files']:
    assert sha(windows / entry['path']) == entry['sha256'], entry['path']
assert sha(delivery / 'package_manifest.json') == sha(windows / 'package_manifest.json')

class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.links=[]
    def handle_starttag(self, tag, attrs):
        self.links.extend(v for k,v in attrs if k in ('href','src'))
parser=Links();parser.feed(page)
for folder in [delivery,windows]:
    for link in parser.links:
        if not urlparse(link).scheme:
            assert (folder / unquote(link.split('#')[0])).is_file(), link
verification = dict(utc=review['utc'], payload_files=len(package['files']), payload_bytes=package['payload_bytes'], local_and_windows_sha_match=True, links_verified_per_copy=len(parser.links), total_native_renders=21, source_unchanged=True)
(base / 'v068_delivery_verification.json').write_text(json.dumps(verification, indent=2))
state = json.loads((base / 'state.json').read_text(encoding='utf-8-sig'))
state.update(stage='v068_all_seven_views_verified_and_delivered', active_pid=None, active_job=None, updated_utc=review['utc'], latest_review=str(delivery / 'evidence/remaining_views_review.json'), package_verification=str(base / 'v068_delivery_verification.json'), next_action='Appearance FROZEN. Windows/project DELIVERY now all21 native renders (3poses xA-G),3blendfiles unchanged; helperOptiX tested. v068 nine CEG rendered, contact sheet all9+nativeclimbC/G quadC headG visually inspected. Residuals:climbhindpawgap,hardpaw/tailhighlights,stylizedlongmuzzle,headraiseGbrightimplantface; reportexplicitnotphotorealisticacceptance. FullSHA+linkchecks pass; noactiveBlender. Browserlocalfilepolicyblock remains no workaround. Current~06:21; harddeadline06:59:16. Next heartbeat final source/artifact integrity and optional handoff-summary only, no new renders/appearance/experiment. At or after06:44 begin shutdown and pauseautomation via app tool bydeadline. Read openai-docsskill (already read+officialdocs fetched thisturn), preserveautomation fields viareadTOML andtoolupdate; ACTIVE every15UNTIL07:14fallback. Original/dataset/training unchanged.')
(base / 'state.json').write_text(json.dumps(state, ensure_ascii=False, indent=2))
for folder in [base, windows.parent]:
    p=folder/'index.html'
    current=p.read_text()
    href='DELIVERY/index.html' if folder==windows.parent else '../../BlenderPyMouseProjects/Cylinder/PyMouseStudio-RealCapture-Replica-20261002/DELIVERY/index.html'
    marker='<p id="delivery-v068"><a href="'+href+'">最终交付：3个工程、21张原分辨率图及真实对照报告</a></p>'
    if 'id="delivery-v068"' not in current:p.write_text(marker+current,encoding='utf-8')
print(json.dumps(verification, indent=2))
