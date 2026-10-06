"""Assemble a self-contained, private review delivery; preserve source artifacts."""
import datetime as dt
import hashlib
import html
import json
import shutil
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlparse
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/pymousestudio_realism_cycle_20261002'
PLAN = json.loads((OUT / 'plan.json').read_text())
DELIVERY = Path(PLAN['project_directory']) / 'DELIVERY'
WINDOWS = Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-RealCapture-Replica-20261002/DELIVERY')
POSES = [('climb', 'Climb', '攀爬', 'FR2', 649, 31), ('quadruped', 'Quadruped', '四足站姿', 'FR2', 1104, 111), ('headraise', 'HeadRaise', '抬头', 'AZR2L2', 129, 216)]

def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def copy(src, relative):
    target = DELIVERY / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, target)
    return relative

assert sha(Path(PLAN['source_blend'])) == PLAN['source_sha256']
assert json.loads((OUT / 'v064_final_delivery/manifest.json').read_text())['complete']
font = ImageFont.truetype('/mnt/c/Windows/Fonts/arial.ttf', 23)
comparisons = []
for tag, title, chinese, session, frame, static in POSES:
    board = Image.new('RGB', (1440, 884), '#152131')
    draw = ImageDraw.Draw(board)
    for col, cam in enumerate('ABDF'):
        synth = OUT / 'v064_final_delivery' / tag / f'cam{cam}.png'
        assert Image.open(synth).size == (1080, 1200)
        copy(synth, f'renders/{tag}/cam{cam}.png')
        real = OUT / 'references' / session / f'frame_{frame:06d}' / f'cam{cam}.png'
        for row, label, path in [(0, f'REAL {session} f{frame} cam{cam}', real), (1, f'SYNTH {title} cam{cam}', synth)]:
            draw.text((col*360+8, row*442+8), label, font=font, fill='white')
            if path.exists():
                board.paste(Image.open(path).convert('RGB').resize((360, 400)), (col*360, row*442+40))
            else:
                draw.text((col*360+12, row*442+195), 'Reference unavailable', font=font, fill='#d4b484')
        if real.exists():
            copy(real, f'references/{tag}/cam{cam}.png')
    target = DELIVERY / f'comparisons/{tag}.jpg'
    target.parent.mkdir(parents=True, exist_ok=True)
    board.save(target, quality=96)
    comparisons.append(f'<section><h2>{chinese} · {session} / {frame}参考</h2><p>上排真实，下排合成；姿态仅近似，不适用于逐像素误差评分。点击查看大图。</p><a href="comparisons/{tag}.jpg"><img src="comparisons/{tag}.jpg" alt="{chinese}四视角真实和合成对照"></a><p>合成原图：'+ ' · '.join(f'<a href="renders/{tag}/cam{c}.png">cam{c}</a>' for c in 'ABDF') +'</p></section>')

heldout = Image.new('RGB', (1440, 442), '#152131')
draw = ImageDraw.Draw(heldout)
for i, c in enumerate('ABDF'):
    p = OUT / 'references/FR2/frame_001429' / f'cam{c}.png'
    copy(p, f'references/heldout_style_only/cam{c}.png')
    draw.text((i*360+8, 8), f'REAL held-out f1429 cam{c}', font=font, fill='white')
    heldout.paste(Image.open(p).convert('RGB').resize((360, 400)), (i*360, 40))
heldout.save(DELIVERY / 'comparisons/heldout_style_only.jpg', quality=96)
copy(OUT / 'v064_final_delivery/delivery_contact_sheet.jpg', 'comparisons/all_synthetic.jpg')
evidence = [
    ('v064_final_delivery/manifest.json', 'saved_reopened_manifest.json'),
    ('v064_final_delivery/pixel_verification.json', 'pixel_verification.json'),
    ('v064_final_delivery/review.json', 'visual_review.json'),
    ('v061_delivery_preflight/audit.json', 'working_file_preflight.json'),
    ('v065_repeatability/review_metrics.json', 'repeat_render_controls.json'),
    ('v065_repeatability/outlier_crops.jpg', 'repeat_render_crops.jpg'),
    ('external_mouse_asset_research.json', 'external_asset_research.json'),
]
for src, dst in evidence:
    copy(OUT / src, 'evidence/' + dst)

links = ''.join(f'<li><a href="PyMouseStudio-RealCapture-{title}-Static.blend">{chinese} .blend</a> — 仅第{static}帧，独立静态姿态</li>' for tag, title, chinese, session, frame, static in POSES)
page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PyMouseStudio 真实采集风格复刻 · 交付</title><style>
body{font:17px/1.8 system-ui,"Microsoft YaHei",sans-serif;background:#101720;color:#e7edf4;margin:0}main{max-width:1440px;padding:36px 24px;margin:auto}h1{font-size:34px}h2{font-size:24px;margin-top:36px}a{color:#8acaff}img{max-width:100%;height:auto}section{border-top:1px solid #364456;margin-top:32px;padding-top:4px}.notice{background:#273042;padding:16px;border-left:4px solid #d1a667}table{border-collapse:collapse;width:100%}td,th{text-align:left;border-bottom:1px solid #364456;padding:12px;vertical-align:top}code{color:#b9dbfd}small{color:#b7c4d4}</style><main>
<h1>PyMouseStudio · RealCapture Replica</h1><p>10小时自主迭代周期：2026-10-02 20:59 → 2026-10-03 06:59 UTC。外观已冻结，交付已重开并完成12视角GPU渲染验证。</p>
<p class="notice">这是可编辑、可复现的工作复刻版，尚非照片级真实。三种姿态不是连续动画，也不是真实视频的可靠3D ground truth。真实参考仅限项目内部复核，不可据此推定有公开再分发许可。</p>
<h2>工程与使用</h2><ul>''' + links + '''</ul><p><a href="README.txt">使用说明</a> · <a href="render_views.py">多视角重渲染脚本</a> · <a href="LICENSE_NOTES.txt">来源与许可</a> · <a href="package_manifest.json">文件校验清单</a></p>
<p>Blender 5.1.2；Cycles OptiX GPU；1080×1200；128采样；A底部，B/D/F侧面；默认F透视相机。毛发GUI隐藏、最终渲染启用。每文件132项图像文件依赖内嵌，无外链库。切换视角需同步相机保存的像素宽高比，随附脚本会自动处理。</p>
<p>几何数值按毫米组织，但scene unit_scale=1，灯光和景深包含补偿，不应将其数值当作真实瓦数、光圈或标定物理测量。</p>
<h2>本轮有效改进</h2><ul><li>接入真实矫正相机的透视标定和逐视角像素比例；统一灰度采集响应、透明圆筒与玻璃平台、背景与多方向照明。</li><li>黑色短毛、毛向及长度限制，耳/爪/尾皮肤分区；保留毛发渲染但关闭GUI预览以改善操作流畅度。</li><li>建立攀爬、四足站姿、抬头三个代表姿态，调整身体接触、足部和尾巴几何，未沿用不可信的3D标签作为真值。</li><li>加入头部装置、相机外壳、支撑与线缆代理结构；修正背景纹理连接导致的非预期强高光，加入低密度表面灰尘。</li></ul>
''' + ''.join(comparisons) + '''
<section><h2>预留真实帧：只作冻结后的风格检查</h2><p>FR2 / 1429，未用于本轮调参，不是姿态对应样本。不计算无意义的像素配准分数，也不据此声称泛化性能。</p><a href="comparisons/heldout_style_only.jpg"><img src="comparisons/heldout_style_only.jpg" alt="预留真实帧四视角"></a><p>仍可见差距：真实毛发微结构和高光更丰富，口鼻/耳/足更柔和明亮；真实玻璃划痕、污迹与器材反射更复杂。</p></section>
<section><h2>尝试后未采用的路线</h2><table><tr><th>试验</th><th>结果与取舍</th></tr><tr><td>更黑的毛发、不同毛发模型、全局gamma</td><td>未稳定改善真实感；部分设置使底部细节压黑或亮部截断，保留受控基线。</td></tr><tr><td>口鼻局部形变、凹陷填充</td><td>没有解决底部三角形口鼻观感，部分侧面更差；最终文件没有保留这些形变键。</td></tr><tr><td>ambientCG指纹PBR、强尾部环纹、背景噪声</td><td>出现云斑、规律条纹或不自然背景，未纳入最终工程。</td></tr><tr><td>头部装置关闭焦散接收、几何倒角</td><td>可消除或减弱白色端点，但物理正确性尚未验证，保留为诊断而非最终修复。</td></tr><tr><td>现成鼠体资产、图像风格迁移/生成路线</td><td>调查SMILify、replicAnt、Mouse Free Demo。未发现已验证优于当前资产且无许可/语义一致性成本的直接替代；未使用生成结果，未向外上传私有数据。</td></tr></table><p><a href="evidence/external_asset_research.json">资产调研记录</a>；具体链接、许可限制及未集成状态见<a href="LICENSE_NOTES.txt">许可说明</a>。</p></section>
<section><h2>验证与限制</h2><ul><li>三个最终.blend保存后重新打开：鼠体9984顶点前后最大位移为0，A–G相机矩阵一致，12张原分辨率图均由重开的工程渲染。</li><li>工作版与交付版RGB平均差异约0.0004–0.0053个8位灰度级；少量高光最大差异较大。未改动工程的重复渲染控制也出现相近波动，因此只能称视觉一致，不能称逐像素确定性。</li><li>皮肤顶点未发现穿越平台/圆筒边界，但此检查不是所有毛发、胡须、三角面的完整碰撞证明；部分足底贴合仍需逐帧细化。</li><li>原始Blender文件SHA256保持不变；没有修改真实标注、标定、dataset或训练模型。</li></ul><p><a href="evidence/saved_reopened_manifest.json">保存/重开记录</a> · <a href="evidence/pixel_verification.json">12视角差异</a> · <a href="evidence/repeat_render_controls.json">重复渲染控制</a> · <a href="evidence/repeat_render_crops.jpg">局部对照</a></p></section>
<section><h2>后续优先级</h2><ol><li>先重做口鼻局部拓扑与解剖轮廓，固定多视角近景检验，避免靠材质掩盖几何问题。</li><li>在可控照明下采集毛发、耳、爪和装置参考，拟合毛发微结构及皮肤响应；单独检查玻璃/MNEE近似引起的高光偏差。</li><li>实测器材尺寸与照明，补齐玻璃表面真实尺度的污迹/划痕，最后才拟合采集噪声。之后扩展更多姿态和时序，逐帧验证接触与跨视角语义。</li><li>风格迁移、3D生成仅作为独立实验：先证明足尖/尾巴/背腹锚点与多视角一致性，再决定能否保留原几何标签。</li></ol></section>
<small>本页面及参考图片仅用于本项目内部评审。原始PNG未调色，网页预览为缩小JPEG；原图链接保留16位PNG输出。</small></main></html>'''
(DELIVERY / 'index.html').write_text(page, encoding='utf-8')

class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.links = []
    def handle_starttag(self, tag, attrs):
        self.links.extend(v for k, v in attrs if k in ('src', 'href'))

def check_links(base):
    parser = Links(); parser.feed((base / 'index.html').read_text())
    missing = [p for p in parser.links if not urlparse(p).scheme and not (base / unquote(p.split('#')[0])).is_file()]
    assert not missing, missing
    return len(parser.links)

manifest = {'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'source_original_unchanged': True, 'private_internal_review': True, 'files': []}
for p in sorted(DELIVERY.rglob('*')):
    if p.is_file() and p.name != 'package_manifest.json':
        manifest['files'].append({'path': p.relative_to(DELIVERY).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p)})
manifest['payload_bytes'] = sum(x['bytes'] for x in manifest['files'])
(DELIVERY / 'package_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
local_links = check_links(DELIVERY)
assert not WINDOWS.exists(), 'Do not overwrite an existing Windows delivery'
shutil.copytree(DELIVERY, WINDOWS)
for entry in manifest['files']:
    p = WINDOWS / entry['path']
    assert p.stat().st_size == entry['bytes'] and sha(p) == entry['sha256'], str(p)
assert sha(DELIVERY / 'package_manifest.json') == sha(WINDOWS / 'package_manifest.json')
report = dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(), local=str(DELIVERY), windows=str(WINDOWS), payload_files=len(manifest['files']), payload_bytes=manifest['payload_bytes'], all_payload_checksums_match=True, local_links_verified=local_links, windows_links_verified=check_links(WINDOWS))
(OUT / 'v066_package_verification.json').write_text(json.dumps(report, indent=2))
state = json.loads((OUT / 'state.json').read_text(encoding='utf-8-sig'))
state.update(stage='v066_windows_delivery_copied_and_hashed', updated_utc=report['utc'], delivery_directory=str(DELIVERY), windows_delivery=str(WINDOWS), package_verification=str(OUT / 'v066_package_verification.json'), next_action='Appearance frozen. Final self-contained DELIVERY assembled and copied to Windows Downloads; all payload SHA match and HTML links pass. Next visually inspect comparison boards and final HTML, smoke-test portable render_views.py on final Windows copy; do not modify appearance. Finish remaining artifact QA, then pause pymousestudio automation by 06:59:16 UTC (last15min wrap). No new experiments, original/dataset untouched.')
(OUT / 'state.json').write_text(json.dumps(state, ensure_ascii=False, indent=2))
print(json.dumps(report, indent=2))
