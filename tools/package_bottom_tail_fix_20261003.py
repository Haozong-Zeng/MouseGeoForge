"""Build native-render comparison delivery and verify the Windows copy."""
import json,hashlib,shutil,datetime as dt
from pathlib import Path
from PIL import Image,ImageDraw
root=Path(__file__).resolve().parents[1];base=root/'output/bottom_tail_fix_20261003'
manifest=json.loads((base/'final_balanced/manifest.json').read_text());assert manifest['complete']
source=root/'BlenderPyMouseProjects/Cylinder/PyMouseStudio-RealCapture-EarStudy-20261003/DELIVERY'
delivery=root/'BlenderPyMouseProjects/Cylinder/PyMouseStudio-RealCapture-BottomTailFix-20261003/DELIVERY-Balanced'
windows=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-BottomTailFix-20261003/DELIVERY');assert not windows.exists()
comparisons=delivery/'comparisons';comparisons.mkdir(exist_ok=True)
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def host(p):return root/Path(p.replace('\\','/').split('/MouseGeoForge/',1)[1])
for pose,r in manifest['poses'].items():
    assert sha(host(r['source']))==r['source_sha256']
    assert sha(host(r['blend']))==r['blend_sha256']
    for c in 'AF':
        previous=source/'renders'/pose/f'cam{c}.png';current=delivery/'renders'/pose/f'cam{c}.png'
        assert Image.open(current).size==(1080,1200)
        old=delivery/'before'/pose/f'cam{c}.png';old.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(previous,old)
        board=Image.new('RGB',(1080,636),'#17202d');draw=ImageDraw.Draw(board)
        for col,(p,label) in enumerate([(previous,'BEFORE'),(current,'AFTER')]):
            board.paste(Image.open(p).convert('RGB').resize((540,600)),(col*540,36));draw.text((col*540+12,10),f'{pose} cam{c} - {label}',fill='white')
        board.save(comparisons/f'{pose}_cam{c}.jpg',quality=95)
board=Image.new('RGB',(720,740),'#17202d');draw=ImageDraw.Draw(board)
for col,(p,label) in enumerate([(source/'renders/headraise/camA.png','BEFORE'),(delivery/'renders/headraise/camA.png','AFTER')]):
    im=Image.open(p).convert('RGB').crop((0,400,180,750)).resize((360,700));board.paste(im,(col*360,40));draw.text((col*360+12,12),label,fill='white')
board.save(comparisons/'tail_closeup.jpg',quality=98)
evidence=delivery/'evidence';evidence.mkdir(exist_ok=True)
for f in ['probes.json','glass_probes.json','transport_probes.json','fix_probes.json','floor_shader_audit.json']:
    shutil.copy2(base/f,evidence/f)
shutil.copy2(base/'final_balanced/manifest.json',evidence/'validation.json')
shutil.copy2(base/'dust_balance.json',evidence/'dust_balance.json')
shutil.copy2(root/'BlenderScripts/fix_bottom_tail_balanced_20261003.py',evidence/'fix_bottom_tail_balanced_20261003.py')
for f in ['LICENSE_NOTES.txt','render_views.py']:shutil.copy2(source/f,delivery/f)
shutil.copy2(root/'BlenderScripts/fix_bottom_tail_shadow_20261003.py',evidence/'fix_bottom_tail_shadow_20261003.py')
sections=[]
for pose,label in [('headraise','HeadRaise'),('climb','Climb'),('quadruped','Quadruped')]:
    sections.append(f'<h2>{label}</h2><a href="PyMouseStudio-BottomTailFix-{label}-Static.blend">下载/打开独立 Blender 工程</a>')
    for c in 'AF':sections.append(f'<h3>cam{c}：左为修复前，右为修复后</h3><img src="comparisons/{pose}_cam{c}.jpg"><p><a href="before/{pose}/cam{c}.png">原图</a> · <a href="renders/{pose}/cam{c}.png">修复后 1080×1200 原图</a></p>')
html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Bottom view 尾巴黑块修复</title><style>body{background:#17202d;color:#e9edf3;font:17px/1.7 system-ui;margin:32px auto;max-width:1120px;padding:0 20px}a{color:#8ed9ff}img{max-width:100%}.note{background:#273647;padding:20px;border-radius:12px}</style><main><h1>Bottom view 尾巴黑块修复</h1><div class="note">定位到玻璃底板透射阴影问题：尾巴未穿过地板，调整尾巴/底板法线、关闭尾巴 SSS、清除灰尘材质、细分身体和消除玻璃交叠均未消除黑块。隐藏底板或关闭底板阴影可消除，后者会改变整体照明，因此只对底板清玻璃分支加入阴影透光近似。</div><h2>最终改动与限制</h2><p>阴影射线使用每界面 0.945 直线透光；相机、反射和折射射线保留原玻璃，灰尘覆盖分支不变。关闭底板 MNEE caster，避免与近似阴影通路混用。没有修改尾巴/耳朵材质、任何皮肤顶点、姿态或相机。部分灰尘和腹面会变亮；这不是完整物理焦散解，也不是恢复了真实玻璃参数。未重导出数据集。</p><p>实现参考：<a href="https://docs.blender.org/manual/en/2.80/render/cycles/optimizations/reducing_noise.html">Blender 官方：Glass and Transparent Shadows</a>。三个独立静态工程保存后重开，几何误差为 0，源文件哈希未变。</p><h2>问题区域近景</h2><img src="comparisons/tail_closeup.jpg">'''+''.join(sections)+'''<h2>可复现记录</h2><a href="evidence/validation.json">保存重开与渲染验证</a> · <a href="evidence/fix_bottom_tail_shadow_20261003.py">修复脚本</a><p>Cycles / OptiX / 128 samples / 1080×1200，3 姿态 × A/F 两视角。旧交付未覆盖。</p></main></html>'''
html=html.replace('灰尘覆盖分支不变。','灰尘覆盖率不变，灰尘漫反射颜色乘 0.02，抑制新透光条件下过亮的白色颗粒。').replace('部分灰尘和腹面会变亮；','灰尘反射经过补偿，腹面照明仍会有所改变；')
(delivery/'index.html').write_text(html,encoding='utf-8')
(delivery/'README.txt').write_text('Bottom-view tail shadow fix. Open index.html for before/after comparisons. Three static Blender projects; no source overwrite or dataset regeneration. Optical glass and dust remain; clear-floor direct shadow transmission is approximated, so dust/underside lighting changes. Geometry and mouse materials unchanged. Use render_views.py with --views AF. See evidence/validation.json and index.html for limitations.',encoding='utf-8')
package=dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),files=[dict(path=p.relative_to(delivery).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(delivery.rglob('*')) if p.is_file() and p.name!='package_manifest.json'])
(delivery/'package_manifest.json').write_text(json.dumps(package,indent=2));shutil.copytree(delivery,windows)
for entry in package['files']:assert sha(windows/entry['path'])==entry['sha256'],entry['path']
verification=dict(windows_delivery=str(windows),files=len(package['files']),bytes=sum(x['bytes'] for x in package['files']),all_hashes_verified=True,original_sources_unchanged=True)
(base/'delivery_verification.json').write_text(json.dumps(verification,indent=2));print(json.dumps(verification,indent=2))
