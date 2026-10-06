import datetime as dt,hashlib,json,shutil
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import unquote,urlparse
import cv2,numpy as np
root=Path(__file__).resolve().parents[1];base=root/'output/pymousestudio_ear_cycle_20261003';plan=json.loads((base/'plan.json').read_text());out=base/'validate_v003';m=json.loads((out/'manifest.json').read_text());assert m['complete']
delivery=Path(plan['project_directory'])/'DELIVERY';winroot=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-EarStudy-20261003');windows=winroot/'DELIVERY';assert not windows.exists()
old=root/'BlenderPyMouseProjects/Cylinder/PyMouseStudio-RealCapture-Replica-20261002/DELIVERY'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def copy(src,relative):
    dst=delivery/relative;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
for pose in m['poses']:
    assert sha(Path(plan['sources'][pose]))==plan['source_sha256'][pose]
    for c in 'ACEF':
        src=out/pose/f'cam{c}.png';im=cv2.imread(str(src),cv2.IMREAD_UNCHANGED);assert im.shape==(1200,1080,4) and im.dtype==np.uint16
        copy(src,f'renders/{pose}/cam{c}.png')
    for suffix in ['ear_before_after','real_comparison']:copy(out/f'{pose}_{suffix}.jpg',f'comparisons/{pose}_{suffix}.jpg')
copy(out/'all_pose_views.jpg','comparisons/all_pose_views.jpg')
for c in 'CF':
    copy(base/'probe_v001'/f'ear_compare_cam{c}.jpg',f'comparisons/round1_cam{c}.jpg')
    copy(base/'probe_v002'/f'ear_refine_cam{c}.jpg',f'comparisons/round2_cam{c}.jpg')
for src,dst in [('audit_baseline/audit.json','baseline_audit.json'),('probe_v001/manifest.json','round1_parameters.json'),('probe_v001/review.json','round1_review.json'),('probe_v002/manifest.json','round2_parameters.json'),('probe_v002/review.json','round2_review.json'),('validate_v003/manifest.json','saved_reopened_validation.json')]:copy(base/src,'evidence/'+dst)
copy(old/'render_views.py','render_views.py')
copy(Path(plan['reference_root'])/'FR2/frame_001429/camF.png','references/heldout_FR2_f1429_camF.png')
review=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),decision='Accept .60 ear Base Color as a local appearance improvement over .12, not photorealistic or anatomically exact reconstruction.',views_actually_reviewed=['all3pose ACEF ear before/after boards','all12fullview contact sheet','quadruped andheadraise realCF vsnewCF boards','priorclimbCF realref andnativebaselineF','heldoutFR2f1429nativeF afterselection'],observations=['Rim and ear bowl become visible instead of dark rubber-like baseline.','Cavity shadows and orientation-dependent shading remain; no observed whole-ear flat white collapse in checked views.','Back surfaces inE and undersides inA are much brighter; realreference ears are also relatively pale, but spectral/lighting mismatch remains.','Ear outline, folding, membrane thickness and root fur transition still differ from real; no geometry/groom correction in this scope.','Headraise small visible ear surfaces remain pose-dependent; material cannot fix orientation mismatch.'],geometry_material_check='All3clones saved/reopened: evaluated skin maxdifference0, identical object/bone/camera matrices and light values, non-ear material graphs/inputs identical; ear graph/inputs identical except Base Color.',heldout='FR2f1429F viewed afterchoice,notuned;ear is pale inreference,consistent qualitative direction,not generalization metric.',scope='No emission, global exposure, other material orpose changes. All3sourceSHAunchanged. No downloaded assets or external uploads.',caveat='Previous work identified GPU render small non-bitwise repeatability; unchanged appearance outside ears is established by scene snapshots, not pixel-zero claims.')
(delivery/'evidence/final_visual_review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2))
readme='''PyMouseStudio 耳部材质专项 — 2026-10-03

先打开 index.html 查看三姿态新旧耳部对照、真实参考和完整试验记录。
最终唯一外观参数变化：MS_Mouse材质槽4，Principled Base Color线性灰度0.12→0.60。
最终材质名称：EarStudy_Final_Gray060。其余粗糙度0.80、Specular IOR Level0.08、
Subsurface Weight0.025及Scale0.15等维持原值。没有新增透光分支或发光。
数字是此灰度采集场景的外观拟合，不是实测小鼠光谱反射率。

三个.blend是独立静态姿态，不是连续动画：Climb第31帧、Quadruped第111帧、HeadRaise第216帧。
Blender5.1.2；默认F透视相机；Cycles OptiX GPU，128采样，1080×1200。
毛发GUI默认隐藏、最终渲染启用。继承源文件内嵌纹理，未新增外部依赖。
本包12张16位RGBA PNG为每姿态A/C/E/F四视角。A底部，其余侧面。
相机仍保留A–G，切换相机需同步其replica_pixel_aspect_y_over_x像素宽高比。

同目录PowerShell重渲染示例（不会保存.blend，输出目录必须尚不存在）：
  & "C:\\Program Files\\Blender Foundation\\Blender 5.1\\blender.exe" -b ".\\PyMouseStudio-EarStudy-Climb-Static.blend" --python ".\\render_views.py" -- --views ACEF --output-dir ".\\rerender_climb"
可加 --dry-run 检查设置，必须有OptiX设备，不静默退回CPU。

几何数值使用毫米，但scene unit_scale仍为1；灯光/景深参数有数值补偿，非实测瓦数/光圈。
耳部外观有所改善，但耳形/折叠/根部毛发、口鼻、足底接触、玻璃和器材等原有局限仍在。
不是精确真实视频重建、不是3D真值，也没有扩展训练数据。
上一轮DELIVERY及其三个工程完全未改动；本包是独立分支。
原始资产及真实参考的许可限制见LICENSE_NOTES.txt，不要擅自公开发布。
'''
(delivery/'README.txt').write_text(readme,encoding='utf-8')
(delivery/'LICENSE_NOTES.txt').write_text('本轮仅修改用户现有工程的耳部材质，未下载或引入外部美术素材，未上传私有数据。网格、毛发、纹理、头部装置继承原有资产许可；未进行全面公开再分发许可审查。真实参考属于项目内部复核数据，本交付不得视为获得公开传播许可。所有试验参数为外观假设，并非动物组织物理测量。',encoding='utf-8')
sections=[]
for pose,label in [('climb','攀爬'),('quadruped','四足'),('headraise','抬头')]:
    title={'climb':'Climb','quadruped':'Quadruped','headraise':'HeadRaise'}[pose]
    sections.append(f'<section><h2>{label}：上排旧版，下排耳材质更新</h2><a href="comparisons/{pose}_ear_before_after.jpg"><img src="comparisons/{pose}_ear_before_after.jpg" alt="{label}新旧耳部对照"></a><p><a href="PyMouseStudio-EarStudy-{title}-Static.blend">打开{label}独立工程</a> · 原分辨率：'+ ' · '.join(f'<a href="renders/{pose}/cam{c}.png">cam{c}</a>' for c in 'ACEF')+f'</p><details><summary>查看真实参考与更新后的全图（姿态不完全对应）</summary><img src="comparisons/{pose}_real_comparison.jpg"></details></section>')
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PyMouseStudio 耳部材质专项</title><style>body{font:17px/1.8 system-ui,"Microsoft YaHei",sans-serif;background:#15202d;color:#eef3fa;margin:0}main{max-width:1440px;margin:auto;padding:28px}img{max-width:100%;height:auto}a{color:#8fcaff}section{border-top:1px solid #435367;margin-top:28px}h1{font-size:32px}h2{font-size:23px}.note{background:#26364b;padding:16px}td,th{padding:10px;border-bottom:1px solid #435367;text-align:left}</style><main><h1>耳部材质专项：保留简单提亮方案</h1>
<p class="note">最终仅将耳部基础反射线性灰度从0.12提高到0.60。九组候选比较后，没有保留额外透光、较强高光或新增散射。耳廓与耳内层次更接近参考视频，但尚非照片级真实。</p>
<p>三个独立静态工程；每个工程A/C/E/F四视角，共12张1080×1200、16位PNG；Cycles OptiX、128采样。旧交付未覆盖。</p><p><a href="README.txt">使用说明</a> · <a href="render_views.py">重渲染脚本</a> · <a href="LICENSE_NOTES.txt">许可与范围</a> · <a href="package_manifest.json">SHA256文件清单</a></p>'''+''.join(sections)+'''
<section><h2>试验取舍</h2><table><tr><th>方案</th><th>观察与决策</th></tr><tr><td>灰度0.30 / 0.45</td><td>改善偏暗问题，但整体仍比参考耳膜偏暗。</td></tr><tr><td>灰度0.60</td><td>耳廓和内侧变亮，腔体阴影仍可辨认；三个姿态方向一致，作为最终选择。</td></tr><tr><td>灰度0.75</td><td>F视角更容易出现较宽的亮白区域，选择保留0.60的明暗余量。</td></tr><tr><td>增加高光/次表面散射</td><td>在本轮参数和照明下未显示足够清晰的额外收益，不增加复杂度。</td></tr><tr><td>15%或30%薄组织透光混合</td><td>当前照明下偏灰暗，没有比简单反射提亮更接近参考；不代表真实耳朵不透光。</td></tr></table>
<details><summary>查看首轮五组 + 第二轮四组候选局部图</summary><h3>首轮C/F</h3><img src="comparisons/round1_camC.jpg"><img src="comparisons/round1_camF.jpg"><h3>第二轮C/F</h3><img src="comparisons/round2_camC.jpg"><img src="comparisons/round2_camF.jpg"></details></section>
<section><h2>验证范围</h2><p>三个工程已保存、重新打开并渲染。变形后的皮肤顶点最大变化为0；物体/骨骼/相机矩阵、灯光和非耳材质一致；耳材质除基础颜色外的节点和输入保持原值。此结论来自场景检查，不是宣称GPU渲染逐像素确定性。</p><p><a href="evidence/saved_reopened_validation.json">保存/重开验证</a> · <a href="evidence/final_visual_review.json">视觉检查结论</a> · <a href="evidence/round1_parameters.json">首轮参数</a> · <a href="evidence/round2_parameters.json">第二轮参数</a></p><img src="comparisons/all_pose_views.jpg"></section>
<section><h2>预留帧与残余问题</h2><p>选择冻结后检查FR2/1429的F视角，真实耳朵也是较浅的灰度；仅支持改进方向，不是定量泛化证明，未继续据此调参。</p><p><a href="references/heldout_FR2_f1429_camF.png">查看预留真实帧</a></p><p>耳形、折叠方向、厚度和根部毛发过渡仍不一致；全身毛发、口鼻、脚底间隙及玻璃/器材局限没有在本轮修正。真实和合成姿态不完全对应，局部图是视觉对照而非配准误差评估。真实参考仅限内部检查。网页图片已检查，未绕过此前浏览器对本地file URL的工具限制。</p></section></main></html>'''
(delivery/'index.html').write_text(page,encoding='utf-8')
manifest=dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),private_internal_review=True,files=[dict(path=p.relative_to(delivery).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(delivery.rglob('*')) if p.is_file() and p.name!='package_manifest.json'])
manifest['payload_bytes']=sum(x['bytes'] for x in manifest['files']);(delivery/'package_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
class Links(HTMLParser):
    def __init__(self):super().__init__();self.items=[]
    def handle_starttag(self,tag,attrs):self.items.extend(v for k,v in attrs if k in ('href','src'))
links=Links();links.feed(page)
for link in links.items:
    if not urlparse(link).scheme:assert (delivery/unquote(link)).is_file(),link
shutil.copytree(delivery,windows)
for entry in manifest['files']:assert sha(windows/entry['path'])==entry['sha256'],entry['path']
assert sha(delivery/'package_manifest.json')==sha(windows/'package_manifest.json')
(winroot/'index.html').write_text('<meta charset="utf-8"><title>耳部材质专项</title><h1>耳部材质专项结果</h1><p><a href="DELIVERY/index.html">打开最终报告、三姿态工程与新旧对照</a></p><p>九组对照后选择只将耳部基础反射灰度0.12改为0.60，旧交付未修改。</p>',encoding='utf-8')
verification=dict(utc=manifest['created_utc'],payload_files=len(manifest['files']),payload_bytes=manifest['payload_bytes'],all_windows_sha_match=True,local_links_checked=len(links.items),original_three_blends_unchanged=True,render_count=12)
(base/'delivery_verification.json').write_text(json.dumps(verification,indent=2))
state=json.loads((base/'state.json').read_text(encoding='utf-8-sig'));state.update(stage='v003_accepted_packaged_windows_verified',active_pid=None,active_job=None,updated_utc=manifest['created_utc'],delivery=str(delivery),windows_delivery=str(windows),latest_review=str(delivery/'evidence/final_visual_review.json'),next_action='v003 actual3pose ACEF ear old/newboards+all12fullsheet+quad/headrealCFboards reviewed; improvement accepted onlyearmaterial, shape/pose mismatch still. HeldoutFR2f1429nativeFcheckedafterfreeze,no tuning. ProjectEarStudy/DELIVERY andWindowsEarStudy/DELIVERY packaged3blends12PNG+boards+READMErenderhelper+evidence,allSHAandlinks pass,source3SHAunchanged. No furtherappearanceexperiments. NEXT ~08:40 heartbeat inspect actualWindowscopiedblends withdryrun helper,checkpackeddependenciesandGPUinactive,finalintegrity; do NOTeditoldDELIVERY. Last10min~08:50pausepymousestudio viaapptool andrecordclosure before08:58UTC. BrowserfileURLpolicyblockedno workaround. Thisstudyonly.60earbasecolor; no newSSS/trans/spec used.')
(base/'state.json').write_text(json.dumps(state,ensure_ascii=False,indent=2));print(json.dumps(verification,indent=2))
