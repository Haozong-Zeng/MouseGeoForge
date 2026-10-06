import json,html,shutil,hashlib,re,importlib.util
from pathlib import Path
from urllib.parse import urlparse,unquote
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/pymousestudio_repair_audit_20261003';DL=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-Repair-Audit-20261003')
assert not DL.exists()
r=json.loads((OUT/'summary.json').read_text());m=json.loads((OUT/'mask_changes.json').read_text())
rows=[]
for name,key in [('全部关注点','focus'),('背部06/07','back'),('足部15–18','paws')]:
    x=r['metrics'][key];rows.append(f"<tr><td>{name}</td><td>{x['epipolar']['before']['median']:.2f} → {x['epipolar']['after']['median']:.2f}</td><td>{x['loo']['before']['median']:.2f} → {x['loo']['after']['median']:.2f}</td></tr>")
section='''<section id="conclusions"><h2>结论与优先复核</h2><p>有局部改善，不能整体直接验收。91张中69张改点；原PNG未变，未发现重复点和越界。31张mask多边形变化。仅比较前后共同相机和编号，避免删点造成虚假改善。单位均为原分辨率像素。</p><table border="1" cellpadding="8"><tr><th>部位</th><th>极线误差中位数 前→后</th><th>留一视角重投影中位数 前→后</th></tr>'''+''.join(rows)+'''</table><p>背部剩余误差明显高于足部。总体留一视角P90为50.83→52.25 px，尾部风险没有全面下降。R03与R10等组存在退步。反馈表仍为空；腹部23有7张变动，尾巴等额外修改未自动接受。</p><p><a href="back_curve_research.html">背侧曲线监督调研与具体实验路线</a> · <a href="pymousestudio_back_curve_and_review_20261003.md">项目工作记录</a> · <a href="curve_evidence.json">稀疏点覆盖与改点方向统计</a></p><h3>R12 camB分割需要回问</h3><p>紫色为新增区域，青色为删除区域。新增区域疑似包含玻璃底板或倒影，不应直接作为改善接受。</p><img src="mask_change_detail_R12_camB.png"><h3>R07 camB补充尾段</h3><p>新增细尾段在视觉上有合理性，仍应按统一可见分割定义复核。</p><img src="mask_change_detail_R07_camB.png"><h3>修改过的分割轮廓对照</h3>'''+''.join(f'<p><a href="mask_{x["group"]}_cam{x["camera"]}.jpg">{x["group"]} cam{x["camera"]}</a> 前后IoU {x["mask_iou_before_after"]:.3f}，仅表示改动幅度</p>' for x in m['records'])+'</section>'
page=(OUT/'index.html').read_text();assert 'id="conclusions"' not in page
page=page.replace('<h1>2026-10-03 第二轮返修核验</h1>','<h1>2026-10-03 第二轮返修核验</h1>'+section)
(OUT/'index.html').write_text(page)
doc=ROOT/'docs/pymousestudio_back_curve_and_review_20261003.md';shutil.copy2(doc,OUT/doc.name);text=doc.read_text()
if importlib.util.find_spec('markdown'):
    import markdown
    body=markdown.markdown(text,extensions=['tables','fenced_code'])
elif importlib.util.find_spec('markdown2'):
    import markdown2
    body=markdown2.markdown(text,extras=['tables','fenced-code-blocks'])
else:body='<pre>'+html.escape(text)+'</pre>'
(OUT/'back_curve_research.html').write_text('<meta charset="utf-8"><title>背侧曲线监督研究</title><style>body{font:18px/1.8 system-ui;max-width:1050px;margin:32px auto;padding:20px}pre{white-space:pre-wrap}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:10px}a{color:#1767ad}</style><a href="index.html">返回返修报告</a>'+body)
visual=dict(reviewed_keypoint_examples=['R07 camF','R03 camF','R10 camC','R05 camA','R07 camA'],reviewed_mask_examples=['R12 camB','R07 camB'],observations=['R07 camF lumbar07 moved substantially; single image does not establish unique anatomical location.','R12 camB newly added mask area extends over glass/reflection near floor; needs manual review.','R07 camB adds plausible thin tail segment.'],scope='Selected visual examples plus full 91-image structural/coordinate and matched multiview QC; not complete anatomical certification.',labels_promoted=False,training_modified=False)
(OUT/'visual_review.json').write_text(json.dumps(visual,indent=2))
DL.mkdir()
for p in OUT.iterdir():
    if p.is_file():shutil.copy2(p,DL/p.name)
checked=0
for p in DL.glob('*.html'):
    for link in re.findall(r'(?:href|src)=["\']([^"\']+)',p.read_text()):
        if not urlparse(link).scheme:
            path=unquote(urlparse(link).path)
            if path:assert (p.parent/path).is_file(),(p.name,link);checked+=1
for p in OUT.iterdir():
    if p.is_file():assert hashlib.sha256(p.read_bytes()).hexdigest()==hashlib.sha256((DL/p.name).read_bytes()).hexdigest()
verification=dict(export=str(DL),files=len(list(DL.iterdir())),links_checked=checked,all_exported_hashes_verified=True,raw_package_not_duplicated_in_delivery=True)
(OUT/'delivery_verification.json').write_text(json.dumps(verification,indent=2));print(json.dumps(verification,indent=2))
