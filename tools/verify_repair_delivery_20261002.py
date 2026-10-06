"""Verify immutable sources, generated artifacts and pilot manifest."""
import html,json,re,shutil,py_compile
from pathlib import Path
from PIL import Image
from audit_repair_return_20261002 import BASE,OUT,DOWNLOAD,ROOT,sha,write
def main():
    summary=json.loads((OUT/'summary.json').read_text()); matched=json.loads((OUT/'matched_followup.json').read_text())
    for r in summary['files']:
        assert sha(BASE/r['source'])==r['old_json_sha256']
        assert sha((BASE/r['source']).with_suffix('.png'))==r['image_sha256']
    m=json.loads((OUT/'human_review_pilot/selection_manifest.json').read_text())
    assert len(m['groups'])==12 and m['center_images']==78
    assert not any(g['session'] in ('GR2','UR2','UR3') for g in m['groups'])
    checked=0
    for g in m['groups']:
        folder=OUT/'human_review_pilot'/f"{g['id']}_{g['session']}_f{g['frame']:06d}"
        for r in g['sources']:
            assert sha(folder/f"cam{r['camera']}.png")==r['image_sha256']
            assert sha(folder/f"cam{r['camera']}.json")==sha(Path(r['annotation_source']))
        checked+=len(g['sources'])
    visual=dict(inspected=['D02_before_after.jpg','D03_before_after.jpg','D05_before_after.jpg','C17_before_after.jpg','human_review_pilot/R01_AZR2L2_f000129/manual.jpg','human_review_pilot/R03_FR2_f001104/manual.jpg','human_review_pilot/R08_CR2_f001559/manual.jpg','human_review_pilot/R09_EL2_f000779/manual.jpg'],scope='Targeted visual inspection, not exhaustive review of all images. C17 full sheet was downscaled; do not claim fine-grained coordinate acceptance.',observations=['D02 duplicate limb labels have been differentiated; anatomical identity still needs multiview confirmation.','D03 duplicate07 was separated from08 tail-base with subsequent tail IDs shifted; do not treat this as proof of tail arc-length semantics.','D05 duplicate06 has become06/07, resolving structure but not proving precise real-surface anchor equivalence.','R01 side views and bottom view require tracing limb connections; no automatic identity swap warranted.','R03/R08/R09 dorsal surface lies on largely featureless dark fur. Exact point position cannot be certified from one silhouette; require anatomical consistency and explicit uncertainty.'],labels_modified=False)
    write(OUT/'visual_review.json',visual)
    lines=['<h2>验收结论</h2><p>结构检查通过，不代表所有语义和多视角对应已通过。未合并到训练标签，也未解除待复核排除项。</p><p>共同观测比较：15组下降、1组不变、2组上升。C16下降主要由删点造成；C17优先回问；C18小幅上升不是错标定论。</p><p>所有返修点的flags均为空，尚无可见/遮挡/无法定位的显式区分。24项原反馈表未填写。</p><table border="1"><tr><th>原编号</th><th>共同观测数</th><th>返修前极线均值px</th><th>返修后px</th></tr>']
    for r in matched['matched_pair_epipolar_clipped100']:lines.append(f'<tr><td>{r["id"]}</td><td>{r["n"]}</td><td>{r["before"]:.2f}</td><td>{r["after"]:.2f}</td></tr>')
    lines.append('</table><p>每项距离截断100px；固定相机，仅诊断，不作为自动接受门槛。</p>')
    index=OUT/'index.html';s=index.read_text();s=s.replace('<h1>2026-10-02 返修包核验</h1>','<h1>2026-10-02 返修包核验</h1>'+''.join(lines));index.write_text(s,encoding='utf-8')
    index=OUT/'human_review_pilot/index.html';s=index.read_text();s=s.replace('<h1>12组小规模多视角复核试标</h1>','<h1>12组小规模多视角复核试标</h1><p><a href="本轮返修还需回问.txt">现有返修C16/C17/C18回问说明</a>（C17/C18另行补充，不计入12组试标）</p>');index.write_text(s,encoding='utf-8')
    images=list(OUT.rglob('*.jpg'))+list(OUT.rglob('*.png'))
    for p in images:
        with Image.open(p) as im:im.verify()
    for p in (OUT/'index.html',OUT/'human_review_pilot/index.html'):
        for link in re.findall(r'(?:href|src)="([^"]+)"',p.read_text()):
            assert (p.parent/html.unescape(link)).is_file(),link
    for f in ('audit_repair_return_20261002.py','prepare_repair_followup_20261002.py','verify_repair_delivery_20261002.py'):py_compile.compile(str(ROOT/'tools'/f),doraise=True)
    result=dict(status='passed',original_label_image_pairs_unchanged=len(summary['files']),pilot_source_pairs_verified=checked,images_decoded=len(images),entry_page_links_valid=True,training_modified=False,context_video_exported=False)
    write(OUT/'verification.json',result)
    export=DOWNLOAD/'PyMouseStudio-Repair-Audit-20261002';shutil.copytree(OUT,export,dirs_exist_ok=True)
    for p in OUT.rglob('*'):
        if p.is_file():assert sha(p)==sha(export/p.relative_to(OUT))
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
