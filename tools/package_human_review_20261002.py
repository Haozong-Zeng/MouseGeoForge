"""Portable reviewer package; never changes source labels or returned archives."""
import hashlib, html, json, re, shutil, zipfile
from pathlib import Path
from urllib.parse import unquote

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'output/pymousestudio_repair_audit_20261002'
DOWNLOAD=Path('/mnt/c/Users/55395/Downloads')
NAME='PyMouseStudio-Multiview-HumanReview-20261002'
DEST=DOWNLOAD/NAME
ARCHIVE=DOWNLOAD/(NAME+'.zip')

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    assert not DEST.exists() and not ARCHIVE.exists(), 'Refuse to overwrite an existing delivery'
    shutil.copytree(SOURCE,DEST)
    (DEST/'index.html').rename(DEST/'repair_report.html')
    old=json.loads((ROOT/'output/pymousestudio_numbered_review_20261001/review_manifest.json').read_text())
    base=ROOT/'dataset/PyMouseStudio/annotation_returns/20261002_numbered_review_repaired'
    returned=next(p for p in base.iterdir() if p.is_dir())
    extra=[]
    for ident in ('C17','C18'):
        case=next(c for c in old['cases'] if c['id']==ident)
        src=returned/case['folder']/'originals'
        dst=DEST/'supplemental_return_cases'/Path(case['folder']).name
        dst.mkdir(parents=True)
        for p in sorted(src.rglob('*')):
            if p.suffix.lower() in ('.png','.json'):
                assert not (dst/p.name).exists()
                shutil.copy2(p,dst/p.name)
                assert digest(p)==digest(dst/p.name)
        extra.append(dict(case=ident,session=case['session'],frame=case['frame'],images=len(list(dst.glob('*.png'))),folder=dst.relative_to(DEST).as_posix()))
    shutil.copy2(returned/'复核反馈模板.txt',DEST/'原24案例反馈模板.txt')
    feedback=['案例\t序列\t源帧\t相机\t关键点编号\t结论\t状态\tx\ty\t修改或删点依据\t标注人\t复核人']
    for e in extra:
        for p in sorted((DEST/e['folder']).glob('*.png')):
            for k in [15,16]:feedback.append(f"{e['case']}\t{e['session']}\t{e['frame']}\t{p.stem}\t{k}\t\tpending\t\t\t\t\t")
    (DEST/'C17_C18补充反馈模板.txt').write_text('\n'.join(feedback)+'\n',encoding='utf-8-sig')
    intro='''本包用途：人工复核，不是新的训练数据发布。
解压后先打开 index.html。

1. human_review_pilot：12组同步多视角、78张中心帧，原PNG及当前人工JSON、拼图、逐点反馈模板。先做R01/R02/R03/R10四组，统一理解后再完成其余八组。
2. supplemental_return_cases：C17/C18的返修原PNG和JSON。C16已在R09，无需重复标注。仅补充返修依据，不能为迎合算法强行改点。
3. 原24案例反馈模板.txt：请按原编号补充返修结论，未改动也可以确认原标无误。
4. repair_report.html及 *_before_after.jpg：基于实际返修JSON重新绘制的前后对照，第二轮需要时再查看，避免先被算法或修改方向诱导。
5. human_review_pilot/approved_guide：已确认的合成定义参考。旧指南提及10个必需点是历史通用要求；本轮试标仅06/07/15/16/17/18，腹部23和尾巴保持不变，不因旧指南扩大本轮工作范围。

左右以小鼠自身为准。先独立看原图，再看人工点；底视仅为辅助证据。
不确定可以明确标为无法定位；未标点不自动等于不可见。
每点填写可见程度、定位质量和判断依据；不强制给遮挡或无清晰标志点精确坐标。
本包不含同步上下文视频，也没有新增模型预测图。需要短片辅助的案例请记录编号，后续再补。
不更改PNG、文件名、帧号、目录或相机参数。修改JSON时另存回交副本。
'''
    (DEST/'先读我.txt').write_text(intro,encoding='utf-8-sig')
    links=''.join(f'<li>{e["case"]} {e["session"]} 源帧{e["frame"]}：'+ ' · '.join(f'<a href="{e["folder"]}/{p.name}">{p.name}</a>' for p in sorted((DEST/e['folder']).glob('*.png')))+f'；同目录含返修JSON。<a href="{e["case"]}_before_after.jpg">前后对照（第二轮再看）</a></li>' for e in extra)
    page='<meta charset="utf-8"><title>多视角人工复核交付包</title><style>body{font:18px sans-serif;max-width:1100px;margin:32px auto;line-height:1.7}pre{white-space:pre-wrap}</style><h1>PyMouseStudio 多视角人工复核</h1><p><a href="human_review_pilot/index.html">开始：12组试标与说明</a> · <a href="human_review_pilot/逐点复核反馈模板.txt">试标逐点反馈模板</a></p><pre>'+html.escape(intro)+'</pre><h2>已有返修的补充核实</h2><p><a href="human_review_pilot/本轮返修还需回问.txt">C16/C17/C18回问说明</a> · <a href="C17_C18补充反馈模板.txt">C17/C18反馈模板</a> · <a href="原24案例反馈模板.txt">原24案例反馈模板</a></p><ul>'+links+'</ul><h2>第二轮参考</h2><p><a href="repair_report.html">返修核验及所有前后对照</a> · <a href="human_review_pilot/approved_guide/index.html">已确认的合成示意</a></p>'
    (DEST/'index.html').write_text(page,encoding='utf-8')
    broken=[]
    for p in DEST.rglob('*.html'):
        for ref in re.findall(r'(?:href|src)=["\']([^"\']+)["\']',p.read_text(encoding='utf-8')):
            ref=unquote(html.unescape(ref)).split('#')[0]
            if not ref or ref.startswith(('https:','http:','data:','mailto:')):continue
            q=(p.parent/ref).resolve()
            if not q.is_relative_to(DEST.resolve()) or not q.is_file():broken.append((str(p.relative_to(DEST)),ref))
    assert not broken, broken
    inventory=[dict(path=p.relative_to(DEST).as_posix(),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(DEST.rglob('*')) if p.is_file()]
    manifest=dict(package=NAME,pilot_groups=12,pilot_images=78,supplemental_cases=extra,unique_real_center_images=78+sum(e['images'] for e in extra),context_video_included=False,source_labels_modified=False,html_links_verified=True,files=inventory)
    (DEST/'package_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    with zipfile.ZipFile(ARCHIVE,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for p in sorted(DEST.rglob('*')):
            if p.is_file():z.write(p,Path(NAME)/p.relative_to(DEST))
    with zipfile.ZipFile(ARCHIVE) as z:
        assert z.testzip() is None
        for row in inventory:assert hashlib.sha256(z.read(NAME+'/'+row['path'])).hexdigest()==row['sha256']
        files=len(z.infolist())
    result=dict(archive=str(ARCHIVE),bytes=ARCHIVE.stat().st_size,sha256=digest(ARCHIVE),files=files,crc_and_contents_verified=True,real_images=manifest['unique_real_center_images'])
    (DEST/'archive_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
