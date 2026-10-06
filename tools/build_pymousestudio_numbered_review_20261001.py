"""Numbered human-review packet. Never changes source annotations or calibration."""
import hashlib
import html
import json
import math
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'dataset/PyMouseStudio/annotation_returns/20260930_full_v2'
AUDIT = ROOT / 'output/pymousestudio_manual_multiview_20260930'
OUT = ROOT / 'output/pymousestudio_numbered_review_20261001'
FONT = '/mnt/c/Windows/Fonts/msyh.ttc'


def font(size):
    return ImageFont.truetype(FONT, size)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def draw_view(source, selected, duplicate=False):
    raw = Image.open(source).convert('RGB')
    ann = json.loads(source.with_suffix('.json').read_text())
    marked = raw.copy()
    d = ImageDraw.Draw(marked)
    counts = defaultdict(int)
    xyall = [xy for s in ann['shapes'] for xy in s['points']]
    for s in ann['shapes']:
        if s['shape_type'] != 'point':
            continue
        k = int(s['label'][:2])
        counts[k] += 1
        x, y = s['points'][0]
        if k not in selected:
            d.ellipse((x-2,y-2,x+2,y+2), fill='#bbbbbb')
            continue
        suffix = chr(96+counts[k]) if duplicate else ''
        color = ('#37e6ff' if counts[k] == 1 else '#ffad47') if duplicate else ('#37e6ff' if k in (15,17) else '#ffad47')
        d.ellipse((x-7,y-7,x+7,y+7), outline='black', width=5)
        d.ellipse((x-6,y-6,x+6,y+6), outline=color, width=3)
        # Stable IDs identify the original annotation occurrences, not corrections.
        dx,dy = (14,-32) if counts[k] % 2 else (14,10)
        if not duplicate and k in (16,18):
            dx,dy = -100,12
        d.line((x,y,x+dx,y+dy+12), fill=color,width=2)
        d.text((x+dx,y+dy),f'{k:02d}{suffix}',font=font(26),fill=color,stroke_width=2,stroke_fill='black')
    xs,ys = zip(*xyall)
    box=(max(0,int(min(xs))-55),max(0,int(min(ys))-55),min(raw.width,int(max(xs))+56),min(raw.height,int(max(ys))+56))
    return raw.crop(box), marked.crop(box), marked


def main():
    if OUT.exists():
        assert not any(OUT.iterdir()), 'Never overwrite an existing report.'
    else:
        OUT.mkdir(parents=True)
    diagnostics=json.loads((AUDIT/'structural_and_bias_diagnostics.json').read_text())
    manifest=json.loads((BASE/'ingestion_manifest.json').read_text())
    grouped=defaultdict(list)
    for row in diagnostics['duplicate_points']:
        grouped[row['image']].append(row)
    cases=[]
    for n,(image,rows) in enumerate(grouped.items(),1):
        p=Path(image)
        cases.append(dict(id=f'D{n:02d}',kind='duplicate',priority='确定的结构问题',split=p.parts[1],session=p.parts[2],frame=int(p.parts[3][6:]),
                          cameras=[p.stem[3:]],source_image=image,points=rows))
    for n,row in enumerate(diagnostics['paw_relative_parity_hypotheses'],1):
        gain=row['baseline_mean_epipolar_px']-row['hypothesis_mean_epipolar_px']
        priority='优先复核' if gain>=10 else '其次复核' if gain>=5 else '弱证据附录'
        g=next(g for g in manifest['frame_groups'] if g['session']==row['session'] and g['frame']==row['frame'])
        cases.append({**row,'id':f'C{n:02d}','kind':'correspondence','priority':priority,'score_gain_px':gain,
                      'paired_cameras':row['cameras'],'cameras':g['cameras']})
    intro=[
        'PyMouseStudio 人工标注编号复核报告 | 2026-10-01',
        '6张重复点图片（7个重复槽位）+18组左右爪对应候选。编号D按图片、C按多视角组。',
        'C01–C10为优先/其次复核，C11–C18仅弱证据附录；全部候选保留，避免隐去弱证据。',
        '重要：对应候选不是已证实错标。相机偏差、遮挡、近邻爪和同步均可影响评分；不能仅凭代价降低改标。',
        '左右以小鼠自身解剖学左右为准，不是图像左右。参考camA只是数学基准，不保证它正确。',
        '原始PNG/JSON均原样复制；未改名点、未交换左右、未修改相机、未改数据划分，留出集仅质量复核。',
        '帧号为原视频源帧索引（从0计），不是报告序号。坐标为原1080×1200图中的(x,y)，左上原点。',
        'D图的a/b只标识JSON中第一次/第二次出现，不能据此判断哪个保留。尾尖不要求贴边。',
        'C图：青色15/17=当前左前/左后爪；橙色16/18=当前右前/右后爪。展示的始终是原标注。',
        '几何代价为视角对称极线距离、单项截断100px后的平均，不是3D毫米误差，也不是重投影误差。',
        '优先级只为安排人工复核：平均代价下降>=10px为优先，5–10px为其次，<5px为弱证据；不是错标阈值。',
        '请按编号反馈：确认原标无误 / 应保留哪一个重复点 / 哪个相机哪个ID改为什么 / 无法判断。',
        '如修改JSON，请保留文件名和目录层级，另存修正版，不覆盖本次原始归档。',
    ]
    sections=[]
    report=intro.copy()
    response=['编号\t序列\t源帧\t复核结论\t相机及关键点ID\t具体修改或备注']
    inventory=[]
    for case in cases:
        ident=case['id']; stem=f"{ident}_{case['session']}_frame_{case['frame']:06d}"
        folder=OUT/'cases'/stem
        folder.mkdir(parents=True)
        header=f"{ident} | {case['priority']} | {case['session']} | 源帧{case['frame']:06d} | {case['split']}"
        details=[header]
        if case['kind']=='duplicate':
            selected={int(r['label'][:2]) for r in case['points']}
            details.append(f"图片：{case['source_image']}")
            for row in case['points']:
                coords='; '.join(f"{int(row['label'][:2]):02d}{chr(97+i)}=({xy[0][0]:.3f}, {xy[0][1]:.3f})" for i,xy in enumerate(row['coordinates']))
                details.append(f"重复点 {row['label']}：{coords}")
            details.append('复核动作：确认重复点应保留哪一个；另一点是误加，还是应改为其他点名？不要默认保留第一个。')
        else:
            selected={17,18} if case['pair']=='hindpaws' else {15,16}
            details.append('检查点：'+('17左后爪 / 18右后爪' if case['pair']=='hindpaws' else '15左前爪 / 16右前爪'))
            flip='/'.join('cam'+c for c in case['relative_flip_cameras'])
            same='/'.join('cam'+c for c in case['paired_cameras'] if c not in case['relative_flip_cameras'])
            details.append(f'相对对应候选：{same} 与 {flip} 两组的左右对应不一致；两组都要复核，不能认定后组必错。')
            details.append(f"只读交换假设的极线代价：{case['baseline_mean_epipolar_px']:.2f} → {case['hypothesis_mean_epipolar_px']:.2f}px，下降{100*case['relative_cost_reduction']:.1f}%。")
            details.append('全部实际视角均附上；缺一只爪的视角仅供辅助，不参与这个左右成对假设计算。')
            details.append('复核动作：结合所有视角确认解剖学左右；几何提示可被否决，绝不要求按提示交换。')
        panels=[];links=[]
        for cam in case['cameras']:
            src=BASE/'real_images'/case['split']/case['session']/f"frame_{case['frame']:06d}"/f'cam{cam}.png'
            dest=folder/'originals'/src.relative_to(BASE)
            dest.parent.mkdir(parents=True,exist_ok=True)
            for source in (src,src.with_suffix('.json')):
                target=dest.with_suffix(source.suffix)
                shutil.copy2(source,target)
                assert sha(source)==sha(target)
                inventory.append(dict(case=ident,source=source.relative_to(BASE).as_posix(),copy=target.relative_to(OUT).as_posix(),sha256=sha(source)))
            raw,marked,full=draw_view(src,selected,case['kind']=='duplicate')
            full.save(folder/f'cam{cam}_annotated.jpg',quality=95)
            comparison=Image.new('RGB',(raw.width*2,raw.height+55),'#17212c')
            comparison.paste(raw,(0,55));comparison.paste(marked,(raw.width,55))
            ImageDraw.Draw(comparison).text((12,12),f'{ident} cam{cam} | 左：原图   右：原标注（未修正）',font=font(23),fill='white')
            comparison.save(folder/f'cam{cam}_comparison.jpg',quality=95)
            marked.thumbnail((480,485))
            panel=Image.new('RGB',(500,540),'#17212c')
            tag='重复点' if case['kind']=='duplicate' else ('成对观测' if cam in case['paired_cameras'] else '辅助视角')
            ImageDraw.Draw(panel).text((10,7),f'{ident} cam{cam} | {tag}',font=font(23),fill='white')
            panel.paste(marked,((500-marked.width)//2,47))
            panels.append(panel)
            links.append(f'<a href="{dest.relative_to(OUT).as_posix()}">cam{cam}原PNG</a> · <a href="{dest.with_suffix(".json").relative_to(OUT).as_posix()}">JSON</a> · <a href="cases/{stem}/cam{cam}_comparison.jpg">放大对照</a> · <a href="cases/{stem}/cam{cam}_annotated.jpg">全幅标注</a>')
        columns=min(3,len(panels))
        sheet=Image.new('RGB',(columns*500,90+math.ceil(len(panels)/columns)*540),'#17212c')
        draw=ImageDraw.Draw(sheet)
        draw.text((12,8),f"{ident} {case['session']} 源帧{case['frame']:06d}",font=font(25),fill='white')
        draw.text((12,48),'原标注：未改动；请结合报告确认',font=font(21),fill='#bed0df')
        for i,panel in enumerate(panels):sheet.paste(panel,((i%columns)*500,90+(i//columns)*540))
        sheet.save(folder/'overview.jpg',quality=95)
        report.extend(['']+details+[f'图片目录：cases/{stem}/'])
        (folder/'case_details.txt').write_text('\n'.join(details),encoding='utf-8-sig')
        response.append(f"{ident}\t{case['session']}\t{case['frame']:06d}\t\t\t")
        case['folder']=f'cases/{stem}'
        sections.append(f'<section id="{ident}"><h2>{html.escape(header)}</h2>'+''.join(f'<p>{html.escape(t)}</p>' for t in details[1:])+f'<a href="cases/{stem}/overview.jpg"><img loading="lazy" src="cases/{stem}/overview.jpg"></a><p>'+'<br>'.join(links)+'</p><a href="#top">返回清单</a></section>')
        print(header,flush=True)
    reportfile=OUT/'复核编号报告.txt'
    reportfile.write_text('\n'.join(report)+'\n',encoding='utf-8-sig')
    (OUT/'复核反馈模板.txt').write_text('\n'.join(response)+'\n',encoding='utf-8-sig')
    (OUT/'review_manifest.json').write_text(json.dumps(dict(cases=cases,raw_copy_inventory=inventory,labels_unchanged=True,calibration_unchanged=True),ensure_ascii=False,indent=2))
    nav='<table><tr><th>编号</th><th>优先级</th><th>序列/源帧</th><th>类型</th></tr>'+''.join(f'<tr><td><a href="#{c["id"]}">{c["id"]}</a></td><td>{c["priority"]}</td><td>{c["session"]} / {c["frame"]:06d}</td><td>{"重复点" if c["kind"]=="duplicate" else "后爪" if c["pair"]=="hindpaws" else "前爪"}</td></tr>' for c in cases)+'</table>'
    (OUT/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>标注编号复核报告</title><style>body{font:17px sans-serif;background:#17212c;color:#eef3f8;max-width:1550px;margin:24px auto;padding:20px;line-height:1.65}a{color:#79d9ff}img{max-width:100%}td,th{border:1px solid #6c7d88;padding:7px}table{border-collapse:collapse}section{border-top:2px solid #73818f;margin-top:35px;padding-top:15px}</style><body id="top"><h1>PyMouseStudio 编号复核报告</h1>'+''.join(f'<p>{html.escape(t)}</p>' for t in intro[1:])+nav+'<p><a href="复核编号报告.txt">纯文字报告</a> · <a href="复核反馈模板.txt">反馈模板</a></p>'+''.join(sections)+'</body></html>',encoding='utf-8')
    for p in OUT.rglob('*'):
        if p.suffix.lower() in ('.jpg','.png'):
            with Image.open(p) as im:im.verify()
    destination=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-Numbered-Review-20261001')
    assert not destination.exists()
    shutil.copytree(OUT,destination)
    for p in OUT.rglob('*'):
        if p.is_file():assert sha(p)==sha(destination/p.relative_to(OUT))
    zip_path=destination.with_suffix('.zip')
    with zipfile.ZipFile(zip_path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in sorted(destination.rglob('*')):
            if p.is_file():z.write(p,Path(destination.name)/p.relative_to(destination))
    with zipfile.ZipFile(zip_path) as z:assert z.testzip() is None
    print(json.dumps(dict(cases=len(cases),raw_files_copied=len(inventory),zip_bytes=zip_path.stat().st_size,windows_folder=str(destination)),indent=2))


if __name__=='__main__':
    main()
