"""Immutable returned-package QC; no training label or calibration changes."""
import collections, hashlib, html, itertools, json, math, shutil, stat, zipfile
from pathlib import Path, PurePosixPath
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from audit_pymousestudio_manual_multiview import fundamental, epi

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'dataset/PyMouseStudio/annotation_returns/20260930_full_v2'
DEST=ROOT/'dataset/PyMouseStudio/annotation_returns/20261002_numbered_review_repaired'
OUT=ROOT/'output/pymousestudio_repair_audit_20261002'
DOWNLOAD=Path('/mnt/c/Users/55395/Downloads')
ZIP=DOWNLOAD/'PyMouseStudio-Numbered-Review-20261001---ok---fanxiu.zip'
OLD=ROOT/'output/pymousestudio_numbered_review_20261001'
FONT='/mnt/c/Windows/Fonts/msyh.ttc'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def points(a):
    d=collections.defaultdict(list)
    for s in a['shapes']:
        if s['shape_type']=='point': d[int(s['label'][:2])].append(s['points'][0])
    return dict(d)
def cost(anns,cameras,cal,ids):
    values=[]
    for c,d in itertools.combinations(cameras,2):
        f=fundamental(np.array(cal[c]['K']),np.array(cal[c]['world_to_camera_mm']),np.array(cal[d]['K']),np.array(cal[d]['world_to_camera_mm']))
        for k in ids:
            if len(anns[c].get(k,[]))==len(anns[d].get(k,[]))==1:
                values.append(epi(anns[c][k][0],anns[d][k][0],f))
    return dict(n=len(values),mean_clipped100=float(np.minimum(values,100).mean()) if values else None,median=float(np.median(values)) if values else None)
def panel(src,ann,title,ids):
    im=Image.open(src).convert('RGB'); p=points(ann)
    coords=[xy for k,xs in p.items() for xy in xs]
    if coords:
        xy=np.array(coords); lo=np.maximum(xy.min(0)-80,0).astype(int); hi=np.minimum(xy.max(0)+80,im.size).astype(int)
        box=(*lo,*hi)
    else: box=(0,0,*im.size)
    draw=ImageDraw.Draw(im)
    colors={6:'#ff70ff',7:'#ffff40',15:'#40ff60',16:'#ff9950',17:'#40ddff',18:'#ff60a0'}
    for k,xs in p.items():
        if k not in ids: continue
        for i,(x,y) in enumerate(xs):
            color=colors.get(k,'cyan');draw.ellipse((x-6,y-6,x+6,y+6),outline=color,width=3)
            label=f'{k:02d}'+(chr(97+i) if len(xs)>1 else '')
            draw.text((max(0,min(im.width-90,x+10)),max(0,min(im.height-40,y-32))),label,font=ImageFont.truetype(FONT,25),fill=color,stroke_width=2,stroke_fill='black')
    im=im.crop(box); im.thumbnail((490,480))
    canvas=Image.new('RGB',(500,540),'#18212d');canvas.paste(im,((500-im.width)//2,55))
    ImageDraw.Draw(canvas).text((8,10),title,font=ImageFont.truetype(FONT,21),fill='white')
    return canvas
def main():
    assert not DEST.exists() and not OUT.exists(), 'Refuse overwrite'
    DEST.mkdir(); OUT.mkdir()
    with zipfile.ZipFile(ZIP) as z:
        assert z.testzip() is None
        for info in z.infolist():
            parts=PurePosixPath(info.filename.replace('\\','/')).parts
            assert parts and not any(p in ('..','/') or ':' in p for p in parts)
            assert not stat.S_ISLNK(info.external_attr>>16)
            target=DEST.joinpath(*parts).resolve();assert target.is_relative_to(DEST.resolve())
        z.extractall(DEST)
    pkg=next(p for p in DEST.iterdir() if p.is_dir())
    manifest=json.loads((OLD/'review_manifest.json').read_text())
    summary=dict(archive=str(ZIP),sha256=sha(ZIP),extracted=str(pkg),training_modified=False,calibration_modified=False,holdout_use='returned-label QC only; no prediction, selection or tuning',cases=[],files=[],issues=[],duplicate_copy_conflicts=[])
    by_source=collections.defaultdict(list); olddup=newdup=0; changed_points=collections.Counter(); sections=[]
    for case in manifest['cases']:
        folder=pkg/case['folder']; oldanns={};newanns={};changed=[]; pics=[]
        for src in sorted((folder/'originals').rglob('*.json')):
            rel=src.relative_to(folder/'originals'); old=BASE/rel; a=json.loads(old.read_text());b=json.loads(src.read_text()); pa,pb=points(a),points(b)
            cam=src.stem[3:];oldanns[cam]=pa;newanns[cam]=pb
            png=src.with_suffix('.png'); assert png.exists()
            with Image.open(png) as im: im.load(); size=im.size
            same_image=sha(png)==sha(old.with_suffix('.png'))
            assert same_image and size==(1080,1200)
            problems=[]
            for k,xys in pb.items():
                if not 1<=k<=23:problems.append(f'unknown id{k}')
                if len(xys)>1:problems.append(f'duplicate id{k}')
                for xy in xys:
                    if len(xy)!=2 or not np.isfinite(xy).all() or not (0<=xy[0]<size[0] and 0<=xy[1]<size[1]):problems.append(f'invalid coordinate id{k}')
            if (b.get('imageWidth'),b.get('imageHeight'))!=size:problems.append('JSON dimensions mismatch')
            olddup+=sum(len(v)>1 for v in pa.values());newdup+=sum(len(v)>1 for v in pb.values())
            delta=[dict(id=k,before=pa.get(k,[]),after=pb.get(k,[])) for k in sorted(pa.keys()|pb.keys()) if pa.get(k,[])!=pb.get(k,[])]
            changed_points.update(r['id'] for r in delta)
            row=dict(case=case['id'],source=rel.as_posix(),camera=cam,raw_json_changed=sha(src)!=sha(old),changes=delta,issues=problems,image_sha256=sha(png),returned_json_sha256=sha(src),old_json_sha256=sha(old))
            summary['files'].append(row);by_source[str(rel)].append((str(src),pb))
            if delta:changed.append(row)
            if problems:summary['issues'].append(row)
            selected=set(r['id'] for r in delta) or ({17,18} if case.get('pair')=='hindpaws' else {15,16})
            if case['kind']=='duplicate':selected|={int(r['label'][:2]) for r in case['points']}
            pics.extend([panel(png,a,f"{case['id']} cam{cam} 返修前",selected),panel(png,b,f"{case['id']} cam{cam} 返修后",selected)])
        rec=dict(id=case['id'],session=case['session'],frame=case['frame'],split=case['split'],kind=case['kind'],changed_images=len(changed),changed_ids=sorted({r['id'] for row in changed for r in row['changes']}))
        if case['kind']=='correspondence':
            cal=json.loads((BASE/'calibration'/case['session']/'rectified_calibration.json').read_text());ids=[17,18] if case['pair']=='hindpaws' else [15,16]
            rec['before_epipolar']=cost(oldanns,case['paired_cameras'],cal,ids);rec['after_epipolar']=cost(newanns,case['paired_cameras'],cal,ids)
        summary['cases'].append(rec)
        sheet=Image.new('RGB',(1000,len(pics)//2*540),'#18212d')
        for i,p in enumerate(pics):sheet.paste(p,(i%2*500,i//2*540))
        file=f"{case['id']}_before_after.jpg";sheet.save(OUT/file,quality=93)
        sections.append(f'<h2>{case["id"]} {case["session"]} / {case["frame"]}</h2><pre>{html.escape(json.dumps(rec,ensure_ascii=False,indent=2))}</pre><a href="{file}"><img loading="lazy" src="{file}"></a>')
        print(rec,flush=True)
    for rel,copies in by_source.items():
        if any(p!=copies[0][1] for _,p in copies[1:]):summary['duplicate_copy_conflicts'].append(dict(source=rel,copies=[p for p,_ in copies]))
    summary['counts']=dict(cases=len(summary['cases']),image_json_pairs=len(summary['files']),unique_images=len(by_source),coordinate_changed_images=sum(bool(r['changes']) for r in summary['files']),changed_cases=sum(c['changed_images']>0 for c in summary['cases']),duplicate_slots_before=olddup,duplicate_slots_after=newdup,changed_id_image_counts=dict(changed_points))
    summary['feedback_template']=(pkg/'复核反馈模板.txt').read_text(encoding='utf-8-sig')
    summary['warning']='Original returned JPG overviews may be stale. Newly generated before/after figures use actual returned JSON. No automatic acceptance, label merge, or exclusion removal.'
    write(OUT/'summary.json',summary)
    page='<meta charset="utf-8"><title>返修包核验</title><style>body{font:17px sans-serif;max-width:1100px;margin:28px auto}img{max-width:100%}pre{white-space:pre-wrap}</style><h1>2026-10-02 返修包核验</h1><p>原图/原标签保留；本页由实际返修JSON重新绘制。几何只是诊断，不是正确性证明。留出集仅返修质量检查，不参与训练或选型。</p>'
    page+='<pre>'+html.escape(json.dumps(summary['counts'],ensure_ascii=False,indent=2))+'</pre>'+''.join(sections)
    (OUT/'index.html').write_text(page,encoding='utf-8')
    export=DOWNLOAD/'PyMouseStudio-Repair-Audit-20261002';assert not export.exists();shutil.copytree(OUT,export)
    write(DEST/'audit_manifest.json',dict(archive_sha256=sha(ZIP),report=str(OUT/'summary.json'),training_allowed=False))
    print(json.dumps(summary['counts'],ensure_ascii=False,indent=2))
if __name__=='__main__': main()
