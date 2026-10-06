"""Isolated 56-image return audit; raw annotations never changed or imported."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from review_corrected_annotation_return_20260914 import proper_crossings, REQUIRED

ROOT=Path(__file__).resolve().parents[1]
EXTRACT=ROOT/'dataset/PyMouseStudio/annotation_returns/20260914_real_images_56'
OUT=ROOT/'output/annotation_review_20260914_real_images_56'
PREVIOUS=ROOT/'dataset/PyMouseStudio/annotation_returns/20260914_real_development_corrected/real_development---ok'

def sha(b):return hashlib.sha256(b).hexdigest()

def main():
    archive=next(Path('/mnt/c/Users/55395/Downloads').glob('*56*OK.zip'))
    assert not EXTRACT.exists() and not OUT.exists(),'Do not overwrite existing returns.'
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert sum(i.file_size for i in z.infolist())<200*1024**2
        for i in z.infolist():
            n=i.filename.replace('\\','/');p=PurePosixPath(n)
            assert not p.is_absolute() and '..' not in p.parts and ':' not in n
            assert not stat.S_ISLNK(i.external_attr>>16)
        EXTRACT.mkdir(parents=True);z.extractall(EXTRACT)
        hashes={i.filename:sha(z.read(i)) for i in z.infolist() if not i.is_dir()}
    OUT.mkdir(parents=True)
    source=next(p for p in EXTRACT.iterdir() if p.is_dir())
    frames=sorted(source.glob('frame_*'));assert len(frames)==8
    rows=[]
    font=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',23)
    for folder in frames:
        assert {p.stem for p in folder.glob('*.png')}=={p.stem for p in folder.glob('*.json')}=={f'cam{x}' for x in 'ABCDEFG'}
        dest=OUT/folder.name
        run=subprocess.run([sys.executable,str(ROOT/'tools/review_pymousestudio_labelme_return.py'),
            '--source',str(folder),'--output',str(dest)],text=True,capture_output=True)
        if run.returncode:raise RuntimeError(run.stderr+run.stdout)
        audit=json.loads((dest/'audit.json').read_text())
        sheet=Image.new('RGB',(1200,1810),'#17212c');draw=ImageDraw.Draw(sheet)
        draw.text((15,10),f"{folder.name} | 原始回标叠加；仅检查，未导入训练",font=font,fill='white')
        for i,row in enumerate(audit['views']):
            cam=row['camera'];p=folder/(cam+'.json');a=json.loads(p.read_text())
            frame=int(folder.name.split('_')[-1]);assert row['source']['frame']==frame
            points={int(s['label'][:2]):s['points'][0] for s in a['shapes'] if s['shape_type']=='point'}
            polygons=[s for s in a['shapes'] if s['shape_type']=='polygon']
            row.update(frame=frame,missing_active_coordinate_ids=sorted(REQUIRED-set(points)),
                       polygon_proper_crossings=[proper_crossings(s['points']) for s in polygons],
                       surface_points={str(k):points.get(k) for k in [6,7,23]},
                       tail_tip=points.get(14),single_instance_id_can_be_assigned=0)
            old=PREVIOUS/folder.name/p.name
            if old.exists():
                b=json.loads(old.read_text());bp={int(s['label'][:2]):s['points'][0] for s in b['shapes'] if s['shape_type']=='point'}
                row['previous_return_comparison']=dict(
                    json_bytes_identical=p.read_bytes()==old.read_bytes(),
                    shapes_identical=a['shapes']==b['shapes'],
                    removed_ids=sorted(bp.keys()-points.keys()),added_ids=sorted(points.keys()-bp.keys()),
                    moved={str(k):dict(old=bp[k],new=points[k]) for k in bp.keys()&points.keys() if bp[k]!=points[k]},
                    polygon_coordinates_unchanged=[s['points'] for s in polygons]==[s['points'] for s in b['shapes'] if s['shape_type']=='polygon'])
            rows.append(row)
            overlay=Image.open(dest/(cam+'_overlay.png')).convert('RGB')
            mask=np.array(Image.open(dest/(cam+'_mask.png')));yy,xx=np.where(mask>0)
            crop=(max(0,int(xx.min())-22),max(0,int(yy.min())-22),min(overlay.width,int(xx.max())+23),min(overlay.height,int(yy.max())+23))
            overlay=overlay.crop(crop);overlay.thumbnail((580,385))
            x=(i%2)*600;y=50+(i//2)*435
            draw.text((x+12,y),f"{cam} · {row['point_count']}点",font=font,fill='white')
            sheet.paste(overlay,(x+(600-overlay.width)//2,y+36))
        sheet.save(OUT/(folder.name+'_overview.jpg'),quality=96)
        links=''.join(f'<h2>cam{c}</h2><img style="max-width:100%" src="cam{c}_detail.jpg">' for c in 'ABCDEFG')
        (dest/'index.html').write_text(f'<!doctype html><meta charset="utf-8"><title>{folder.name}</title><body style="background:#17212c;color:white;font-family:sans-serif"><h1>AZL3 {folder.name}</h1><p>Single mouse. Unmarked active points have no coordinate supervision; deferred slots remain excluded. No training import.</p>'+links)
        print('REVIEWED',folder.name,'points',sum(r['point_count'] for r in audit['views']),flush=True)
    for n,h in hashes.items():assert sha((EXTRACT/n).read_bytes())==h
    summary=dict(archive=str(archive),archive_sha256=sha(archive.read_bytes()),crc_passed=True,
        extract_root=str(source),image_count=len(rows),frame_numbers=[int(p.name.split('_')[-1]) for p in frames],
        total_points=sum(r['point_count'] for r in rows),train_allowed=False,raw_sha256=hashes,views=rows,
        interpretation='Single known mouse: instance_id=0 may be assigned in conversion; absent point is not a coordinate target. Visibility interpretation requires visible-only completion convention; deferred points excluded.')
    (OUT/'audit_56.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>56-image annotation review</title><h1>AZL3: 8 frames / 56 views</h1><p>Review only; original labels unchanged.</p>'+''.join(f'<h2><a href="{p.name}/index.html">{p.name}</a></h2><img style="max-width:100%" src="{p.name}_overview.jpg">' for p in frames))
    print('TOTAL',summary['total_points'],'points',len(rows),'images')
    for r in rows:
        if r['errors'] or r['points_outside_polygon'] or (r['tail_tip_edge_distance_px'] is not None and r['tail_tip_edge_distance_px']<10) or any(r['polygon_proper_crossings']):
            print('FLAG',json.dumps({k:r[k] for k in ['frame','camera','errors','points_outside_polygon','tail_tip_edge_distance_px','polygon_proper_crossings']}))

if __name__=='__main__':main()
