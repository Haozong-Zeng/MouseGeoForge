"""Safely extract and review a corrected development-only annotation delivery.

Raw files are immutable observations; no training labels are imported or fixed.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=Path('/mnt/c/Users/55395/Downloads/real_development---ok.zip')
EXTRACT=ROOT/'dataset/PyMouseStudio/annotation_returns/20260914_real_development_corrected'
OUTPUT=ROOT/'output/annotation_review_20260914_real_development_corrected'
OLD=ROOT/'dataset/PyMouseStudio/annotation_returns/20260913_frame000000/frame_000000---ok'
REQUIRED={1,6,7,8,14,15,16,17,18,23}


def digest(data):return hashlib.sha256(data).hexdigest()


def proper_crossings(poly):
    """Strict nonadjacent segment crossings only; not a full polygon validity proof."""
    p=np.asarray(poly,float);out=[];n=len(p)
    def cross(a,b,c):
        u=b-a;v=c-a;return u[0]*v[1]-u[1]*v[0]
    for i in range(n):
        for j in range(i+2,n):
            if i==0 and j==n-1:continue
            a,b=p[i],p[(i+1)%n];c,d=p[j],p[(j+1)%n]
            if cross(a,b,c)*cross(a,b,d)<-1e-8 and cross(c,d,a)*cross(c,d,b)<-1e-8:out.append([i,j])
    return out


def main():
    assert not EXTRACT.exists() and not OUTPUT.exists(),'Use a new review directory; never overwrite returns.'
    with zipfile.ZipFile(ARCHIVE) as z:
        assert z.testzip() is None
        infos=z.infolist();assert sum(i.file_size for i in infos)<100*1024**2
        for item in infos:
            name=item.filename.replace('\\','/');p=PurePosixPath(name)
            assert not p.is_absolute() and '..' not in p.parts and ':' not in name
            assert not stat.S_ISLNK(item.external_attr>>16)
        EXTRACT.mkdir(parents=True)
        z.extractall(EXTRACT)
        raw_hashes={i.filename:digest(z.read(i)) for i in infos if not i.is_dir()}
        for name,h in raw_hashes.items():assert digest((EXTRACT/name).read_bytes())==h
    OUTPUT.mkdir(parents=True)
    source=EXTRACT/'real_development---ok'
    all_rows=[]
    for folder in sorted(source.glob('frame_*')):
        assert {p.stem for p in folder.glob('*.png')}=={p.stem for p in folder.glob('*.json')}=={f'cam{x}' for x in 'ABCDEFG'}
        out=OUTPUT/folder.name
        result=subprocess.run([sys.executable,str(ROOT/'tools/review_pymousestudio_labelme_return.py'),
            '--source',str(folder),'--output',str(out)],capture_output=True,text=True)
        if result.returncode:raise RuntimeError(result.stderr+result.stdout)
        audit=json.loads((out/'audit.json').read_text())
        rows=[]
        for row in audit['views']:
            camera=row['camera'];data=json.loads((folder/(camera+'.json')).read_text())
            frame=int(folder.name.split('_')[-1]);assert row['source']['frame']==frame
            points={int(s['label'][:2]):s['points'][0] for s in data['shapes'] if s['shape_type']=='point'}
            polys=[s for s in data['shapes'] if s['shape_type']=='polygon']
            row.update(frame=frame,required_ids_missing=sorted(REQUIRED-set(points)),
                deferred_ids_with_coordinates=sorted(set(points)-REQUIRED),
                surface_points={str(k):points.get(k) for k in [6,7,23]},
                polygon_proper_crossings=[proper_crossings(s['points']) for s in polys],
                polygon_labels=[s['label'] for s in polys],
                image_flags=data.get('flags',{}))
            before=OLD/(camera+'.json')
            if frame==0 and before.exists():
                old=json.loads(before.read_text());op={int(s['label'][:2]):s['points'][0] for s in old['shapes'] if s['shape_type']=='point'}
                oldpoly=[s for s in old['shapes'] if s['shape_type']=='polygon']
                moved={str(k):dict(old=op[k],new=points[k],distance_px=float(np.linalg.norm(np.array(points[k])-op[k])))
                       for k in points.keys()&op.keys() if points[k]!=op[k]}
                row['previous_return_comparison']=dict(added_ids=sorted(points.keys()-op.keys()),
                    removed_ids=sorted(op.keys()-points.keys()),moved=moved,
                    polygon_coordinates_unchanged=[p['points'] for p in polys]==[p['points'] for p in oldpoly],
                    original_image_unchanged=digest((folder/(camera+'.png')).read_bytes())==digest((OLD/(camera+'.png')).read_bytes()))
            rows.append(row)
        # Replace inherited first-frame page title with verified per-frame provenance.
        links=''.join(f'<h2>{r["camera"]}</h2><img style="max-width:100%" src="{r["camera"]}_detail.jpg">' for r in rows)
        (out/'index.html').write_text(f'<!doctype html><meta charset="utf-8"><title>{folder.name} corrected return</title>'
            f'<body style="background:#18202b;color:white;font-family:sans-serif"><h1>AZL3 {folder.name}</h1>'
            '<p>Raw observations only; visibility UNKNOWN. Old point-name aliases retained, new surface semantics pending review. No training import.</p>'+links)
        all_rows.extend(rows)
    summary=dict(archive=str(ARCHIVE),archive_sha256=digest(ARCHIVE.read_bytes()),archive_crc='passed',
                 extract_root=str(EXTRACT),raw_files=len(raw_hashes),image_count=len(all_rows),
                 frame_count=2,total_points=sum(r['point_count'] for r in all_rows),
                 train_allowed=False,views=all_rows,raw_sha256=raw_hashes,
                 semantic_policy='Legacy label names parsed by stable slot only; not automatic acceptance of new anatomy.',
                 polygon_test_scope='Strict proper crossings only; collinear overlaps/touches are not fully audited.')
    for name,h in raw_hashes.items():assert digest((EXTRACT/name).read_bytes())==h
    (OUTPUT/'audit_corrected.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    links=''.join(f'<li><a href="{f}/index.html">{f}: 7 views</a></li>' for f in ['frame_000000','frame_000064'])
    (OUTPUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Corrected real development annotation review</title>'
        '<body style="font-family:sans-serif"><h1>Corrected development annotations</h1><p>14 images, no training import.</p><ul>'+links+'</ul>')
    # Full-body small panels for overview; detail files preserve larger pixel scale.
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
    for frame in ['frame_000000','frame_000064']:
        sheet=Image.new('RGB',(1400,4*560),'#18202b');draw=ImageDraw.Draw(sheet)
        for i,camera in enumerate('ABCDEFG'):
            image=Image.open(OUTPUT/frame/f'cam{camera}_detail.jpg');image.thumbnail((690,520))
            x=(i%2)*700;y=(i//2)*560
            draw.text((x+10,y+8),f'{frame} cam{camera}: original / annotation',font=font,fill='white')
            sheet.paste(image,(x+(700-image.width)//2,y+36))
        sheet.save(OUTPUT/f'{frame}_overview.jpg',quality=95)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('raw_sha256','views')},indent=2))
    for r in all_rows:
        print(json.dumps({k:v for k,v in r.items() if k in ['frame','camera','point_count','required_ids_missing','errors','points_outside_polygon','surface_points','polygon_proper_crossings','previous_return_comparison']},ensure_ascii=False))


if __name__=='__main__':main()
