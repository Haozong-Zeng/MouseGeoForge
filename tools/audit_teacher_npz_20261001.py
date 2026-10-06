"""Source-pixel metrics and paired visual review for any fixed teacher NPZ."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from audit_teacher_dlc_predictions_20261001 import stats
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/teacher_temporal_cycle_20261001'

def main():
    pa=argparse.ArgumentParser();pa.add_argument('run');args=pa.parse_args();assert '/' not in args.run
    m=json.loads((OUT/'dlc_dataset_manifest.json').read_text())
    with np.load(OUT/args.run/'predictions_source_pixels.npz') as z:p=dict(z)
    err=np.linalg.norm(p['uv']-p['gt'],axis=-1);records=[]
    for i,e in enumerate(m['entries']):
        for j,k in enumerate(m['ids_1based']):
            if np.isfinite(err[i,j]):records.append(dict(role=e['role'],camera=e['camera'],session=e['session'],frame=e['frame'],id=k,error_px=float(err[i,j]),confidence=float(p['confidence'][i,j])))
    report={'units':'original1080x1200 pixels','run':args.run,'train_allowed':False}
    for role in ['train','validation']:
        rows=[r for r in records if r['role']==role]
        report[role]=dict(all=stats([r['error_px'] for r in rows]),
            cameras={c:stats([r['error_px'] for r in rows if r['camera']==c]) for c in sorted({r['camera'] for r in rows})},
            confidence_curve={str(t):dict(coverage=sum(r['confidence']>=t for r in rows)/len(rows),errors=stats([r['error_px'] for r in rows if r['confidence']>=t])) for t in [0,.3,.6,.8,.9,.99]})
    dest=OUT/args.run/'review';dest.mkdir(exist_ok=True)
    chosen=[next(i for i,e in enumerate(m['entries']) if (e['session'],e['frame'],e['camera'])==s) for s in [('FR2',1299,'A'),('AZR2L2',259,'E'),('FR2',909,'F')]]
    candidates=sorted(m['validation_indices'],key=lambda i:np.nanmean(err[i]))
    chosen+=candidates[-3:]+candidates[len(candidates)//2:len(candidates)//2+2]
    shared=OUT/args.run/'shared_failure_cases.json'
    if shared.exists():
        cases=sorted(json.loads(shared.read_text()),key=lambda r:r['error_px'],reverse=True)
        for r in cases[:4]:
            chosen.append(next(i for i,e in enumerate(m['entries']) if (e['session'],e['frame'],e['camera'])==(r['session'],r['frame'],r['camera'])))
    font=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',21)
    images=[]
    for i in dict.fromkeys(chosen):
        e=m['entries'][i];im=Image.open(ROOT/e['source']).convert('RGB');d=ImageDraw.Draw(im)
        for j,k in enumerate(m['ids_1based']):
            x,y=p['gt'][i,j];u,v=p['uv'][i,j]
            if not np.isfinite([x,y,u,v]).all():continue
            d.ellipse((x-6,y-6,x+6,y+6),outline='lime',width=3);d.text((x+8,y-20),str(k),fill='lime',font=font,stroke_width=1,stroke_fill='black')
            d.line((x,y,u,v),fill='magenta',width=2);d.line((u-6,v-6,u+6,v+6),fill='magenta',width=3);d.line((u-6,v+6,u+6,v-6),fill='magenta',width=3)
            d.text((u+7,v+3),f'{k}:{p["confidence"][i,j]:.2f}',fill='magenta',font=font,stroke_width=1,stroke_fill='black')
        name=f'{e["session"]}_{e["frame"]:06d}_{e["camera"]}.jpg';im.save(dest/name,quality=94);images.append(name)
    (dest/'summary.json').write_text(json.dumps(report,indent=2));(dest/'observations.json').write_text(json.dumps(records,indent=2))
    (dest/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>'+args.run+'</h1><p>Green: human. Magenta: teacher. Confidence is NOT calibrated probability. Three fixed comparison cases, three worst cases, two median cases, followed by shared-error examples when available. NOT GT.</p>'+''.join(f'<h2>{x}</h2><img width="720" src="{x}">' for x in images))
    export=Path('/mnt/c/Users/55395/Downloads/MouseGeoForge-TeacherTemporal-20261001')/f'{args.run}_review'
    shutil.copytree(dest,export,dirs_exist_ok=True)
    print(json.dumps(report['validation'],indent=2))

if __name__=='__main__':main()
