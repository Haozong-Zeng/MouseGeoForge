"""True held-view diagnostic panels: primary, multi-peak and joint assignment."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from run_teacher_cycle_dlc_20261001 import ROOT,OUT
BASE=ROOT/'dataset/PyMouseStudio/annotation_returns/20260930_full_v2'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='joint_paw_assignment_v1');ap.add_argument('--remaining',action='store_true');args=ap.parse_args()
    dest=OUT/args.run/'review_v2';dest.mkdir(exist_ok=True)
    m=json.loads((OUT/'dlc_dataset_manifest.json').read_text());entries={(e['session'],e['frame'],e['camera']):e for e in m['entries'] if e['role']=='validation'}
    groups={(g['session'],g['frame']):g for g in json.loads((BASE/'ingestion_manifest.json').read_text())['frame_groups']}
    font=ImageFont.truetype('/mnt/c/Windows/Fonts/arial.ttf',19);cases=[]
    for cohort in (['remaining'] if args.remaining else ['pilot','stress1']):
        rows=json.loads((OUT/args.run/('observations.json' if args.remaining else f'{cohort}_observations.json')).read_text())
        worst=sorted([r for r in rows if r['joint'] is not None],key=lambda r:-r['joint'])[:2]
        gains=sorted([r for r in rows if r['joint'] is not None and r['merged_top3'] is not None and r['joint']<r['merged_top3']-1],key=lambda r:r['joint']-r['merged_top3'])[:2]
        rejected=sorted([r for r in rows if r['joint'] is None and r['merged_top3'] is not None],key=lambda r:-r['merged_top3'])[:2]
        seen=set()
        for kind,selected in [('worst',worst),('gain_vs_top3',gains),('rejected_not_corrected',rejected)]:
            for r in selected:
                key=(r['session'],r['frame'],r['camera'],r['id'])
                if key in seen:continue
                seen.add(key);e=entries[key[:3]];g=groups[key[:2]]
                with np.load(BASE/g['normalized_file']) as z:gt=z['uv'][g['cameras'].index(r['camera']),0]
                im=Image.open(ROOT/e['source']).convert('RGB').resize((450,500));canvas=Image.new('RGB',(1350,580),'#111111');draw=ImageDraw.Draw(canvas)
                for i,(method,label) in enumerate([('merged_top1','Primary peak'),('merged_top3','Independent top3'),('joint','Joint identity')]):
                    panel=im.copy();d=ImageDraw.Draw(panel)
                    for k in [15,16,17,18]:
                        v=gt[k-1]
                        if not np.isfinite(v).all():continue
                        x,y=v*np.array([450/1080,500/1200]);color='lime' if k==r['id'] else 'white'
                        d.ellipse((x-6,y-6,x+6,y+6),outline=color,width=3);d.text((x+6,y-22),str(k),font=font,fill=color,stroke_width=1,stroke_fill='black')
                    v=np.array(r[method+'_uv'])
                    if np.isfinite(v).all():
                        x,y=v*np.array([450/1080,500/1200]);d.line((x-7,y-7,x+7,y+7),fill='red',width=3);d.line((x-7,y+7,x+7,y-7),fill='red',width=3)
                    canvas.paste(panel,(i*450,80));err=r[method];et=f'{err:.1f}px' if err is not None else 'UNKNOWN'
                    draw.text((i*450+5,4),label+': '+et,font=font,fill='white')
                    draw.text((i*450+5,29),f'{r["session"]} f{r["frame"]} cam{r["camera"]} k{r["id"]}',font=font,fill='yellow')
                    draw.text((i*450+5,54),'HELD OUT | green manual / red inferred',font=font,fill='lime')
                name=f'{cohort}_{kind}_{r["session"]}_f{r["frame"]}_cam{r["camera"]}_k{r["id"]}.jpg';canvas.save(dest/name,quality=94)
                cases.append(dict(file=name,cohort=cohort,selection=kind,top1=r['merged_top1'],top3=r['merged_top3'],joint=r['joint']))
    (dest/'cases.json').write_text(json.dumps(cases,indent=2));(dest/'README.txt').write_text('TrueLOO projected candidates. Held camera entirely excluded from reconstruction and joint selection. Green target manual, white other manual paws, red inferred target. Selected two worst accepted joint cases, up to two actual improvements>1px, and two largest rejected errors per cohort. For original pilot/stress there was no actual>1px common-point improvement; older gain-named examples had zero gain. No GT released.\n')
    export=Path('/mnt/c/Users/55395/Downloads/MouseGeoForge-TeacherTemporal-20261001')/(args.run+'_review_v2');shutil.copytree(dest,export,dirs_exist_ok=True);print(json.dumps(cases,indent=2))

if __name__=='__main__':main()
