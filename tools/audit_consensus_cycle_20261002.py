"""Inspect accepted-but-wrong consensus and manual-only correspondence controls."""
import json,shutil
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from consensus_geometry_cycle_20261002 import solve,Geometry,CONFIGS,ROOT,OUT,BASE
from audit_teacher_dlc_predictions_20261001 import stats

def main():
    manual=json.loads((BASE/'ingestion_manifest.json').read_text())
    groups={(g['session'],g['frame']):g for g in manual['frame_groups']}
    data=json.loads((OUT/'dlc_dataset_manifest.json').read_text());ids=data['ids_1based']
    entries={(e['session'],e['frame'],e['camera']):e for e in data['entries'] if e['role']=='validation'}
    font=ImageFont.truetype('/mnt/c/Windows/Fonts/arial.ttf',18)
    outputs={}
    for run,suffix in [('consensus_pilot_v1_final','_exactnorm'),('consensus_stress1_v1','_stress1_exactnorm')]:
        rows=json.loads((OUT/run/'observations.json').read_text());cache={}
        manifests=[json.loads((OUT/f'continuous_lp_seed{s}_kl{suffix}/manifest.json').read_text()) for s in [17,31,73]]
        for ci,clip in enumerate(manifests[0]):
            session=clip['session'];ps=[]
            for m in manifests:
                with np.load(Path(m[ci]['folder'])/'predictions_source_pixels.npz') as z:ps.append(dict(z))
            cams=ps[0]['cameras'].tolist();raw=np.stack([p['uv'] for p in ps]);conf=np.mean([p['confidence'] for p in ps],axis=0)
            cache[session]=dict(cams=cams,frames=ps[0]['source_frames'].tolist(),uv=np.median(raw,axis=0),conf=conf,
                spread=np.sqrt(np.square(raw-raw.mean(0)).sum(-1).mean(0)),var=np.var(raw,axis=0)/np.maximum(conf[...,None],1e-12),
                geo=Geometry(json.loads((BASE/'calibration'/session/'rectified_calibration.json').read_text()),cams))
        enriched=[]
        for row in rows:
            r=dict(row);p=cache[r['session']];cams=p['cams'];vi=cams.index(r['camera']);j=ids.index(r['id'])
            g=groups[(r['session'],r['frame'])]
            with np.load(BASE/g['normalized_file']) as z:gt=z['uv'][:,0,r['id']-1]
            uv=np.full((len(cams),2),np.nan)
            for gi,c in enumerate(g['cameras']):
                if c in cams:uv[cams.index(c)]=gt[gi]
            others=[i for i in range(len(cams)) if i!=vi and np.isfinite(uv[i]).all()]
            x,meta=p['geo'].solve(uv,np.ones_like(uv)*16,np.ones(len(cams)),np.zeros(len(cams)),others,CONFIGS['weighted_robust_floor4'])
            r['manual_only_loo']=float(np.linalg.norm(p['geo'].project(x,np.array([vi]))[0][0]-uv[vi])) if np.isfinite(x).all() else None
            r['manual_other_views']=[cams[i] for i in others];enriched.append(r)
        accepted=[r for r in enriched if r['consensus'] is not None]
        worst=sorted(accepted,key=lambda r:r['consensus'],reverse=True)[:4]
        report=dict(accepted=len(accepted),over50px=sum(r['consensus']>50 for r in accepted),over100px=sum(r['consensus']>100 for r in accepted),
            worst=worst,by_camera={c:stats([r['consensus'] for r in accepted if r['camera']==c]) for c in sorted({r['camera'] for r in rows})},
            by_keypoint={str(k):stats([r['consensus'] for r in accepted if r['id']==k]) for k in ids},
            manual_only_all=stats([r['manual_only_loo'] for r in enriched if r['manual_only_loo'] is not None]),
            manual_only_on_consensus_retained=stats([r['manual_only_loo'] for r in accepted if r['manual_only_loo'] is not None]),
            warning='Manual-only control not independent3D truth or error floor; minimum two manual other views, no modifications. Images deliberately show worst accepted LOO errors, not typical performance.',train_allowed=False)
        dest=OUT/run/'failure_review';dest.mkdir(exist_ok=True)
        for rank,r in enumerate(worst,1):
            p=cache[r['session']];cams=p['cams'];fi=p['frames'].index(r['frame']);j=ids.index(r['id']);vi=cams.index(r['camera'])
            x,meta=solve(p['geo'],p['uv'][:,fi,j],p['var'][:,fi,j],p['conf'][:,fi,j],p['spread'][:,fi,j],[i for i in range(len(cams)) if i!=vi])
            projections=p['geo'].project(x,np.arange(len(cams)))[0]
            g=groups[(r['session'],r['frame'])]
            with np.load(BASE/g['normalized_file']) as z:gt=z['uv'][:,0,r['id']-1]
            canvas=Image.new('RGB',(1440,((len(cams)+2)//3)*580),'#111111');d=ImageDraw.Draw(canvas)
            for i,c in enumerate(cams):
                e=entries.get((r['session'],r['frame'],c));ox=(i%3)*480;oy=(i//3)*580
                if e is None:continue
                im=Image.open(ROOT/e['source']).convert('RGB').resize((450,500));draw=ImageDraw.Draw(im)
                def point(v,color,shape):
                    if not np.isfinite(v).all():return
                    a,b=v*np.array([450/1080,500/1200])
                    if shape=='circle':draw.ellipse((a-6,b-6,a+6,b+6),outline=color,width=2)
                    else:draw.line((a-6,b-6,a+6,b+6),fill=color,width=3);draw.line((a-6,b+6,a+6,b-6),fill=color,width=3)
                if c in g['cameras']:point(gt[g['cameras'].index(c)],'white','circle')
                point(p['uv'][i,fi,j],'cyan','circle');point(projections[i],'red','cross')
                canvas.paste(im,(ox+10,oy+55));status='HELD OUT' if i==vi else ('INLIER' if i in meta['inliers'] else 'REJECTED')
                d.text((ox+10,oy+5),f'{r["session"]} f{r["frame"]} cam{c} k{r["id"]} {status}',font=font,fill='yellow' if i==vi else 'white')
                d.text((ox+10,oy+29),f'conf {p["conf"][i,fi,j]:.2f} spread {p["spread"][i,fi,j]:.1f}px',font=font,fill='cyan')
            name=f'{rank:02d}_{r["session"]}_f{r["frame"]}_k{r["id"]}_held{r["camera"]}.jpg';canvas.save(dest/name,quality=93)
        (dest/'summary.json').write_text(json.dumps(report,indent=2));(dest/'observations.json').write_text(json.dumps(enriched,indent=2))
        (dest/'README.txt').write_text('White ring: original manual point; cyan ring: ensemble2D; red cross: consensus3D projected to all cameras, fit excludes HELD OUT view entirely.\nTop4 accepted-but-worst cases selected by manual held-view error. NOT GT.\n')
        export=Path('/mnt/c/Users/55395/Downloads/MouseGeoForge-TeacherTemporal-20261001')/run/'failure_review'
        shutil.copytree(dest,export,dirs_exist_ok=True)
        outputs[run]={k:v for k,v in report.items() if k not in ['worst']}
        print(run,'WORST',[(r['session'],r['frame'],r['camera'],r['id'],round(r['consensus'],1),r['manual_only_loo']) for r in worst])
    print(json.dumps(outputs,indent=2))

if __name__=='__main__':main()
