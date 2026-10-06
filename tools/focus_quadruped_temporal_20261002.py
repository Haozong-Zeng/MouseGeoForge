"""Adjacent-frame inspection around existing manual errors, without correction."""
import json
from pathlib import Path
import shutil
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from bottom_anchor_common_20261002 import ROOT,OUT,budget,write_state
from review_quadruped_temporal_20261002 import COLORS


def main():
    _,_,check=budget(120);check()
    run=OUT/'quadruped_temporal_frozen_v1';dest=OUT/'quadruped_temporal_review_v1'
    assert not (dest/'focused_cases.json').exists()
    records=json.loads((run/'manifest.json').read_text())
    dm=json.loads((OUT/'superanimal_quadruped_hrnet_v1/dataset_manifest.json').read_text())
    entries=[dm['entries'][i] for i in dm['validation_indices']]
    lookup={(e['session'],e['camera'],int(e['frame'])):i for i,e in enumerate(entries)}
    with np.load(OUT/'quadruped_epoch110_review_v1/predictions_source_pixels.npz') as z:gt=z['gt'].copy()
    A=np.array(dm['affine_source_to_input']);scale=np.diag(A)[:2];ids=dm['ids_1based']
    font=ImageFont.truetype('/mnt/c/Windows/Fonts/arial.ttf',16)
    focused=[]
    for cohort,session,camera,center,k in [('pilot','AZR2L2','E',129,15),('pilot','FR2','F',844,18),('stress1','FR2','E',1104,7),('stress1','FR2','A',1299,1)]:
        check();rec=next(r for r in records if r['seed']==17 and r['clip']['cohort']==cohort and r['clip']['session']==session)
        clip=rec['clip'];vi=clip['cameras'].index(camera);fi=center-clip['source_start'];j=ids.index(k)
        paths=list(Path(clip['video_folder']).glob(f'{session}_cam{camera}_*.avi'));assert len(paths)==1
        cap=cv2.VideoCapture(str(paths[0]));cap.set(cv2.CAP_PROP_POS_FRAMES,fi-2);images=[]
        for _ in range(5):ok,im=cap.read();assert ok;images.append(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
        cap.release();canvas=Image.new('RGB',(1760,820),'#161616');details={}
        for row,seed in enumerate([17,29]):
            with np.load(run/f'seed{seed}_{cohort}_{session}.npz') as z:uv=z['uv'][vi].copy()
            details[str(seed)]=dict(center_error_source=float(np.linalg.norm(uv[fi,j]-gt[lookup[(session,camera,center)],j])),
                frames=list(range(center-2,center+3)),keypoint_source_coordinates=uv[fi-2:fi+3,j].tolist())
            for col,frame in enumerate(range(center-2,center+3)):
                im=Image.fromarray(images[col]).resize((352,384),Image.Resampling.LANCZOS);d=ImageDraw.Draw(im)
                for jj,kk in enumerate(ids):
                    if kk not in set([k,15,16,17,18]):continue
                    x,y=(uv[fi-2+col,jj]*scale+A[:2,2])/2
                    d.ellipse((x-4,y-4,x+4,y+4),outline=COLORS[kk],width=2)
                    d.text((min(max(x+6,0),324),min(max(y-15,0),360)),str(kk),font=font,fill=COLORS[kk],stroke_width=1,stroke_fill='black')
                if frame==center:
                    x,y=(gt[lookup[(session,camera,center)],j]*scale+A[:2,2])/2
                    d.rectangle((x-6,y-6,x+6,y+6),outline='white',width=2)
                canvas.paste(im,(352*col,410*row+26))
                ImageDraw.Draw(canvas).text((352*col+4,410*row+3),f'S{seed} {session} {camera} f{frame} id{k}',font=font,fill='white')
        file=f'focused_{session}_cam{camera}_f{center}_id{k}.jpg';canvas.save(dest/file,quality=95)
        focused.append(dict(session=session,camera=camera,frame=center,id=k,image=file,details=details))
    parities=[dict(seed=r['seed'],session=r['clip']['session'],**p) for r in records for p in r['manual_frame_parity']]
    parity=dict(image_checks=len(parities),images_all_exact=all(p['pixels_exact'] for p in parities),
        prediction_frame_records_exact=sum(p['all_channel_max_source_px']==0 for p in parities),
        nonexact=[p for p in parities if p['all_channel_max_source_px']>0],
        interpretation='166/168 seed-image records match exactly;2 seed29 records differ by one4-input-pixel heatmap cell (~6.25 original px). Input pixels are identical. Batch/numerical argmax sensitivity is plausible but not yet proven; do not claim perfect prediction parity.')
    report=dict(cases=focused,parity=parity,selection='Targeted known errors, NOT representative/random validation.',
        warning='Only center white square is manual ground truth; neighbors have no dense GT. Adjacent stability is not evidence of accuracy.',
        next='Record visual review, then bounded read-only audit of two batch-sensitive peak cases before interpreting sub-cell temporal changes. No new training or pseudoGT.')
    (dest/'focused_cases.json').write_text(json.dumps(report,indent=2))
    page='<meta charset="utf-8"><h1>相邻五帧：短时稳定与跳点</h1><p>每格为相邻1/52秒，只有中央帧白方框是该关键点人工标注。上seed17、下seed29；彩圈未做平滑或纠正。</p>'
    page+=''.join(f'<h2>{r["image"]}</h2><img style="max-width:100%" src="{r["image"]}">' for r in focused)
    (dest/'focused.html').write_text(page+'<pre>'+json.dumps(report,indent=2)+'</pre>')
    export=Path('/mnt/c/Users/55395/Downloads/MouseGeoForge-BottomAnchor-20261002')/dest.name
    for file in [r['image'] for r in focused]+['focused_cases.json','focused.html']:shutil.copy2(dest/file,export/file)
    write_state(latest_decision=str(dest/'focused_cases.json'),next_action=report['next'])
    print(json.dumps(parity,indent=2))


if __name__=='__main__':main()
