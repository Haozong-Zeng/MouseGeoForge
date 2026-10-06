"""Human review of GT-oracle recoverable/unrecoverable candidate cases."""
import json,shutil
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from run_teacher_cycle_dlc_20261001 import ROOT,OUT

def main():
    dest=OUT/'paw_heatmap_peaks_v1/review';dest.mkdir(exist_ok=True)
    m=json.loads((OUT/'dlc_dataset_manifest.json').read_text());ps=[]
    for seed in [17,31,73]:
        with np.load(OUT/f'paw_heatmap_peaks_v1/seed{seed}.npz') as z:ps.append(dict(z))
    ind=ps[0]['indices'];gt=ps[0]['gt'];ids=ps[0]['ids'];js=[m['ids_1based'].index(k) for k in ids]
    with np.load(OUT/'lp_ensemble_17_31_73/predictions_source_pixels.npz') as z:ensemble=z['uv_median'][ind][:,js]
    errors=np.linalg.norm(ensemble-gt,axis=-1)
    peaks=np.stack([p['peaks_source_px'] for p in ps]);ratios=np.stack([p['relative_scores'] for p in ps])
    hit=((np.linalg.norm(peaks-gt[None,:,:,None],axis=-1)<=15)&(ratios>=.1)).any(0).any(-1)
    severe=np.argwhere(np.isfinite(errors)&(errors>50));cases=[]
    for recoverable in [True,False]:
        subset=[(int(i),int(j)) for i,j in severe if bool(hit[i,j])==recoverable]
        subset.sort(key=lambda ij:-errors[ij])
        cases.extend([(i,j,recoverable) for i,j in subset[:3]])
    font=ImageFont.truetype('/mnt/c/Windows/Fonts/arial.ttf',18);exports=[]
    for rank,(i,j,recoverable) in enumerate(cases,1):
        entry=m['entries'][int(ind[i])];source=Image.open(ROOT/entry['source']).convert('RGB');canvas=Image.new('RGB',(1350,550),'#111111');draw=ImageDraw.Draw(canvas)
        for si,seed in enumerate([17,31,73]):
            im=source.resize((450,500));d=ImageDraw.Draw(im)
            def xy(p):return p*np.array([450/1080,500/1200])
            a,b=xy(gt[i,j]);d.ellipse((a-8,b-8,a+8,b+8),outline='lime',width=3)
            for k in range(5):
                ratio=ratios[si,i,j,k]
                if ratio<.1:continue
                a,b=xy(peaks[si,i,j,k]);color='red' if k==0 else 'yellow'
                d.line((a-5,b-5,a+5,b+5),fill=color,width=2);d.line((a-5,b+5,a+5,b-5),fill=color,width=2)
                d.text((a+6,b+6),f'{k+1}:{ratio:.2f}',font=font,fill=color,stroke_width=1,stroke_fill='black')
            canvas.paste(im,(si*450,50));draw.text((si*450+5,5),f'seed{seed} k{ids[j]} | green=manual',font=font,fill='white')
            draw.text((si*450+5,27),'red=top1 yellow=alternative (ratio>=.1)',font=font,fill='yellow')
        name=f'{rank:02d}_{entry["session"]}_f{entry["frame"]}_cam{entry["camera"]}_k{ids[j]}_oracle{recoverable}.jpg'
        canvas.save(dest/name,quality=93);exports.append(dict(file=name,source=entry['source'],ensemble_error_px=float(errors[i,j]),oracle_recoverable=recoverable))
    (dest/'cases.json').write_text(json.dumps(exports,indent=2));(dest/'README.txt').write_text('Diagnostic examples deliberately selected using manual labels: three largest-error oracle-recoverable and three unrecoverable cases. Green manual; red highest heatmap local peak; yellow alternative peaks with relative strength>=.1. Labels NEVER changed. Candidates NOT predictions/GT.\n')
    export=Path('/mnt/c/Users/55395/Downloads/MouseGeoForge-TeacherTemporal-20261001/paw_heatmap_peaks_v1')
    shutil.copytree(dest,export,dirs_exist_ok=True);print(json.dumps(exports,indent=2))

if __name__=='__main__':main()
