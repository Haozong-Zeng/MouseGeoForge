"""CPU-only diagnostic: expectation, maximum cell, spread; not confidence GT."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image,ImageDraw
from mousegeoforge.models.back_point_integral import BackPointIntegralProbe
CYCLE=ROOT/'output/surface_curve_cycle_20261003';BASE=CYCLE/'real_point_heatmap_v1'


def main():
    assert (BASE/'results.json').exists();torch.set_num_threads(2)
    out=BASE/'heatmap_review';out.mkdir(exist_ok=False)
    saved=torch.load(CYCLE/'back_point_probe_inputs_v1/real_inputs.pt',map_location='cpu',weights_only=False);data=saved['tensors'];meta=saved['metadata']
    manifest=json.loads((CYCLE/'real_point_spatial_memory_v1/manifest.json').read_text());bindings={}
    for record in manifest['records']:
        for b in record['bindings']:bindings[b['observation']]=(record['path'],b['view_index'])
    reference=json.loads((BASE/'real_visual_review/manifest.json').read_text())
    selected=[r['observation'] for r in reference if r['role']=='validation' or (r['session']=='AZL3' and r['view'] in (1,2,5))]
    maps=[]
    for i in selected:
        path,v=bindings[i];m=torch.load(ROOT/path,map_location='cpu',weights_only=False)['memory'][v:v+1].float()
        maps.append(F.interpolate(m,(48,48),mode='bilinear',align_corners=False)[0])
    memory=torch.stack(maps);arms=['coordinate_seed17','heatmap_seed17'];outputs={};rows=[]
    yy,xx=torch.meshgrid((torch.arange(48)+.5)/48,(torch.arange(48)+.5)/48,indexing='ij');grid=torch.stack([xx,yy],-1).numpy()
    for arm in arms:
        model=BackPointIntegralProbe('spatial').eval();model.load_state_dict(torch.load(BASE/f'{arm}_latest.pt',map_location='cpu',weights_only=False)['state'])
        with torch.no_grad():xy,p=model(data['pose'][selected],data['coarse_xy'][selected],memory,True)
        outputs[arm]=(xy.numpy(),p.numpy())
    panels=[]
    for j,i in enumerate(selected):
        m=meta[i];source=ROOT/m['scene']/'blended_images'/f'{m["view"]:08d}.png'
        with Image.open(source) as image:im=image.convert('RGB').resize((320,349))
        canvas=Image.new('RGB',(640,395),(20,20,25));draw=ImageDraw.Draw(canvas)
        for col,arm in enumerate(arms):
            xy,prob=outputs[arm];p=prob[j];heat=np.stack([p[1]/max(p[1].max(),1e-9),p[0]/max(p[0].max(),1e-9),np.zeros((48,48))],-1)
            heat_im=Image.fromarray(np.uint8(255*heat)).resize(im.size,Image.Resampling.BILINEAR)
            overlay=Image.blend(im,heat_im,.3);canvas.paste(overlay,(320*col,46));draw.text((320*col+3,3),arm,fill='white')
            entries=[]
            for k in range(2):
                peak_y,peak_x=np.unravel_index(p[k].argmax(),p[k].shape);peak=grid[peak_y,peak_x];pixel=data['pixels'][i].numpy()
                distance=float(np.linalg.norm((peak-xy[j,k])*pixel));spread=float(np.sqrt((p[k]*(((grid-xy[j,k])*pixel)**2).sum(-1)).sum()))
                entropy=float(-(p[k]*np.log(p[k]+1e-12)).sum()/np.log(48*48))
                entries.append(dict(point=k+6,peak_expectation_distance_px=distance,rms_spread_px=spread,normalized_entropy=entropy,observed=bool(data['valid'][i,k])))
                x,y=xy[j,k]*[320,349]+[320*col,46];draw.ellipse((x-4,y-4,x+4,y+4),outline='#39dfff',width=2)
                px,py=peak*[320,349]+[320*col,46];draw.line((px-4,py,px+4,py),fill='#ff55ff',width=2);draw.line((px,py-4,px,py+4),fill='#ff55ff',width=2)
                if data['valid'][i,k]:
                    gx,gy=data['gt_xy'][i,k].numpy()*[320,349]+[320*col,46];draw.ellipse((gx-2,gy-2,gx+2,gy+2),fill='#ffe454')
            draw.text((320*col+3,22),'peak-mean px: '+','.join(f'{r["peak_expectation_distance_px"]:.1f}' for r in entries),fill='white')
            rows.append(dict(observation=i,arm=arm,session=m['session'],frame=m['frame'],view=m['view'],role=m['role'],points=entries))
        name=f'{i:03d}.jpg';canvas.save(out/name,quality=95);panels.append(dict(file=name,**m))
    html=['<!doctype html><meta charset="utf-8"><title>Point probability diagnostics</title><style>body{background:#202027;color:white;font:16px sans-serif}img{max-width:100%}</style><h1>Spatial expectation diagnostics</h1><p>Cyan circles=expectation predictions; magenta crosses=maximum probability cells; yellow=manual observations. Green=06 heat, red=07 heat, each independently max-normalized for display (NOT calibratedconfidence). Broad or multiplepeaks can put expectation between structures. Onlyheatmaparm usedGaussianKL supervision(sigma12pxchosenkernel, notcalibrateduncertainty); coordinatearm got equalextraoptimization. Fixedseed17qualitativeinspection, notbestseedselection.</p>']
    for r in panels:html.append(f'<h2>{r["role"]}: {r["session"]} frame{r["frame"]} view{r["view"]}</h2><img src="{r["file"]}">')
    (out/'index.html').write_text('\n'.join(html));(out/'metrics.json').write_text(json.dumps(rows,indent=2));(out/'manifest.json').write_text(json.dumps(panels,indent=2))
    print('Inspected diagnostic panels ready:',len(panels))


if __name__=='__main__':main()
