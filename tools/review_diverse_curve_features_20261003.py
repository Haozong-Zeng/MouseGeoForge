"""Post-run per-source metrics and frozen-query curve visualizations."""
import json,collections,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from PIL import Image,ImageDraw
from mousegeoforge.surface_curve_supervision import points_to_polyline
BASE=ROOT/'output/surface_curve_cycle_20261003/diverse_curve_features_v1'
INPUTS=BASE/'frozen_inputs.pt'
REFERENCE_PREDICTIONS={}
DISPLAY_ARMS=None
PREDICTION_SUFFIX=None
CACHE=ROOT/'output/surface_curve_cycle_20261003/training_v1/pilot/cache'
def main():
    torch.set_num_threads(2);assert (BASE/'results.json').exists()
    saved=torch.load(INPUTS,map_location='cpu',weights_only=False);data=saved['tensors'];meta=saved['metadata']
    with np.load(BASE/'predictions.npz') as z:
        pred={k:z[k] for k in z.files} if PREDICTION_SUFFIX is None else {k[:-len(PREDICTION_SUFFIX)]:z[k] for k in z.files if k.endswith(PREDICTION_SUFFIX)}
    pred.update(REFERENCE_PREDICTIONS);arms=list(pred);columns=DISPLAY_ARMS or arms
    out=BASE/'visual_review';out.mkdir(exist_ok=False);groups=collections.defaultdict(list);stats=collections.defaultdict(list)
    for i,m in enumerate(meta):
        groups[(m['dataset'],m['role'])].append(i)
        for arm in arms:
            p=torch.tensor(pred[arm][i])*data['pixels'][i];g=data['gt'][i]*data['pixels'][i]
            d=points_to_polyline(p,g,torch.ones(65,dtype=torch.bool))[0].sqrt().mean()
            stats[(arm,m['dataset'],m['role'])].append(float(d))
    selected=[ids[len(ids)//2] for _,ids in sorted(groups.items())]
    # Include all three new poses in a common side camera as well.
    selected+= [i for i,m in enumerate(meta) if m['kind']=='curve' and m['view']==1]
    selected=list(dict.fromkeys(selected));rows=[]
    for i in selected:
        m=meta[i]
        if 'source_image' in m:
            h,w=m['image_hw'];ph=round(h*352/w)
            with Image.open(ROOT/m['source_image']) as source:im=source.convert('RGB').resize((352,ph))
        else:
            item=torch.load(CACHE/f"{m['key']}.pt",map_location='cpu',weights_only=False);v=item['record']['views'].index(m['view'])
            rgb=item['batch']['images'][0,v].permute(1,2,0).numpy();h,w=rgb.shape[:2];ph=round(h*352/w)
            im=Image.fromarray(np.uint8(rgb.clip(0,1)*255)).resize((352,ph))
        canvas=Image.new('RGB',(352*(len(columns)+1),ph+28),(24,24,30));draw=ImageDraw.Draw(canvas)
        for col,arm in enumerate(['GT']+columns):
            canvas.paste(im,(col*352,28));draw.text((col*352+5,5),arm,fill='white')
            gt=data['gt'][i].numpy()*[352,ph]+[col*352,28];draw.line([tuple(p) for p in gt],fill='#ffe454',width=2)
            if arm!='GT':
                line=pred[arm][i]*[352,ph]+[col*352,28];draw.line([tuple(p) for p in line],fill='#39dfff',width=2)
        name=f'{i:03d}.jpg';canvas.save(out/name,quality=96);rows.append(dict(file=name,index=i,**m))
    summary=[dict(arm=a,dataset=d,role=r,n=len(values),mean=float(np.mean(values)),median=float(np.median(values)),p90=float(np.quantile(values,.9))) for (a,d,r),values in stats.items()]
    (out/'metrics_by_source.json').write_text(json.dumps(summary,indent=2));(out/'manifest.json').write_text(json.dumps(dict(rows=rows,inspection_pending=True),indent=2))
    html=['<!doctype html><meta charset="utf-8"><title>Diverse curve feature comparison</title><style>body{background:#202027;color:#eee;font:16px sans-serif}img{max-width:100%}</style><h1>Frozen-model curve feature comparison</h1><p>Yellow=known synthetic material curve (amodal, includes occlusion). Cyan=prediction. Equal2400updates. Existing point/depth/detection heads frozen; instances associated by predicted boxes/objectness. Train and validation roles listed. Same assets: not an external real benchmark.</p>']
    for r in rows:html.append(f'<h2>{r["role"]}: {r["scene"]}, view{r["view"]}, instance{r["instance"]}</h2><img src="{r["file"]}">')
    (out/'index.html').write_text('\n'.join(html));print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
