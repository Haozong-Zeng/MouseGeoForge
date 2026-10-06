"""Fixed-source validation visual review, not a production label export."""
import argparse, collections, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import numpy as np
import torch
from PIL import Image, ImageDraw
from run_depth_pose_ablation import write_json
CYCLE=ROOT/'output/surface_curve_cycle_20261003';OUT=CYCLE/'controlled_back_noise_probe_v1'


def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--curve-aux',action='store_true');args=parser.parse_args()
    if args.curve_aux:OUT=CYCLE/'controlled_back_curve_aux_v1'
    results=json.loads((OUT/'results.json').read_text())['results']
    saved=torch.load(CYCLE/'controlled_back_noise_inputs_v1/inputs.pt',map_location='cpu',weights_only=False)
    d=saved['tensors'];meta=saved['metadata'];pred=np.load(OUT/'predictions.npz')
    modes=(['noisy_strict','noisy_strict_curve','noisy_weak_chord_curve'] if args.curve_aux else
           ['clean_strict_reference','noisy_strict','noisy_weak_chord','noisy_weak_oracle'])
    summary={}
    for seed in (17,29,43):
        selected={r['mode']:r for r in results if r['seed']==seed}
        summary[str(seed)]={m:{k:selected[m]['validation'][k]['mean'] for k in ('clean_anchor_px','point_to_curve_px','nearest_material_u_error')} for m in modes}
        for m in modes[1 if args.curve_aux else 2:]:
            summary[str(seed)][m]['anchor_delta_vs_noisy_strict']=summary[str(seed)][m]['clean_anchor_px']-summary[str(seed)]['noisy_strict']['clean_anchor_px']
    write_json(OUT/'paired_summary.json',summary)
    review=OUT/'visual_review';review.mkdir(exist_ok=True)
    groups=collections.defaultdict(list)
    for i,m in enumerate(meta):
        if m['role']=='validation':groups[m['dataset']].append(i)
    chosen=[]
    for name,ids in sorted(groups.items()):
        chosen.extend(ids[j] for j in sorted({0,len(ids)//2,len(ids)-1}))
    panels=[]
    for i in chosen:
        m=meta[i];paths=list((ROOT/m['scene']/'blended_images').glob(f'{m["view"]:08d}.*'));assert len(paths)==1
        w,h=map(int,d['pixels'][i].tolist())
        with Image.open(paths[0]) as original:base=original.convert('RGB').resize((w,h))
        canvas=Image.new('RGB',(w*2,(h+46)*2),'#17171b');draw=ImageDraw.Draw(canvas)
        for j,mode in enumerate(modes):
            panel=base.copy();pd=ImageDraw.Draw(panel)
            pd.line([tuple(x) for x in d['curve_gt'][i].numpy()*[w,h]],fill='#ffe454',width=2)
            for k,xy in enumerate(d['clean_xy'][i].numpy()*[w,h]):
                x,y=xy;pd.ellipse((x-5,y-5,x+5,y+5),outline='#44ff44',width=2);pd.text((x+6,y-15),'AB'[k],fill='#44ff44')
            xy=pred[f'{mode}_seed17'][i]*[w,h]
            for k,(x,y) in enumerate(xy):
                pd.line((x-5,y,x+5,y),fill='#ff66ee',width=3);pd.line((x,y-5,x,y+5),fill='#ff66ee',width=3);pd.text((x+6,y+3),'AB'[k],fill='#ff66ee')
            col,row=j%2,j//2;canvas.paste(panel,(col*w,row*(h+46)+46))
            errors=np.linalg.norm(xy-d['clean_xy'][i].numpy()*[w,h],axis=-1)
            draw.text((col*w+8,row*(h+46)+8),mode+f' | error A/B {errors[0]:.1f}/{errors[1]:.1f}px',fill='white')
        filename=f'{i:03d}.jpg';canvas.save(review/filename,quality=94)
        panels.append(dict(file=filename,observation=i,**m))
    html=['<!doctype html><meta charset="utf-8"><title>Controlled surface annotation noise</title><style>body{background:#202027;color:white;font:16px sans-serif}img{max-width:100%}</style><h1>Synthetic virtual-anchor noise: fixed validation review</h1><p>Yellow: known surface line. Green A/B: clean material anchors u=.30/.70 (NOT Core23). Magenta: prediction. Seed17 displayed, three seeds in summary. First/middle/last validation observation per dataset, no error-based selection. Oracle uses true tangent in TRAIN loss only and is not deployable. Auxiliary curve modes use known synthetic geometry in TRAIN loss only; cannot apply this to untraced real data. Distance to a curve is not material-point accuracy. No real data or annotation changes.</p>', '<pre>'+json.dumps(summary,indent=2)+'</pre>']
    for p in panels:html.extend([f'<h2>{p["dataset"]}, view {p["view"]}, observation {p["observation"]}</h2>',f'<img src="{p["file"]}">'])
    (review/'index.html').write_text('\n'.join(html),encoding='utf-8');write_json(review/'manifest.json',panels)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
