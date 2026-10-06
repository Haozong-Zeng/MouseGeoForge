"""Point-only review for spatial expectation vs pooled-memory controls."""
import collections,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import numpy as np
import torch
from PIL import Image,ImageDraw
CYCLE=ROOT/'output/surface_curve_cycle_20261003'
BASE=CYCLE/'integral_back_point_probe_v1';SOURCE=CYCLE/'back_point_probe_inputs_v1'


def main():
    assert (BASE/'results.json').exists();torch.set_num_threads(2)
    inputs={k:torch.load(SOURCE/f'{k}_inputs.pt',map_location='cpu',weights_only=False) for k in ['synthetic','real']}
    with np.load(BASE/'predictions.npz') as z:pred={k:z[k] for k in z.files}
    reference=json.loads((CYCLE/'real_curve_transfer_v1/results.json').read_text())['visuals']
    real_selected={(r['session'],r['frame'],r['view']) for r in reference}
    bysource=[]
    for kind,saved in inputs.items():
        data=saved['tensors'];meta=saved['metadata'];out=BASE/(kind+'_visual_review');out.mkdir(exist_ok=False)
        groups=collections.defaultdict(list)
        for i,m in enumerate(meta):groups[(m.get('session',m['dataset']),m['role'])].append(i)
        for key,arr in pred.items():
            if not key.endswith('__'+kind):continue
            for (source,role),ids in groups.items():
                delta=(arr[ids]-data['gt_xy'][ids].numpy())*data['pixels'][ids,None].numpy();valid=data['valid'][ids].numpy();errors=np.linalg.norm(delta,axis=-1)[valid]
                if len(errors):bysource.append(dict(kind=kind,arm=key.split('__')[0],source=source,role=role,n_points=len(errors),mean_px=float(errors.mean()),median_px=float(np.median(errors))))
        if kind=='real':selected=[i for i,m in enumerate(meta) if (m['session'],m['frame'],m['view']) in real_selected]
        else:
            selected=[ids[len(ids)//2] for key,ids in sorted(groups.items())]
            selected += [i for i,m in enumerate(meta) if m['kind']=='curve' and m['view']==1]
            selected=list(dict.fromkeys(selected))
        rows=[];arms=['coarse','global_strict_seed17','spatial_strict_seed17','global_weak_pair_seed17','spatial_weak_pair_seed17']
        for i in selected:
            m=meta[i]
            if kind=='real':source=ROOT/m['scene']/'blended_images'/f'{m["view"]:08d}.png'
            else:
                matches=list((ROOT/m['scene']/'blended_images').glob(f'{m["view"]:08d}.*'));assert len(matches)==1;source=matches[0]
            width,height=data['pixels'][i].tolist();ph=round(height*320/width)
            with Image.open(source) as image:im=image.convert('RGB').resize((320,ph))
            canvas=Image.new('RGB',(320*len(arms),ph+30),(22,22,28));draw=ImageDraw.Draw(canvas)
            for col,arm in enumerate(arms):
                canvas.paste(im,(320*col,30));draw.text((320*col+4,5),arm,fill='white')
                xy=data['coarse_xy'][i].numpy() if arm=='coarse' else pred[arm+'__'+kind][i]
                for k in range(2):
                    x,y=xy[k]*[320,ph]+[320*col,30];draw.ellipse((x-3,y-3,x+3,y+3),outline='#39dfff',width=2);draw.text((x+4,y+1),str(k+6),fill='#39dfff')
                    if data['valid'][i,k]:
                        gx,gy=data['gt_xy'][i,k].numpy()*[320,ph]+[320*col,30];draw.ellipse((gx-2,gy-2,gx+2,gy+2),fill='#ffe454');draw.text((gx+3,gy-12),str(k+6),fill='#ffe454')
            name=f'{i:03d}.jpg';canvas.save(out/name,quality=95);rows.append(dict(file=name,observation=i,**m))
        html=['<!doctype html><meta charset="utf-8"><title>Back point uncertainty controls</title><style>body{background:#202027;color:white;font:16px sans-serif}img{max-width:100%}</style><h1>06/07 spatial expectation comparison</h1><p>Yellow=existing observed targets (synthetic exact; real manual uncertain). Cyan=experimental point predictions. No fullcurve is drawn. Coarse is frozen originalcoarse output, NOT the full local-refinedCore23head. Allfourmodes have same data/init/updatebudget. Global=pooledmemorycontrol, spatial=fullspatialmap. No GT crops. Soft expectation may average multiple peaks. Commonproxy-normal metrics are not independenttruth. Seed17shown,all3seeds reported.</p>']
        for r in rows:html.append(f'<h2>{r["role"]}: {r["scene"]}, view{r["view"]}</h2><img src="{r["file"]}">')
        (out/'index.html').write_text('\n'.join(html));(out/'manifest.json').write_text(json.dumps(rows,indent=2))
    (BASE/'metrics_by_source.json').write_text(json.dumps(bysource,indent=2))
    results=json.loads((BASE/'results.json').read_text())['results'];pairs=[]
    for seed in [17,29,43]:
        for loss in ['strict','weak_pair']:
            control=next(r for r in results if r['seed']==seed and r['mode']=='global_'+loss)
            actual=next(r for r in results if r['seed']==seed and r['mode']=='spatial_'+loss)
            pairs.append(dict(seed=seed,loss=loss,
                synthetic_val_point_delta_spatial_minus_global=actual['synthetic_validation']['point_error_px']['mean']-control['synthetic_validation']['point_error_px']['mean'],
                real_val_point_delta_spatial_minus_global=actual['real_validation']['point_error_px']['mean']-control['real_validation']['point_error_px']['mean'],
                real_val_proxy_normal_delta_spatial_minus_global=actual['real_validation']['common_proxy_normal_px']['mean']-control['real_validation']['common_proxy_normal_px']['mean']))
    report=dict(pairs=pairs,no_production_promotion=True,limitations='New2pointreadout,manualpointGTuncertain; lowerproxy-normalerrornotindependent evidence oftrueanatomy. Fixeddevelopmentvalidation,notexternaltest.')
    (BASE/'paired_summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
