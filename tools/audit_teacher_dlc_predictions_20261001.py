"""Independent, original-pixel teacher error and confidence/coverage diagnostics."""
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/teacher_temporal_cycle_20261001'

def stats(v):
    a=np.asarray(v,float);a=a[np.isfinite(a)]
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),p90=float(np.percentile(a,90))) if len(a) else dict(n=0)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prediction-pattern',default='*snapshot_best-30.h5')
    parser.add_argument('--review-name',default='dlc_baseline_review')
    args=parser.parse_args()
    assert '/' not in args.review_name and '\\' not in args.review_name
    m=json.loads((OUT/'dlc_dataset_manifest.json').read_text());project=Path(m['project_config']).parent
    matches=list((project/'evaluation-results-pytorch').rglob(args.prediction_pattern))
    assert len(matches)==1,matches
    p=matches[0]
    pred=pd.read_hdf(p);truth=pd.read_hdf(project/'labeled-data/camA/CollectedData_manualv3.h5')
    pred=pred.reindex(truth.index)
    uv=np.stack([pred.xs(n,level='bodyparts',axis=1).droplevel(['scorer','individuals'],axis=1)[['x','y']].values for n in m['names']],axis=1)
    conf=np.stack([pred.xs(n,level='bodyparts',axis=1).xs('likelihood',level='coords',axis=1).values[:,0] for n in m['names']],axis=1)
    gt=truth.values.reshape(-1,7,2);A=np.array(m['affine_source_to_input'])
    uv=(uv-A[:2,2])/np.diag(A)[:2];gt=(gt-A[:2,2])/np.diag(A)[:2]
    err=np.linalg.norm(uv-gt,axis=-1);records=[];image_scores=[]
    for i,e in enumerate(m['entries']):
        for j,k in enumerate(m['ids_1based']):
            if not np.isfinite(err[i,j]):continue
            records.append(dict(role=e['role'],session=e['session'],frame=e['frame'],camera=e['camera'],id=k,
                                error_px=float(err[i,j]),confidence=float(conf[i,j])))
        if e['role']=='validation' and np.isfinite(err[i]).any():image_scores.append((float(np.nanmean(err[i])),i))
    result={'units':'Original1080x1200 pixels; no confidence filtering unless explicitly stated','checkpoint':p.name}
    for role in ['train','validation']:
        rs=[r for r in records if r['role']==role]
        result[role]=dict(all=stats([r['error_px'] for r in rs]),
            keypoints={str(k):stats([r['error_px'] for r in rs if r['id']==k]) for k in m['ids_1based']},
            cameras={c:stats([r['error_px'] for r in rs if r['camera']==c]) for c in sorted({r['camera'] for r in rs})},
            confidence_curve={str(t):dict(coverage=sum(r['confidence']>=t for r in rs)/len(rs),
                errors=stats([r['error_px'] for r in rs if r['confidence']>=t])) for t in [0,.1,.3,.6,.8,.9]})
    folder=OUT/args.review_name;folder.mkdir(exist_ok=False)
    chosen=[i for _,i in sorted(image_scores)[-4:]]+[i for _,i in sorted(image_scores)[len(image_scores)//2:len(image_scores)//2+4]]
    font=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',21);images=[]
    for i in chosen:
        e=m['entries'][i];im=Image.open(ROOT/e['source']).convert('RGB');d=ImageDraw.Draw(im)
        for j,k in enumerate(m['ids_1based']):
            if not np.isfinite(gt[i,j]).all():continue
            x,y=gt[i,j];u,v=uv[i,j]
            d.ellipse((x-6,y-6,x+6,y+6),outline='#76ff6b',width=3)
            d.text((x+7,y-20),str(k),fill='#76ff6b',font=font,stroke_width=1,stroke_fill='black')
            d.line((x,y,u,v),fill='#ff75d5',width=2);d.line((u-6,v-6,u+6,v+6),fill='#ff75d5',width=3);d.line((u-6,v+6,u+6,v-6),fill='#ff75d5',width=3)
            d.text((u+9,v+3),f'{k}:{conf[i,j]:.2f}',fill='#ff75d5',font=font,stroke_width=1,stroke_fill='black')
        dest=folder/f"{e['session']}_{e['frame']:06d}_cam{e['camera']}.jpg";im.save(dest,quality=95);images.append(dest.name)
    (folder/'summary.json').write_text(json.dumps(result,indent=2));(folder/'observations.json').write_text(json.dumps(records,indent=2))
    np.savez_compressed(folder/'predictions_source_pixels.npz',uv=uv,confidence=conf,gt=gt,error_px=err,ids=m['ids_1based'])
    (folder/'index.html').write_text(f'<!doctype html><meta charset="utf-8"><h1>{p.stem} 验证检查</h1><p>绿圈人工，粉叉预测；粉色数字为ID与模型置信度，不是校准概率。原图1080×1200像素。</p><p>前4图为验证均值误差最高，后4图为中位附近病例。</p>'+''.join(f'<h2>{p}</h2><img style="max-width:100%" src="{p}">' for p in images))
    print(json.dumps(result['validation'],indent=2))

if __name__=='__main__':main()
