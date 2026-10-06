"""Shared spatial feature auxiliary point loss review, not real curve targets."""
import json
import numpy as np
import torch
from PIL import Image,ImageDraw
import review_diverse_curve_features_20261003 as review
CYCLE=review.ROOT/'output/surface_curve_cycle_20261003'
BASE=CYCLE/'joint_surface_point_integral_v1'


def crossings(line):
    a=line[:-1];b=line[1:];count=0
    cross=lambda x,y:x[0]*y[1]-x[1]*y[0]
    for i in range(len(a)):
        for j in range(i+2,len(a)):
            if cross(b[i]-a[i],a[j]-a[i])*cross(b[i]-a[i],b[j]-a[i])<0 and cross(b[j]-a[j],a[i]-a[j])*cross(b[j]-a[j],b[i]-a[j])<0:count+=1
    return count


def main():
    review.BASE=BASE;review.INPUTS=CYCLE/'expanded_curve_features_v1/frozen_inputs.pt'
    review.PREDICTION_SUFFIX='__synthetic';review.DISPLAY_ARMS=['replay_seed17','joint_points_seed17'];review.main()
    path=BASE/'visual_review/index.html';html=path.read_text().replace('<h1>Frozen-model curve feature comparison</h1>',
        '<h1>Synthetic retention after weak real adaptation</h1><p>Both arms start with real+synthetic pretrained pointkeys and syntheticcurvequery. Same syntheticbatches; joint_points uses separate realpointheadauxloss throughsharedkey. No realfullcurveGT.</p>');path.write_text(html)
    real=torch.load(CYCLE/'real_curve_adaptation_inputs_v1/frozen_inputs.pt',map_location='cpu',weights_only=False);data=real['tensors'];meta=real['metadata']
    with np.load(BASE/'predictions.npz') as z:pred={k[:-6]:z[k] for k in z.files if k.endswith('__real')}
    original=json.loads((CYCLE/'real_curve_transfer_v1/results.json').read_text())['visuals']
    selected={(r['session'],r['frame'],r['view']) for r in original};out=BASE/'real_visual_review';out.mkdir(exist_ok=False);rows=[]
    for i,m in enumerate(meta):
        if (m['session'],m['frame'],m['view']) not in selected:continue
        source=review.ROOT/m['scene']/'blended_images'/f'{m["view"]:08d}.png'
        with Image.open(source) as im:im=im.convert('RGB').resize((352,384))
        arms=['replay_seed17','joint_points_seed17'];canvas=Image.new('RGB',(704,412),(22,22,28));draw=ImageDraw.Draw(canvas)
        for col,arm in enumerate(arms):
            canvas.paste(im,(352*col,28));draw.text((352*col+5,5),arm,fill='white')
            line=pred[arm][i]*[352,384]+[352*col,28];draw.line([tuple(p) for p in line],fill='#39dfff',width=2)
            for k in range(2):
                if data['observed_valid'][i,k]:
                    x,y=data['observed_xy'][i,k].numpy()*[352,384]+[352*col,28]
                    draw.ellipse((x-3,y-3,x+3,y+3),fill='#ffe454');draw.text((x+4,y-8),str(k+6),fill='#ffe454')
        name=f'{m["session"]}_{m["frame"]:06d}_cam{chr(65+m["view"])}.jpg';canvas.save(out/name,quality=94);rows.append(dict(file=name,observation=i,**m))
    html=['<!doctype html><meta charset="utf-8"><title>Shared point supervision</title><style>body{background:#202027;color:white;font:16px sans-serif}img{max-width:100%}</style><h1>Shared features: separate point auxiliary task</h1><p>Yellow=manual06/07, not a fullcurve or endpoints. Cyan=predictions. Replay andjoint_points sharerealpoint-pretrainedinitialization; joint_points adds separatepointheadloss, not realcurvepointdistance. No pointfractionassignment. ReplayisNOTsynthetic-only. Check role: real TRAIN examples are fitting, validation is development, not external benchmark. No pseudoGT exported.</p>']
    for r in rows:html.append(f'<h2>{r["role"]}: {r["session"]} frame{r["frame"]} view{r["view"]}</h2><img src="{r["file"]}">')
    (out/'index.html').write_text('\n'.join(html));(out/'manifest.json').write_text(json.dumps(rows,indent=2))
    results=json.loads((BASE/'results.json').read_text())['results'];pairs=[]
    for seed in [17,29,43]:
        a=next(r for r in results if r['seed']==seed and r['mode']=='replay');b=next(r for r in results if r['seed']==seed and r['mode']=='joint_points')
        pairs.append(dict(seed=seed,
            synthetic_val_pred_gt_delta=b['synthetic_validation']['pred_to_gt_px']['mean']-a['synthetic_validation']['pred_to_gt_px']['mean'],
            synthetic_val_gt_pred_delta=b['synthetic_validation']['gt_to_pred_px']['mean']-a['synthetic_validation']['gt_to_pred_px']['mean'],
            real_val_weak_distance_delta=b['real_validation']['observed_point_to_curve_px']['mean']-a['real_validation']['observed_point_to_curve_px']['mean'],
            real_val_point_head_error_delta=b['real_validation']['actual_point_head_error_px']['mean']-a['real_validation']['actual_point_head_error_px']['mean'],
            real_val_order_reversal_replay=a['real_validation']['observed_order_reversal']['mean'],
            real_val_order_reversal_weak=b['real_validation']['observed_order_reversal']['mean']))
    shapes=[]
    for arm,line in pred.items():
        for role in ['train','validation']:
            indices=[i for i,m in enumerate(meta) if m['role']==role];values=[crossings(line[i]) for i in indices]
            shapes.append(dict(arm=arm,role=role,n=len(indices),curves_with_self_crossing=sum(v>0 for v in values),total_segment_crossings=sum(values)))
    payload=dict(pairs=pairs,shape_diagnostics=shapes,limitations='Sparsepointdistance is not fullcurve accuracy. Self-crossings are only warnings: projected3Dcurves can overlap. No promotion before visual QA/synthetic-retention checks.')
    (BASE/'paired_summary.json').write_text(json.dumps(payload,indent=2));print(json.dumps(payload,indent=2))


if __name__=='__main__':main()
