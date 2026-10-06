"""Deterministic conditional-readout visual review; images are local artifacts."""
import argparse
import html
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import numpy as np
from PIL import Image,ImageDraw
import torch
from run_depth_pose_ablation import write_json
CYCLE=ROOT/'output/core23_spatial_cycle_20261004'
BASE=CYCLE/'readout_pilot_v1'
OUT=CYCLE/'review_v1'


def main():
    global BASE,OUT
    parser=argparse.ArgumentParser();parser.add_argument('--softbox',action='store_true');parser.add_argument('--expanded',action='store_true');parser.add_argument('--all-real',action='store_true');args=parser.parse_args()
    if args.softbox:
        BASE=CYCLE/'readout_softbox_v1';OUT=CYCLE/'review_softbox_v1'
    if args.expanded:
        assert args.softbox
        BASE=CYCLE/'readout_expanded_synthetic_soft_box_v1';OUT=CYCLE/'review_expanded_synthetic_softbox_v1'
    if args.all_real:
        assert args.expanded and args.softbox
        BASE=CYCLE/'readout_expanded_all_soft_box_v1';OUT=CYCLE/'review_expanded_all_softbox_v1'
    torch.set_num_threads(2)
    manifest=json.loads((CYCLE/'features_v1/manifest.json').read_text())
    if args.expanded:
        expanded=json.loads((CYCLE/('features_expanded_all_v1' if args.all_real else 'features_expanded_synthetic_v1')/'manifest.json').read_text())
        assert expanded['records'][:len(manifest['records'])]==manifest['records']
    rows=[]
    for r in manifest['records']:
        p=torch.load(ROOT/r['path'],map_location='cpu',weights_only=False)
        for o in p['observations']:rows.append(dict(o,image_hw=p['image_hw']))
    with np.load(BASE/'predictions.npz') as a:pred={k:a[k] for k in a.files}
    metrics=[]
    for i,o in enumerate(rows):
        m=o['metadata'];valid=o['valid'].numpy().copy()
        if m['instance_count']>1:valid[7:14]=False
        err=np.linalg.norm((pred['spatial_seed17'][i]-o['gt_xy'].numpy())*o['image_hw'][::-1],axis=-1)
        metrics.append(float(err[valid].mean()) if valid.any() else -1)
    selected=[]
    for kind,count in [('real',1),('synthetic',1),('synthetic',2)]:
        for role in ['train','validation']:
            ids=[i for i,o in enumerate(rows) if ('real' if o['metadata']['record']['kind']=='real' else 'synthetic')==kind and o['metadata']['instance_count']==count and o['metadata']['record']['role']==role]
            ids.sort(key=lambda i:metrics[i])
            if ids:
                selected.extend([(ids[len(ids)//2],'median'),(ids[-1],'worst')])
    if args.expanded:
        previous=[r['observation'] for folder in ['review_v1','review_softbox_v1'] for r in json.loads((CYCLE/folder/'manifest.json').read_text())]
        selected.extend((i,'fixedprevious') for i in dict.fromkeys(previous) if i not in {j for j,_ in selected})
    OUT.mkdir(exist_ok=False)
    reports=[]
    for i,selection in selected:
        o=rows[i];m=o['metadata'];r=m['record']
        cache=torch.load(ROOT/f'output/surface_curve_cycle_20261003/training_v1/pilot/cache/{r["key"]}.pt',map_location='cpu',weights_only=False)
        pixels=(cache['batch']['images'][0,m['view_index']].permute(1,2,0).numpy()*255).clip(0,255).astype('uint8')
        h,w=pixels.shape[:2];scale=min(550/h,500/w);ph,pw=round(h*scale),round(w*scale)
        raw=Image.fromarray(pixels).resize((pw,ph));canvas=Image.new('RGB',(pw*3,ph+65),(22,24,30));draw=ImageDraw.Draw(canvas)
        valid=o['valid'].numpy().copy()
        if m['instance_count']>1:valid[7:14]=False
        # Focus on nose, back and feet for legibility; full-point metrics stay separate.
        shown=[j for j in [0,5,6,14,15,16,17] if valid[j]]
        for col,arm in enumerate(['original_refined','global_seed17','spatial_seed17']):
            canvas.paste(raw,(col*pw,65));draw.text((col*pw+4,5),arm,fill='white')
            draw.text((col*pw+4,22),'yellow cross=label; cyan circle=prediction',fill='white')
            xy=o['original_xy'].numpy() if col==0 else pred[arm][i]
            for j in shown:
                x,y=xy[j]*[pw,ph]+[col*pw,65];gx,gy=o['gt_xy'][j].numpy()*[pw,ph]+[col*pw,65]
                draw.line((gx,gy,x,y),fill='#888888',width=1)
                draw.ellipse((x-3,y-3,x+3,y+3),outline='#44ddff',width=2);draw.text((x+4,y+1),str(j+1),fill='#44ddff')
                draw.line((gx-4,gy,gx+4,gy),fill='#ffe45c',width=2);draw.line((gx,gy-4,gx,gy+4),fill='#ffe45c',width=2);draw.text((gx+4,gy-12),str(j+1),fill='#ffe45c')
        name=f'{i:03d}_{selection}.jpg';canvas.save(OUT/name,quality=95)
        reports.append(dict(file=name,observation=i,selection=selection,mean_supported_error=metrics[i],metadata=m))
    write_json(OUT/'manifest.json',reports)
    baseline=json.loads((BASE/'baseline.json').read_text());results=json.loads((BASE/'results.json').read_text())
    table=[]
    for name,metric in [(k,v) for k,v in baseline.items()]+[(r['arm'],r['metrics']) for r in results['results']]:
        table.append('<tr><td>'+name+'</td>'+''.join('<td>%.2f</td>'%metric[s][k]['mean'] for s,k in [('real_validation','reliable'),('real_validation','back06_07'),('synthetic_validation','single_instance'),('synthetic_validation','multi_instance')])+'</tr>')
    page=['<!doctype html><meta charset="utf-8"><title>Core23 spatial readout pilot</title><style>body{background:#20232b;color:#eee;font:16px sans-serif;margin:28px}img{max-width:100%}td,th{padding:8px;border:1px solid #666}table{border-collapse:collapse}</style>',
          '<h1>Core23 spatial readout: conditional pose review</h1><p>768-long-side pixels. GT box/objectness-assisted query assignment; NOT end-to-end detection. Real labels support only IDs1,6,7,15-18. Multi-mouse tails are off in every arm. 3 matched seeds, 1600 updates. Existing original head did not receive the same real-label adaptation; architecture claim relies on global/spatial paired controls.</p>',
          '<p>Prediction-only soft expanded-box prior: '+str(args.softbox)+'. Compare with the ungated pilot separately. No production promotion. Images show nose/back/feet only for readability; all supported-point errors are reported in the table. Median and worst are selected by spatial seed17 supported error within each stratum, not handpicked favorable examples.</p>',
          '<p>Expanded synthetic TRAIN: '+str(args.expanded)+'. Visuals show original393-observation prefix; added TRAIN poses are not shown here. Fixedprevious cases reuse earlier selected frames regardless of this model error. Validation is unchanged; metrics include entire active train set.</p>',
          '<p>Expanded real TRAIN: '+str(args.all_real)+'. Real training observations188 to1043; real validation48 unchanged.</p>',
          '<table><tr><th>Arm</th><th>Real reliable</th><th>Real back</th><th>Synthetic single</th><th>Synthetic dual</th></tr>'+''.join(table)+'</table>']
    for r in reports:page.append('<h2>'+html.escape(str(r['metadata']['record'])+' '+r['selection']+' view='+str(r['metadata']['view'])+' instance='+str(r['metadata']['instance']))+'</h2><img src="'+r['file']+'">')
    (OUT/'index.html').write_text('\n'.join(page),encoding='utf-8')
    dest=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-Core23-Spatial-20261004')/OUT.name
    assert not dest.exists();shutil.copytree(OUT,dest)
    print(json.dumps(reports,indent=2))


if __name__=='__main__':main()
