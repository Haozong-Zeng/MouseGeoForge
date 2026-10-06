"""TRAIN-only sanity probe images; not a deployment or validation gallery."""
import html,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from PIL import Image,ImageDraw
from mousegeoforge.detection_readout_loss import match_detection
from run_depth_pose_ablation import fingerprint,write_json
from run_surface_curve_pilot_20261003 import load_item
CYCLE=ROOT/'output/core23_spatial_cycle_20261004';BASE=CYCLE/'bottom_train_fit_v1';OUT=CYCLE/'review_bottom_train_fit_v1'

def main():
    assert (BASE/'results.json').exists() and not OUT.exists();OUT.mkdir();torch.set_num_threads(2);rows=[]
    for r in json.loads((CYCLE/'detection_features_v1/manifest.json').read_text())['records']:
        assert fingerprint(ROOT/r['path'])==r['sha256'];rows.extend(torch.load(ROOT/r['path'],map_location='cpu',weights_only=False)['observations'])
    a=np.load(BASE/'predictions.npz');ids=a['image_indices'].tolist();det=np.load(CYCLE/'detection_context_roi_v1/predictions.npz');result=json.loads((BASE/'results.json').read_text())
    page=['<!doctype html><meta charset="utf-8"><title>TRAIN-only fit sanity</title><style>body{background:#20232b;color:white;font:16px sans-serif}img{max-width:100%}</style><h1>TRAIN-only fitting sanity - NOT validation</h1><p>14training images, no validation scores. Left initial; middle frozen decoder; right unfrozen decoder after1600steps on these same images. GT box-assisted matched query shown for diagnostic clarity; NOT raw detection output. All138queries decoded before matching. Yellow crosses=labels, cyan circles=predictions IDs1/15-18. No label changes.</p><pre>'+html.escape(json.dumps(result,indent=2))+'</pre>'];meta=[]
    for j,i in enumerate(ids):
        o=rows[i];m=o['metadata'];r=m['record'];assert r['role']=='train' and r['session'] not in ['GR2','UR2','UR3']
        p={k:torch.tensor(det['seed17__'+k][i:i+1]) for k in ['instance_boxes_cxcywh','instance_objectness_logits']};qs,ms=match_detection(p,o['boxes'][None],(o['box_valid']&o['instance_valid'])[None])[0];q=int(qs[0]);inst=int(ms[0])
        cache=ROOT/f'output/surface_curve_cycle_20261003/training_v1/pilot/cache/{r["key"]}.pt';item=torch.load(cache,map_location='cpu',weights_only=False) if cache.exists() else load_item(r)
        rgb=(item['batch']['images'][0,m['view_index']].permute(1,2,0).numpy()*255).clip(0,255).astype('uint8');h,w=rgb.shape[:2];pw=360;ph=round(h*pw/w);im=Image.fromarray(rgb).resize((pw,ph));canvas=Image.new('RGB',(3*pw,ph+30),(22,24,30));draw=ImageDraw.Draw(canvas)
        for col,name in enumerate(['initial','frozen','unfrozen']):
            canvas.paste(im,(col*pw,30));draw.text((col*pw+5,5),name+' TRAIN only',fill='white')
            for k in [0,14,15,16,17]:
                x,y=a[name][j,q,k]*[pw,ph]+[col*pw,30];gx,gy=o['gt_xy'][inst,k].numpy()*[pw,ph]+[col*pw,30];draw.ellipse((x-3,y-3,x+3,y+3),outline='#55ddff',width=2);draw.text((x+3,y),str(k+1),fill='#55ddff');draw.line((gx-3,gy,gx+3,gy),fill='#ffee55',width=2);draw.line((gx,gy-3,gx,gy+3),fill='#ffee55',width=2)
        name=f'{j:02d}_{r["session"]}.jpg';canvas.save(OUT/name,quality=95);meta.append(dict(file=name,observation=i,metadata=m,query=q));page.append('<h2>'+html.escape(r['session']+' frame'+str(r['frame']))+'</h2><img src="'+name+'">')
    write_json(OUT/'manifest.json',meta);write_json(OUT/'results.json',result);(OUT/'index.html').write_text('\n'.join(page),encoding='utf-8');dest=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-Core23-Spatial-20261004')/OUT.name;assert not dest.exists();shutil.copytree(OUT,dest)

if __name__=='__main__':main()
