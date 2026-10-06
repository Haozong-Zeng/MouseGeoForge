"""Show accepted target masks/boxes versus frozen predictions, no relabeling."""
import html,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from PIL import Image,ImageDraw
from run_surface_curve_pilot_20261003 import load_item
from run_depth_pose_ablation import write_json
CYCLE=ROOT/'output/core23_spatial_cycle_20261004';OUT=CYCLE/'review_box_contract_v1'


def main():
    assert not OUT.exists();OUT.mkdir();torch.set_num_threads(2);records=json.loads((CYCLE/'detection_features_v1/manifest.json').read_text())['records'];rows=[]
    for r in records:rows.extend(torch.load(ROOT/r['path'],map_location='cpu',weights_only=False)['observations'])
    selected=[]
    for kind in ['real','synthetic','curve']:
        ids=[i for i,o in enumerate(rows) if o['metadata']['record']['kind']==kind and o['metadata']['detection_complete'] and (kind=='curve' or o['metadata']['record']['role']=='validation')]
        ids.sort(key=lambda i:float((rows[i]['boxes'][rows[i]['box_valid'],2:].prod(-1)).mean()))
        if ids:selected.extend([ids[len(ids)//2],ids[-1]])
    page=['<!doctype html><meta charset="utf-8"><title>Mask-derived box contract</title><style>body{background:#20232b;color:white;font:16px sans-serif}img{max-width:100%}</style><h1>Detection target contract audit</h1><p>Left blue=accepted instance masks, green=their exact normalized pixel-edge bounding boxes. Right magenta=frozen predicted boxes with confidence>=.5. No source masks/labels edited. Existing R12oldmask protected. Full visible instance includes tail if accepted mask contains it.15realTRAINviews without complete masks excluded from detection-only training, not treated as background.</p>'];meta=[]
    for index,i in enumerate(selected):
        o=rows[i];m=o['metadata'];r=m['record'];cache=ROOT/f'output/surface_curve_cycle_20261003/training_v1/pilot/cache/{r["key"]}.pt';item=torch.load(cache,map_location='cpu',weights_only=False) if cache.exists() else load_item(r);v=m['view_index']
        pix=(item['batch']['images'][0,v].permute(1,2,0).numpy()*255).clip(0,255).astype('uint8');h,w=pix.shape[:2];pw=450;ph=round(h*pw/w);im=Image.fromarray(pix).resize((pw,ph));canvas=Image.new('RGB',(pw*2,ph+35),(22,24,30));draw=ImageDraw.Draw(canvas)
        mask=item['batch']['masks'][0,v].any(0).numpy();alpha=Image.fromarray((mask*100).astype('uint8')).resize((pw,ph),Image.Resampling.NEAREST);canvas.paste(Image.composite(Image.new('RGB',(pw,ph),(30,150,255)),im,alpha),(0,35));canvas.paste(im,(pw,35));draw.text((5,8),'accepted masks + GT boxes',fill='white');draw.text((pw+5,8),'frozen predicted boxes >=0.5',fill='white')
        scores=o['original_objectness'].sigmoid().numpy()
        for col,boxes,valid,color in [(0,o['boxes'].numpy(),o['box_valid'].numpy(),'#66ff88'),(1,o['original_boxes'].numpy(),scores>=.5,'#ee77ff')]:
            for j in np.flatnonzero(valid):
                cx,cy,bw,bh=boxes[j];x0,y0=np.array([cx-bw/2,cy-bh/2])*[pw,ph]+[col*pw,35];x1,y1=np.array([cx+bw/2,cy+bh/2])*[pw,ph]+[col*pw,35];draw.rectangle((x0,y0,x1,y1),outline=color,width=2)
        name=f'{index:02d}_{r["kind"]}.jpg';canvas.save(OUT/name,quality=95);meta.append(dict(file=name,observation=i,metadata=m));page.append('<h2>'+html.escape(str(r))+'</h2><img src="'+name+'">')
    write_json(OUT/'manifest.json',meta);(OUT/'index.html').write_text('\n'.join(page),encoding='utf-8');dest=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-Core23-Spatial-20261004')/OUT.name;assert not dest.exists();shutil.copytree(OUT,dest)


if __name__=='__main__':main()
