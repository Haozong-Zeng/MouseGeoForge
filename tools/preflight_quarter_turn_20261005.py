"""Check all FIT coordinate rotations and render first eligible FIT examples."""
import sys,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import torch,numpy as np
from PIL import Image,ImageDraw
from mousegeoforge.models.quarter_turn_evidence import quarter_turn
from review_pixel_adapter_20261005 import font
C=ROOT/'output/singleview_fusion_cycle_20261005';I=C/'mixed_local_fulltrain_rgb_inputs_v1';O=C/'quarter_turn_preflight_v1'
def main():
    O.mkdir(exist_ok=True);rows=torch.load(I/'evaluation_only.pt',weights_only=False,map_location='cpu');m={r['index']:r for r in json.loads((I/'inference_manifest.json').read_text())};torch.set_num_threads(4);n=0
    for r in rows:
        if r['metadata']['stage']!='fit':continue
        h,w=m[r['index']]['image_hw'];dummy=torch.zeros(1,3,h,w)
        for k in range(4):quarter_turn(dummy,r['target'],r['eligible'],(h,w),k);n+=1
    selected=[];colors=['red','cyan','magenta','lime','orange'];mean=torch.tensor([.485,.456,.406])[None,:,None,None];std=torch.tensor([.229,.224,.225])[None,:,None,None]
    for kind in ['real','single','dual']:
        r=next(r for r in rows if r['metadata']['stage']=='fit' and r['metadata']['kind']==kind);z=m[r['index']];h,w=z['image_hw'];a=np.asarray(Image.open(ROOT/z['image']).convert('RGB')).copy();x=(torch.from_numpy(a).permute(2,0,1)[None].float()/255-mean)/std;tile=Image.new('RGB',(1600,1720),'white');d=ImageDraw.Draw(tile)
        for k in range(4):
            xx,t,hw=quarter_turn(x,r['target'],r['eligible'],(h,w),k);p=Image.fromarray(((xx[...,:hw[0],:hw[1]]*std+mean)[0].permute(1,2,0).clamp(0,1)*255).round().byte().numpy());draw=ImageDraw.Draw(p)
            for q,j in r['eligible'].nonzero().tolist():
                px,py=(t[q,j]*torch.tensor([hw[1],hw[0]])).tolist();draw.ellipse((px-5,py-5,px+5,py+5),outline=colors[j],width=2);draw.text((px+6,py),str([1,15,16,17,18][j]),fill=colors[j],font=font(16))
            p.thumbnail((780,790));row,col=divmod(k,2);tile.paste(p,(col*800+(800-p.width)//2,row*850+50));d.text((col*800+10,row*850+10),f'{kind} index{r["index"]} CCW{k*90}',fill='black',font=font(22))
        tile.save(O/(kind+'.jpg'),quality=95);selected.append(r['index'])
    (O/'results.json').write_text(json.dumps(dict(tests=7,fit_coordinate_transforms_checked=n,selected=selected,no_mirror=True,no_identity_swap=True,no_interpolation=True),indent=2));print(n,selected)
if __name__=='__main__':main()
