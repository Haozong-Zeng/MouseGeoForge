"""Visual review of fixed resampling diagnostic; GT crop is display-only."""
from pathlib import Path
import json,sys,datetime
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image,ImageDraw
from mousegeoforge.semseg_training import digest
OUT=ROOT/'output/calibrated_mouse_resume8h_20261025'
def main():
 d=json.loads((OUT/'inverse_resampling_audit_v1.json').read_text());items={r['id']:r for r in json.loads((ROOT/'output/calibrated_mouse_10h_20261025/semseg_items.json').read_text())['items']};selected=[]
 for domain in ['real','synthetic']:
  for bottom in [True,False]:
   rows=[r for r in d['rows'] if r['arm']=='M' and r['domain']==domain and r['bottom']==bottom]
   selected.append(max(rows,key=lambda r:(abs(r['bilinear_probability_fixed_half']['iou']-r['legacy_nearest_mask']['iou']),r['id'])))
 receipt=dict(selection='M; maximum absolute IoU delta in each fixed domain/bottom stratum; diagnostic ranking only',ids=[r['id'] for r in selected],GT_display_crop_only=True,source_sha256=digest(__file__),audit_sha256=digest(OUT/'inverse_resampling_audit_v1.json'))
 (OUT/'inverse_resampling_visual_selection_v1.json').write_text(json.dumps(receipt,indent=2)+'\n')
 panel=Image.new('RGB',(1280,4*300),(245,245,245));draw=ImageDraw.Draw(panel)
 for i,r in enumerate(selected):
  item=items[r['id']];path=ROOT/item['label']
  if path.suffix=='.npz':
   with np.load(path) as z:gt=z['body'].astype(bool)
  else:gt=np.array(Image.open(path).convert('L'))>0
  base=np.array(Image.open(ROOT/item['image']).convert('RGB'));old=np.array(Image.open(OUT/'M_dev_v1/masks_ema'/(r['id']+'.png')))>0
  with np.load(OUT/'M_dev_v1/masks_ema'/(r['id']+'_probability.npz')) as z:new=z['foreground'].astype(np.float32)>.5
  y,x=np.where(gt|old|new);x0,x1=max(0,int(x.min())-16),min(base.shape[1],int(x.max())+17);y0,y1=max(0,int(y.min())-16),min(base.shape[0],int(y.max())+17)
  for j,name in enumerate(['GT','nearest','bilinear>.5','changed (green add/red remove)']):
   picture=base.astype(float)*.6
   colors=[]
   if j==0:colors=[(gt,[0,230,0])]
   elif j in [1,2]:
    pred=old if j==1 else new;colors=[(pred&gt,[0,230,0]),(pred&~gt,[255,0,255]),(~pred&gt,[255,220,0])]
   else:colors=[(new&~old,[0,255,0]),(old&~new,[255,0,0])]
   for mask,color in colors:picture[mask]=.35*base[mask]+.65*np.array(color)
   im=Image.fromarray(picture.astype(np.uint8)[y0:y1,x0:x1]);im.thumbnail((316,260));panel.paste(im,(j*320+(320-im.width)//2,i*300+35+(260-im.height)//2))
   text=name
   if j==0:text=f"{r['domain']} bottom={r['bottom']} {r['id'][:7]} GT"
   if j in [1,2]:text+=f" IoU {r['legacy_nearest_mask' if j==1 else 'bilinear_probability_fixed_half']['iou']:.4f}"
   draw.text((j*320+4,i*300+10),text,fill=(0,0,0))
 output=OUT/'inverse_resampling_visual_v1.png';panel.save(output)
 print(output)
if __name__=='__main__':main()
