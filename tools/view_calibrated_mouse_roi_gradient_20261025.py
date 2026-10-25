"""Fixed first-scene CPU gallery of registered ROI gradient comparison; no model calls."""
from pathlib import Path
import json,hashlib,datetime as dt,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from export_calibrated_mouse_pointcloud_review_20261025 import palette
import numpy as np
from PIL import Image,ImageDraw
OUT=ROOT/'output/calibrated_mouse_resume8h_20261025';DEST=OUT/'roi_gradient_visual_review_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert not DEST.exists();DEST.mkdir();manifest=ROOT/'output/calibrated_mouse_10h_20261025/data_manifest.json';scenes=json.loads(manifest.read_text())['scenes'];pins={str(manifest.relative_to(ROOT)):sha(manifest)};rows=[]
 for index in [0,360]:
  scene=scenes[index];assert scene['split']=='dev'
  for view in [0,2]:
   v=scene['views'][view];ip=ROOT/v['image'];assert sha(ip)==v['image_sha256'];pins[v['image']]=sha(ip);rgb=np.asarray(Image.open(ip).convert('RGB'));saved=[]
   for name in ['roi_pipeline_s17_roi_v1','roi_gradient_pipeline_s17_v1']:
    npz=OUT/name/f'scene{index:06d}'/f'view{view}.npz';pins[str(npz.relative_to(ROOT))]=sha(npz)
    with np.load(npz) as z:saved.append({k:z[k].copy() for k in ['depth_mm','mask','K','E']})
   a,b=saved
   for key in ['mask','K','E']:np.testing.assert_array_equal(a[key],b[key])
   mask=a['mask'];y,x=np.where(mask);x0,x1=max(0,int(x.min())-12),min(rgb.shape[1],int(x.max())+13);y0,y1=max(0,int(y.min())-12),min(rgb.shape[0],int(y.max())+13)
   values=np.concatenate([z['depth_mm'][mask] for z in saved]);lo,hi=float(values.min()),float(values.max())
   images=[rgb[y0:y1,x0:x1]]+[palette(z['depth_mm'],mask,lo,hi)[y0:y1,x0:x1] for z in saved]
   diff=np.abs(a['depth_mm']-b['depth_mm']);images.append(palette(diff,mask,0,5)[y0:y1,x0:x1]);rows.append((f'{scene["source"]} scene{index} v{view} | shared Z [{lo:.1f},{hi:.1f}]mm | absolute difference display0..5mm',images))
 canvas=Image.new('RGB',(1200,4*300),'white');draw=ImageDraw.Draw(canvas)
 for i,(label,images) in enumerate(rows):
  draw.text((4,i*300+3),label,fill='black')
  for j,(data,title) in enumerate(zip(images,['native RGB','ROI D1','ROI +.2 gradient','absolute Z difference'])):
   im=Image.fromarray(data);im.thumbnail((292,258));canvas.paste(im,(j*300+(300-im.width)//2,i*300+37+(258-im.height)//2));draw.text((j*300+4,i*300+20),title,fill='black')
 canvas.save(DEST/'comparison.png')
 receipt=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),source_sha256=sha(__file__),input_sha256=pins,image_sha256=sha(DEST/'comparison.png'),selection='Fixed first admitted DEV scene per source0/360,views0/2,seed17; same scenes as prior pointcloud review; no error ranking',model_calls=0,GT_inputs=False,predicted_mask_crop='DISPLAY ONLY',palette='Shared per-view D1/D2 native camera-Z extrema; difference0..5mm fixed display; no numeric filtering',limitations='Single-seed representative visual diagnosis; all-three-seed quantitative report governs tradeoffs; synthetic visible skin only; no inference or metric mutation')
 (DEST/'execution.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print('gallery complete')
if __name__=='__main__':main()
