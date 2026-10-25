"""Fixed DEV examples, display-only native segmentation comparison; zero model calls."""
from pathlib import Path
import datetime as dt,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from PIL import Image
from mousegeoforge.semseg_training import load_items,digest
from mousegeoforge.semseg_pixel_center_20261025 import native_label
OUT=ROOT/'output/calibrated_mouse_next12h_20261025'
def main():
 dest=OUT/'semseg_roi_result_review_v1';assert not dest.exists();dest.mkdir()
 reg=json.loads((OUT/'registrations/semseg_roi_result_review_v1.json').read_text());assert digest(Path(__file__))==reg['source_sha256'][str(Path(__file__).relative_to(ROOT))]
 rows=load_items(OUT/'semseg_roi_coarse_cache_v1/items.json',ROOT,verify=True);lookup={r['id']:r for r in rows};tiles=[];examples=[]
 def tile(a,title):
  h,w=a.shape[:2];scale=300/max(h,w);a=cv2.resize(a,(max(1,round(w*scale)),max(1,round(h*scale))),interpolation=cv2.INTER_NEAREST);pad=np.zeros((325,300,3),np.uint8);pad[:a.shape[0],:a.shape[1]]=a;cv2.putText(pad,title,(3,319),0,.43,(255,255,255),1);return pad
 for identifier in reg['selected_ids']:
  row=lookup[identifier];assert row['split']=='dev';rgb=np.asarray(Image.open(ROOT/row['image']).convert('RGB'));gt=native_label(row,ROOT)==1;x0,y0,x1,y1=row['roi_box'];ss=(slice(y0,y1),slice(x0,x1));record=dict(id=identifier,domain=row['domain'],bottom=row['bottom'],display_box=row['roi_box'],image_sha256=digest(ROOT/row['image']),label_sha256=row['label_sha256'],masks={})
  panels=[tile(rgb[ss],f"{row['domain']} bottom={row['bottom']} RGB"),tile(np.repeat((gt[ss]*255).astype(np.uint8)[...,None],3,2),'Original native GT')]
  for arm in ('full','roi'):
   file=OUT/f'semseg_roi_pair_v2/{arm}_dev/masks_ema/{identifier}.png';mask=np.asarray(Image.open(file))>0;assert mask.shape==gt.shape
   colors=rgb.copy();colors[gt&~mask]=[255,0,0];colors[mask&~gt]=[0,100,255];colors[gt&mask]=(colors[gt&mask]*.5+np.array([0,255,0])*.5).astype(np.uint8);panels.append(tile(colors[ss],arm+' TPgreen/FNred/FPblue'));record['masks'][arm]=dict(path=str(file.relative_to(ROOT)),sha256=digest(file))
  tiles.append(np.concatenate(panels,1));examples.append(record)
 gallery=np.concatenate(tiles,0);assert cv2.imwrite(str(dest/'fixed_native_comparison.png'),cv2.cvtColor(gallery,cv2.COLOR_RGB2BGR))
 receipt=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),registration_sha256=digest(OUT/'registrations/semseg_roi_result_review_v1.json'),source_sha256=digest(Path(__file__)),examples=examples,gallery_sha256=digest(dest/'fixed_native_comparison.png'),model_calls=0,display_only=True,limitations=['First sorted2DEV IDs per4strata, no outcome selection','Fixed teacher ROI used solely for display; no scoring/filter/exports changed','Single seed/reused DEV and six correlated synthetic poses; no real-mm/generalization claim'],actual_visual_review='pending')
 (dest/'execution.json').write_text(json.dumps(receipt,indent=2)+'\n');print(str(dest/'fixed_native_comparison.png'))
if __name__=='__main__':main()
