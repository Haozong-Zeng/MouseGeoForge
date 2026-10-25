"""Fixed representative crop QA and full-denominator coverage, never crop selection."""
from pathlib import Path
import datetime as dt
import json
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from PIL import Image
from mousegeoforge.semseg_training import load_items,digest
from mousegeoforge.semseg_pixel_center_20261025 import native_label
OUT=ROOT/'output/calibrated_mouse_next12h_20261025';CACHE=OUT/'semseg_roi_coarse_cache_v1'
def main():
 output=OUT/'semseg_roi_crop_review_v1';assert not output.exists();output.mkdir()
 receipt=json.loads((CACHE/'completion.json').read_text());assert digest(CACHE/'items.json')==receipt['items_sha256'];rows=load_items(CACHE/'items.json',ROOT,verify=True)
 groups={};tiles=[];examples=[]
 for row in rows:
  gt=native_label(row,ROOT)==1;x0,y0,x1,y1=row['roi_box'];h,w=gt.shape;assert 0<=x0<x1<=w and 0<=y0<y1<=h
  key=f"{row['split']}/{row['domain']}/{row['bottom']}";total=int(gt.sum());covered=int(gt[y0:y1,x0:x1].sum());groups.setdefault(key,[]).append(dict(id=row['id'],gt_pixels=total,covered_pixels=covered,coverage=covered/max(1,total)))
 for split in ['train','dev']:
  for domain,bottom in [('real',True),('real',False),('synthetic',True),('synthetic',False)]:
   row=sorted([r for r in rows if (r['split'],r['domain'],r['bottom'])==(split,domain,bottom)],key=lambda r:r['id'])[0]
   image=np.asarray(Image.open(ROOT/row['image']).convert('RGB'));x0,y0,x1,y1=row['roi_box'];full=image.copy();cv2.rectangle(full,(x0,y0),(x1-1,y1-1),(0,255,0),max(1,round(image.shape[1]/256)))
   def tile(a):
    h,w=a.shape[:2];s=256/max(h,w);a=cv2.resize(a,(max(1,round(w*s)),max(1,round(h*s))));pad=np.zeros((280,256,3),np.uint8);pad[:a.shape[0],:a.shape[1]]=a;return pad
   a,b=tile(full),tile(image[y0:y1,x0:x1]);cv2.putText(a,f'{split} {domain} bottom={bottom}',(2,270),0,.35,(255,255,255),1);cv2.putText(b,'predicted ROI / display only',(2,270),0,.35,(255,255,255),1);tiles.append(np.concatenate([a,b],1));examples.append(dict(id=row['id'],box=row['roi_box'],image_sha256=row['image_sha256'],coarse_mask_sha256=row['coarse_mask_sha256']))
 image=np.concatenate([np.concatenate(tiles[i:i+2],1) for i in range(0,8,2)],0);assert cv2.imwrite(str(output/'fixed_crop_gallery.png'),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
 summary={key:dict(images=len(rr),mean_gt_body_crop_coverage=float(np.mean([r['coverage'] for r in rr])),minimum_gt_body_crop_coverage=min(r['coverage'] for r in rr),images_with_any_gt_outside_crop=sum(r['covered_pixels']<r['gt_pixels'] for r in rr),gt_pixels=sum(r['gt_pixels'] for r in rr),covered_pixels=sum(r['covered_pixels'] for r in rr)) for key,rr in groups.items()}
 result=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),source_sha256=digest(Path(__file__)),items_sha256=digest(CACHE/'items.json'),summary=summary,examples_fixed_before_visual_review=examples,gallery_sha256=digest(output/'fixed_crop_gallery.png'),GT_policy='Diagnostic coverage only. GT never entered fixedcoarse inference/boxgeneration; no crop/threshold/checkpoint selection; fullnative GTdenominators preserved during model evaluation.',actual_visual_review='pending separate inspection')
 (output/'coverage.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary))
if __name__=='__main__':main()
