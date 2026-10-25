"""Frozen originalM predicted boxes for all already-admitted TRAIN/DEV images."""
from pathlib import Path
import datetime as dt
import json
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from PIL import Image
import torch
from mousegeoforge.semseg_training import MouseSemseg,make_encoder,load_items,digest,write_json
from mousegeoforge.models.mouse_semseg import semantic_scores
from mousegeoforge.semseg_pixel_center_20261025 import label_letterbox,native_probability
from mousegeoforge.semseg_roi_20261025 import predicted_box
OUT=ROOT/'output/calibrated_mouse_next12h_20261025';PARENT=ROOT/'output/calibrated_mouse_10h_20261025/semseg_campaign_s17_v1/M_018000/step_018000.pt';RELEASE=ROOT/'vggt-omega/vggt_omega_1b_512.pt'
def main():
 dest=OUT/'semseg_roi_coarse_cache_v1'
 if dest.exists():raise FileExistsError(dest)
 reg=json.loads((OUT/'registrations/semseg_roi_pair_v1.json').read_text())
 for f,h in reg['source_sha256'].items():assert digest(ROOT/f)==h,f
 assert digest(PARENT)==reg['parent_sha256'] and digest(RELEASE)==reg['encoder_sha256']
 rows=load_items(ROOT/'output/calibrated_mouse_10h_20261025/semseg_items.json',ROOT,verify=True);dest.mkdir();(dest/'masks').mkdir()
 torch.set_num_threads(4);cv2.setNumThreads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 model=MouseSemseg(make_encoder(RELEASE)).cuda().eval().requires_grad_(False)
 parent=torch.load(PARENT,map_location='cpu',weights_only=False)
 for name,value in parent['ema'].items():
  if name.startswith('head.'):dict(model.named_parameters())[name].copy_(value)
 began=time.monotonic();result=[]
 with torch.inference_mode():
  for i,row in enumerate(rows):
   image=np.asarray(Image.open(ROOT/row['image']).convert('RGB'));h,w=image.shape[:2]
   # A zero placeholder only builds image letterbox metadata; no GT values enter inference/crop.
   tensor,_,meta=label_letterbox(image,np.zeros((h,w),np.uint8),512,'center_exact')
   scores=semantic_scores(model(tensor[None].cuda(),'M')[-1],(512,512));prob=(scores[:,1]/scores.sum(1).clamp_min(1e-12))[0].cpu().numpy();native,mask=native_probability(prob,meta)
   f=dest/'masks'/(row['id']+'.png');assert cv2.imwrite(str(f),mask*255)
   result.append(dict(**row,roi_box=predicted_box(mask>0),coarse_mask=str(f.relative_to(ROOT)),coarse_mask_sha256=digest(f),coarse_predicted_pixels=int(mask.sum()),coarse_native_hw=[h,w]))
   if (i+1)%100==0:print('COARSE_ROI',i+1,len(rows),round(time.monotonic()-began,1),flush=True);write_json(dest/'progress.json',dict(images=i+1,total=len(rows),seconds=time.monotonic()-began))
 write_json(dest/'items.json',dict(schema='mouse-semseg-v1',items=result,source_manifest_sha256=digest(ROOT/'output/calibrated_mouse_10h_20261025/semseg_items.json'),coarse_model_sha256=digest(PARENT),no_GT_box=True))
 write_json(dest/'completion.json',dict(completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),images=len(result),train=sum(r['split']=='train' for r in result),dev=sum(r['split']=='dev' for r in result),items_sha256=digest(dest/'items.json'),registration_sha256=digest(OUT/'registrations/semseg_roi_pair_v1.json'),parent_sha256=digest(PARENT),encoder_sha256=digest(RELEASE),no_GT_crop=True,empty_fallback_full=sum(r['coarse_predicted_pixels']==0 for r in result),padding_factor=1.4,minimum_side=96,seconds=time.monotonic()-began))
 print('COARSE_COMPLETE',len(result),flush=True)
if __name__=='__main__':main()
