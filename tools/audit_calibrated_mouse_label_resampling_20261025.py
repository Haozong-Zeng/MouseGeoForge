"""Ground-truth-only resize operator control; no model ceiling or metric replacement."""
from pathlib import Path
import json,datetime,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from PIL import Image
from mousegeoforge.semseg_training import digest,mask_metrics
OUT=ROOT/'output/calibrated_mouse_resume8h_20261025';OLD=ROOT/'output/calibrated_mouse_10h_20261025'
def resize_roundtrip(a,side,operator):
 h,w=a.shape;s=side/max(h,w);size=(max(1,round(w*s)),max(1,round(h*s)))
 return cv2.resize(cv2.resize(a,size,interpolation=operator),(w,h),interpolation=operator)
def main():
 target=OUT/'label_resampling_control_v1.json';assert not target.exists()
 items=[r for r in json.loads((OLD/'semseg_items.json').read_text())['items'] if r['split']=='dev'];assert len(items)==879
 rows=[]
 for r in items:
  path=ROOT/r['label'];assert digest(path)==r['label_sha256']
  if path.suffix=='.npz':
   with np.load(path) as z:a=z['body'].astype(np.uint8)
  else:
   a=np.array(Image.open(path).convert('L'));a=a if r.get('label_encoding')=='class_index_0_1_255' else (a>0).astype(np.uint8)
  row=dict(id=r['id'],domain=r['domain'],bottom=r['bottom'],native_hw=list(a.shape),label_sha256=r['label_sha256'])
  for name,op in [('legacy_nearest',cv2.INTER_NEAREST),('center_nearest_exact',cv2.INTER_NEAREST_EXACT)]:row[name]=mask_metrics(resize_roundtrip(a,512,op),a,1)
  rows.append(row)
 groups={}
 for name,selected in [('all',rows),*[(f'{d}/{b}',[r for r in rows if r['domain']==d and r['bottom']==b]) for d in ['real','synthetic'] for b in [True,False]]]:
  groups[name]=dict(images=len(selected),**{op:{k:float(np.mean([r[op][k] for r in selected])) for k in ['iou','dice','boundary_f1']} for op in ['legacy_nearest','center_nearest_exact']})
 result=dict(schema='GT-label-resize-operator-control-v1',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=digest(__file__),items_sha256=digest(OLD/'semseg_items.json'),side=512,groups=groups,rows=rows,limitations=['GT-only self-agreement, not a model accuracy upper bound or new model result.','No current training target, method, native primary score or prediction changed.','Two operators fixed before full audit, no tuning.'])
 target.write_text(json.dumps(result,indent=2)+'\n');compact={k:v for k,v in result.items() if k!='rows'};compact['full_audit_sha256']=digest(target)
 (ROOT/'docs/experiments/calibrated_mouse_label_resampling_20261025.json').write_text(json.dumps(compact,indent=2)+'\n');print(json.dumps(groups))
if __name__=='__main__':main()
