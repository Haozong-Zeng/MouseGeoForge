"""Original-native semantic mask audit for completed evaluation receipts."""
from pathlib import Path
import argparse,json,datetime,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from PIL import Image
from mousegeoforge.semseg_training import digest,mask_metrics
OLD=ROOT/'output/calibrated_mouse_10h_20261025'
def audit(directory,output):
 assert not output.exists();saved=json.loads((directory/'evaluation.json').read_text());items=json.loads((OLD/'semseg_items.json').read_text())['items'];items=[x for x in items if x['split']=='dev'];expected={x['id']:x for x in items};legacy={x['id']:x for x in saved['ema']};assert set(legacy)==set(expected) and len(expected)==879
 rows=[]
 for id,row in expected.items():
  label=ROOT/row['label'];assert digest(label)==row['label_sha256']
  if label.suffix=='.npz':
   with np.load(label) as z:gt=z['body'].astype(np.uint8)
  else:
   raw=np.array(Image.open(label).convert('L'));gt=raw if row.get('label_encoding')=='class_index_0_1_255' else (raw>0).astype(np.uint8)
  path=directory/'masks_ema'/(id+'.png');prediction=cv2.imread(str(path),0)
  if prediction is None:raise FileNotFoundError(path)
  prediction=(prediction>0).astype(np.uint8);assert prediction.shape==gt.shape;metrics=mask_metrics(prediction,gt,1)
  rows.append(dict(id=id,domain=row['domain'],bottom=row['bottom'],group=row['group'],native_original=metrics,legacy_roundtrip_native=legacy[id]['native'],label_sha256=digest(label),prediction_sha256=digest(path)))
 groups={}
 for name,subset in [('all',rows),*[(f'{d}/{b}',[x for x in rows if (x['domain'],x['bottom'])==(d,b)]) for d in ['real','synthetic'] for b in [True,False]]]:
  groups[name]={'images':len(subset),'native_original_iou':float(np.mean([x['native_original']['iou'] for x in subset])),'native_original_dice':float(np.mean([x['native_original']['dice'] for x in subset])),'native_original_boundary_f1_tol1':float(np.mean([x['native_original']['boundary_f1'] for x in subset])),'legacy_roundtrip_iou':float(np.mean([x['legacy_roundtrip_native']['iou'] for x in subset]))}
 result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checkpoint_sha256':saved['checkpoint_sha256'],'evaluation_receipt_sha256':digest(directory/'evaluation.json'),'source_sha256':digest(Path(__file__)),'items_sha256':digest(OLD/'semseg_items.json'),'weights':'EMA','native_truth_policy':'original label pixels, no forward/inverse resize','denominator':879,'groups':groups,'rows':rows,'limits':['Manual real masks vs synthetic skin surrogate remain separate','Native boundary tolerance1px differs in physical size/resolution','No semantic-union instance detection, real-mm or model-promotion claim']};output.write_text(json.dumps(result,indent=2)+'\n');return {k:v for k,v in result.items() if k!='rows'}
def main():
 p=argparse.ArgumentParser();p.add_argument('--evaluation-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();print(json.dumps(audit(args.evaluation_dir,args.output)))
if __name__=='__main__':main()
