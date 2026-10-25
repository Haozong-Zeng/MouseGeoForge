"""Audit exported masks against original native labels, preserving legacy metrics."""
import pathlib,json,sys,datetime
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,cv2
from PIL import Image
from mousegeoforge.semseg_training import digest,mask_metrics,read_sample,native_mask
OUT=ROOT/'output/calibrated_mouse_resume8h_20261025';OLD=ROOT/'output/calibrated_mouse_10h_20261025'
def main():
 rows=[r for r in json.loads((OLD/'semseg_items.json').read_text())['items'] if r['split']=='dev'];assert len(rows)==879
 result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':digest(pathlib.Path(__file__)),'denominator':879,'arms':{},'interpretation':'Legacy native evaluator compared resized-back labels. Original native labels are primary here; prior artifacts retained unchanged.'}
 for arm in ['L','M']:
  saved=json.loads((OUT/f'{arm}_dev_v1/evaluation.json').read_text());predrows={x['id']:x for x in saved['ema']};audit=[]
  for row in rows:
   path=ROOT/row['label'];assert digest(path)==row['label_sha256']
   if path.suffix=='.npz':
    with np.load(path) as z:truth=z['body'].astype(np.uint8)
   else:
    raw=np.array(Image.open(path).convert('L'));truth=raw if row.get('label_encoding')=='class_index_0_1_255' else (raw>0).astype(np.uint8)
   image,label,meta=read_sample(row,ROOT,512);roundtrip=native_mask(label.numpy(),meta)
   p=OUT/f'{arm}_dev_v1/masks_ema'/(row['id']+'.png');pred=(cv2.imread(str(p),0)>0).astype(np.uint8);assert pred.shape==truth.shape
   actual=mask_metrics(pred,truth,1);replayed=mask_metrics(pred,roundtrip,1);assert abs(replayed['iou']-predrows[row['id']]['native']['iou'])<1e-12
   audit.append({'id':row['id'],'domain':row['domain'],'bottom':row['bottom'],'group':row['group'],'native_original':actual,'legacy_roundtrip':replayed,'gt_changed_pixels':int((roundtrip!=truth).sum()),'gt_pixels':int((truth==1).sum()),'mask_sha256':digest(p)})
  groups={}
  for name,subset in [('all',audit),*[(f'{d}/{b}',[x for x in audit if (x['domain'],x['bottom'])==(d,b)]) for d in ['real','synthetic'] for b in [True,False]]]:
   groups[name]={'images':len(subset),'native_original_iou':float(np.mean([x['native_original']['iou'] for x in subset])),'legacy_roundtrip_iou':float(np.mean([x['legacy_roundtrip']['iou'] for x in subset])),'native_original_boundary_f1_tol1':float(np.mean([x['native_original']['boundary_f1'] for x in subset])),'changed_gt_pixels':sum(x['gt_changed_pixels'] for x in subset)}
  result['arms'][arm]={'checkpoint_sha256':saved['checkpoint_sha256'],'groups':groups,'rows':audit}
 pipeline=json.loads((OUT/'pipeline_D1_M_dev_v1/evaluation.json').read_text());views=[v for x in pipeline['scenes'] for v in x['views']]
 result['pipeline']={'summary':pipeline['summary'],'scene_denominator':len(pipeline['scenes']),'view_denominator':len(views),'mean_view_overlap_rmse_mm':float(np.mean([v['rmse_mm_on_overlap'] for v in views])),'mean_view_missing_as_zero_rmse_mm':float(np.mean([v['rmse_mm_missing_as_zero'] for v in views])),'overlap_metric_excludes_missed_foreground':True,'no_gt_scale_alignment':True}
 control=json.loads((OLD/'controls_report.json').read_text())['held_source_frame_pairs'];result['held_source_frame_pairs']={k:v for k,v in control.items()}
 empty=OUT/'empty_background_v1';e=json.loads((empty/'M_common_endpoint_eval/evaluation.json').read_text());assert json.loads((empty/'formal_visual_review.json').read_text())['passed'];result['empty_diagnostic']={k:v for k,v in e.items() if k!='views'};result['empty_diagnostic']['positive_frames']=[x for x in e['views'] if x['any_foreground']]
 result['limits']=['Synthetic visible skin excludes fur/glass; no real-mm reference','6 correlated source poses; no independent unseen-real-pose proof','Native boundary tolerance1 pixel differs across image resolutions','Semantic union bbox is not an instance detector','No DEV checkpoint, threshold, or component-filter selection']
 p=OUT/'native_metric_audit_v1.json';assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n')
 (ROOT/'docs/experiments/calibrated_mouse_resume8h_native_audit_20261025.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'groups':{a:v['groups'] for a,v in result['arms'].items()},'pipeline':result['pipeline'],'empty':result['empty_diagnostic']['summary']}))
if __name__=='__main__':main()
