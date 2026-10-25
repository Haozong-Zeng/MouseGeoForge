"""Descriptive geometric failure strata and cross-view comparisons; no model selection."""
from pathlib import Path
import json,datetime,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from mousegeoforge.semseg_training import digest
OUT=ROOT/'output/calibrated_mouse_resume8h_20261025'
def read(p):return json.loads(p.read_text()) if p.exists() else None
def mean(values):return float(np.mean(values)) if values else None
def main():
 cross=read(OUT/'crossview_audit_v1.json');gradient=read(OUT/'roi_gradient_crossview_audit_v1.json');surface=read(OUT/'surface_quality_audit_v1.json');result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':digest(Path(__file__)),'crossview':{},'foreground_area_strata':{},'worst_surface_cases':read(OUT/'surface_worst_cases_selection_v1.json'),'gradient_crossview_available':gradient is not None,'limits':['All descriptive DEV reuse, no model/threshold/checkpoint selection','Cross-view support selected by diagnostic GT visibility; consistency is not correctness','Only~14percent of candidate GT directed rays are covisible with strict4corner skin criterion; full-body/missing accuracy is separate','Surface local gradients/normals target GTbody; falsepositive surfaces counted in full-chain cloud metrics','No real-mm accuracy, unseen-pose, fur/glass surface or instance-count claim']}
 if cross:
  result['crossview_registration_sha256']=digest(OUT/'crossview_registration_v1.json');result['crossview_audit_sha256']=digest(OUT/'crossview_audit_v1.json')
  for name,value in cross['pipelines'].items():result['crossview'][name]={'summary':value['summary'],'by_source':value['by_source']}
  for seed in [17,29,43]:
   full=cross['pipelines'][f'roi_pipeline_s{seed}_full_v1']['summary'];roi=cross['pipelines'][f'roi_pipeline_s{seed}_roi_v1']['summary'];assert full['GT_covisible_pixels']==roi['GT_covisible_pixels']
   result.setdefault('roi_minus_full_consistency',{})[str(seed)]={key:roi[key]-full[key] for key in ['prediction_coverage','mean_absolute_consistency_mm_on_covered','rmse_consistency_mm_on_covered','fraction_consistent_le5mm_over_GT_covisible']}
 if gradient:
  for name,value in gradient['pipelines'].items():result['crossview'][name]={'summary':value['summary'],'by_source':value['by_source']}
  result['gradient_minus_ROI_D1_consistency']={}
  for seed in [17,29,43]:
   name=f'roi_gradient_pipeline_s{seed}_v1'
   if name not in gradient['pipelines']:continue
   d2=gradient['pipelines'][name]['summary'];d1=cross['pipelines'][f'roi_pipeline_s{seed}_roi_v1']['summary'];assert d1['GT_covisible_pixels']==d2['GT_covisible_pixels'];result['gradient_minus_ROI_D1_consistency'][str(seed)]={k:d2[k]-d1[k] for k in ['prediction_coverage','mean_absolute_consistency_mm_on_covered','rmse_consistency_mm_on_covered','fraction_consistent_le5mm_over_GT_covisible']}
 for mode in ['full','roi']:
  name=f'roi_pipeline_s17_{mode}_v1';evaluation=read(OUT/name/'evaluation.json');local={(x['index'],x['view']):x for x in surface['pipelines'][name]['views']};rows=[]
  # The scene index is stored in local audit and uniquely identifies manifest ID.
  byid={(x['scene'],x['view']):x for x in surface['pipelines'][name]['views']}
  for scene in evaluation['scenes']:
   for v in scene['views']:
    area=v['ground_truth_foreground_pixels']/(384*384);row={'source':scene['source'],'view':v['view'],'fraction':area,**v,'local_gradient_mae':byid[(scene['id'],v['view'])]['gradient_mae_mm_on_covered_pairs']};rows.append(row)
  groups={}
  for label,condition in [('<1percent',lambda a:a<.01),('1to5percent',lambda a:.01<=a<.05),('atleast5percent',lambda a:a>=.05)]:
   values=[x for x in rows if condition(x['fraction'])];groups[label]={'views':len(values),'mean_native_iou':mean([x['segmentation']['iou'] for x in values]),'mean_GT_depth_coverage':mean([x['gt_depth_coverage'] for x in values]),'mean_overlap_RMSE_mm':mean([x['rmse_mm_on_overlap'] for x in values if x['rmse_mm_on_overlap'] is not None]),'overlap_evaluable_views':sum(x['rmse_mm_on_overlap'] is not None for x in values),'mean_missing_zero_RMSE_mm':mean([x['rmse_mm_missing_as_zero'] for x in values]),'mean_local_gradient_MAE_mm':mean([x['local_gradient_mae'] for x in values if x['local_gradient_mae'] is not None])}
  result['foreground_area_strata'][mode]=groups
 review=read(OUT/'geometric_failure_visual_review_v1.json');result['visual_review']=review
 (OUT/'geometric_failures_report_v1.json').write_text(json.dumps(result,indent=2)+'\n');(ROOT/'docs/experiments/calibrated_mouse_geometric_failures_20261025.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'roi_minus_full':result.get('roi_minus_full_consistency'),'area_strata':result['foreground_area_strata'],'gradient_followup_available':gradient is not None}))
if __name__=='__main__':main()
