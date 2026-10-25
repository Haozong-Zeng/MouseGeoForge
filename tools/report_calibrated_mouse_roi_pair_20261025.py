"""Paired full/ROI feature evaluation, shared native predicted masks and denominators."""
from pathlib import Path
import json,sys,datetime
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from mousegeoforge.semseg_training import digest
OUT=ROOT/'output/calibrated_mouse_resume8h_20261025'
def read(p):return json.loads(p.read_text()) if p.exists() else None
def chain_summary(j):
 views=[v for s in j['scenes'] for v in s['views']]
 return dict(j['summary'],scene_denominator=len(j['scenes']),view_denominator=len(views),mean_view_overlap_rmse_mm=float(np.mean([v['rmse_mm_on_overlap'] for v in views if v['rmse_mm_on_overlap'] is not None])),nonempty_overlap_views=sum(v['rmse_mm_on_overlap'] is not None for v in views),mean_view_missing_as_zero_rmse_mm=float(np.mean([v['rmse_mm_missing_as_zero'] for v in views])))
def main():
 result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pairs':{},'cuda_identity_control':read(OUT/'roi_cuda_identity_control_v1.json'),'long_training_identity':read(OUT/'roi_full_identity_control_v1.json'),'limits':['Original native GT and full native support; oracle selection explicitly separate from shared fixed predicted masks','ROI mask generator was trained on admitted TRAIN poses; no unseen-pose proof','Frozen encoder, six correlated source poses, synthetic visible skin excludes glass/fur','No independent real-mm reference, GT alignment, model promotion, threshold or DEV checkpoint selection']}
 required=['seed','steps','planes','rays','fit_only','roi_cache_receipt_sha256','cost_enabled','holdout_frame','gradient_weight','peak_lr','warmup','cosine_floor','ema_decay','train_scene_indices','capture_aabb_mm','cache_receipt_sha256','manifest_sha256','source_sha256']
 for seed in [17,29,43]:
  base={m:OUT/f'roi_depth_s{seed}_{m}_6000_v1' for m in ['full','roi']};oracle={m:read(Path(str(p)+'_dev')/'evaluation.json') for m,p in base.items()};chains={m:read(OUT/f'roi_pipeline_s{seed}_{m}_v1/evaluation.json') for m in base}
  if not all(oracle.values()):continue
  checkpoints={m:torch.load(p/'step_006000.pt',map_location='cpu',weights_only=False) for m,p in base.items()};protocols={m:j['protocol'] for m,j in checkpoints.items()};different=[k for k in required if protocols['full'][k]!=protocols['roi'][k]]
  exposure={m:[(x['step'],x['scene'],x['reference']) for x in j['records']] for m,j in checkpoints.items()};same_exposure=exposure['full']==exposure['roi'];comparable=not different and same_exposure
  pair={'comparable':comparable,'required_differences':different,'same_recorded_scene_view_exposure':same_exposure,'oracle':{m:j['summary'] for m,j in oracle.items()},'chain':{m:chain_summary(j) for m,j in chains.items() if j}}
  if comparable:
   pair['roi_minus_full']={'oracle_mean_view_rmse_mm':oracle['roi']['summary']['rmse_mm']-oracle['full']['summary']['rmse_mm'],'oracle_f1_5mm':oracle['roi']['summary']['pointcloud_fscore']['5']-oracle['full']['summary']['pointcloud_fscore']['5']}
   if len(pair['chain'])==2:pair['roi_minus_full']['chain_f1_5mm']=pair['chain']['roi']['pointcloud_fscore']['5']-pair['chain']['full']['pointcloud_fscore']['5']
  result['pairs'][str(seed)]=pair
 receipt=read(OUT/'roi_cache_v1/cache_receipt.json')
 if receipt:
  result['roi_crop_coverage_diagnostic']={split:{'views':len(v:=[x for x in receipt['diagnostics_after_crop_only'] if x['split']==split]),'mean_gt_fraction_inside_crop':float(np.mean([x['gt_fraction_inside_crop'] for x in v])),'min_gt_fraction_inside_crop':float(min(x['gt_fraction_inside_crop'] for x in v)),'full_fallback_or_fullsize_views':sum(x['crop'][2]==384 for x in v)} for split in ['train','dev']}
 p=OUT/'roi_pair_report_v1.json';p.write_text(json.dumps(result,indent=2)+'\n');(ROOT/'docs/experiments/calibrated_mouse_roi_pair_results_20261025.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'paired_seeds':list(result['pairs']),'differences':{s:v.get('roi_minus_full') for s,v in result['pairs'].items()}}))
if __name__=='__main__':main()
