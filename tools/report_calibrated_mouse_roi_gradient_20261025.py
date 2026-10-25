"""Exploratory ROI log-depth-gradient pair; preserve D1 and full denominators."""
from pathlib import Path
import json,sys,datetime
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np
from report_calibrated_mouse_roi_pair_20261025 import chain_summary
OUT=ROOT/'output/calibrated_mouse_resume8h_20261025'
def read(p):return json.loads(p.read_text()) if p.exists() else None
def main():
 result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'schema':'roi-gradient-pair-v1','pairs':{},'exploratory_DEV_reuse':True,'no_threshold_or_checkpoint_selection':True,'no_model_promotion':True,'limits':['Synthetic visible skin only; six correlated poses, no independent real-mm or unseen-pose proof','Shared fixed original M18000 predictions and ROI cache; depth feature mode unchanged','DEV surface audit motivated this separate prospective loss comparison; not fresh confirmatory validation']}
 base_surface=read(OUT/'surface_quality_audit_v1.json');grad_surface=read(OUT/'roi_gradient_surface_audit_v1.json')
 for seed in [17,29,43]:
  paths={'D1':OUT/f'roi_depth_s{seed}_roi_6000_v1','D2':OUT/f'roi_gradient_s{seed}_6000_v1'};oracles={m:read(Path(str(p)+'_dev')/'evaluation.json') for m,p in paths.items()}
  if not all(oracles.values()):continue
  checkpoints={m:torch.load(p/'step_006000.pt',map_location='cpu',weights_only=False) for m,p in paths.items()};a,b=[checkpoints[m]['protocol'] for m in ['D1','D2']]
  ignored={'began','gradient_weight'};different=[k for k in set(a)|set(b) if k not in ignored and a.get(k)!=b.get(k)];same_exposure=[(x['scene'],x['reference']) for x in checkpoints['D1']['records']]==[(x['scene'],x['reference']) for x in checkpoints['D2']['records']]
  comparable=not different and same_exposure and a['gradient_weight']==0 and b['gradient_weight']==.2
  names={'D1':f'roi_pipeline_s{seed}_roi_v1','D2':f'roi_gradient_pipeline_s{seed}_v1'};chains={m:read(OUT/n/'evaluation.json') for m,n in names.items()}
  pair={'comparable':comparable,'unexpected_protocol_differences':different,'same_scene_view_exposure':same_exposure,'oracle':{m:x['summary'] for m,x in oracles.items()},'chain':{m:chain_summary(x) for m,x in chains.items() if x},'surface':{}}
  for mode,surface in [('D1',base_surface),('D2',grad_surface)]:
   if surface and names[mode] in surface['pipelines']:pair['surface'][mode]=surface['pipelines'][names[mode]]['summary']
  if comparable:
   pair['D2_minus_D1']={'oracle_mean_view_rmse_mm':oracles['D2']['summary']['rmse_mm']-oracles['D1']['summary']['rmse_mm']}
   if len(pair['chain'])==2:pair['D2_minus_D1']['chain_f1_5mm']=pair['chain']['D2']['pointcloud_fscore']['5']-pair['chain']['D1']['pointcloud_fscore']['5']
   if len(pair['surface'])==2:
    for key in ['gradient_mae_mm_on_covered_pairs','normal_angle_mean_deg_on_covered','false_step_fraction_on_covered_smooth']:pair['D2_minus_D1'][key]=pair['surface']['D2'][key]-pair['surface']['D1'][key]
  result['pairs'][str(seed)]=pair
 result['completed_pairs']=len(result['pairs']);result['surface_audit_complete']=grad_surface is not None
 (OUT/'roi_gradient_pair_report_v1.json').write_text(json.dumps(result,indent=2)+'\n');(ROOT/'docs/experiments/calibrated_mouse_roi_gradient_results_20261025.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'pairs':result['completed_pairs'],'deltas':{s:v.get('D2_minus_D1') for s,v in result['pairs'].items()}}))
if __name__=='__main__':main()
