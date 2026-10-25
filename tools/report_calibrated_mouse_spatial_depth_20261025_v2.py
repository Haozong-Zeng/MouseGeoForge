"""Finite spatial-neck paired outcomes; allthree comparable seeds required for means."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import math
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from mousegeoforge.semseg_training import digest
OUT=ROOT/'output/calibrated_mouse_next12h_20261025'
def read(p,default=None):return json.loads(p.read_text()) if p.exists() else default
def main():
 reg=read(OUT/'registrations/spatial_depth_pair_v2.json');result=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),registration_sha256=digest(OUT/'registrations/spatial_depth_pair_v2.json'),arms={},all_three_pairs_comparable=False,equal_seed_mean=None,limits=reg['limitations']);exposures={}
 surface=read(OUT/'spatial_depth_surface_audit_v2.json',{}).get('pipelines',{});cross=read(OUT/'spatial_depth_crossview_audit_v2.json',{}).get('pipelines',{});paired=[]
 for seed in (17,29,43):
  protocols={};records={};seed_complete=True
  for kernel in (1,3):
   name=f'spatial_depth_s{seed}_k{kernel}_6000_v2';folder=OUT/name;chain=OUT/f'spatial_pipeline_s{seed}_k{kernel}_v2';oracle=OUT/(name+'_dev');required=[folder/'completion.json',folder/'step_006000.pt',oracle/'evaluation.json',chain/'evaluation.json'];key=f'{seed}/{kernel}'
   if not all(p.exists() for p in required):result['arms'][key]=dict(status='missing_or_partial',missing=[str(p.relative_to(OUT)) for p in required if not p.exists()]);seed_complete=False;continue
   receipt=read(required[0]);ck=torch.load(required[1],map_location='cpu',weights_only=False);protocol=ck['protocol'];assert protocol['score_bias_trainable'] is False;assert ck['step']==6000 and len(ck['records'])==6000 and receipt['passed'];assert (protocol['neck_kernel'],protocol['neck_parameters'],protocol['seed'],protocol['steps'],protocol['gradient_weight'])==(kernel,18432,seed,6000,0.)
   observed=[]
   for order,record in enumerate(ck['records']):
    assert record['step']==order+1;lr=.001*min(1,(order+1)/100)*(.1+.9*.5*(1+math.cos(math.pi*order/6000)));assert abs(record['lr']-lr)<1e-16
    rng=np.random.default_rng(seed*1000003+order);index=protocol['train_scene_indices'][int(rng.integers(len(protocol['train_scene_indices'])))];view=int(rng.integers(3));assert (record['scene'],record['reference'])==(index,view)
    observed.append((index,view,record['lr']))
   exposure=hashlib.sha256(json.dumps(observed).encode()).hexdigest();protocols[kernel]=protocol;records[kernel]=exposure
   a=read(oracle/'evaluation.json');b=read(chain/'evaluation.json');assert a['failure_denominator']==b['full_scene_failure_denominator']==144 and len(b['scenes'])==144;views=[v for s in b['scenes'] for v in s['views']];assert len(views)==432
   summary=dict(oracle_mean_view_rmse_mm=a['summary']['rmse_mm'],chain_overlap_mean_view_rmse_mm=float(np.mean([v['rmse_mm_on_overlap'] for v in views if v['rmse_mm_on_overlap'] is not None])),chain_overlap_nonempty_views=sum(v['rmse_mm_on_overlap'] is not None for v in views),gt_depth_coverage=b['summary']['gt_depth_coverage'],fscore=b['summary']['pointcloud_fscore'],precision={str(t):float(np.mean([s['pointcloud']['precision'][str(t)] for s in b['scenes']])) for t in (1,2,5)},recall={str(t):float(np.mean([s['pointcloud']['recall'][str(t)] for s in b['scenes']])) for t in (1,2,5)})
   surface_row=surface.get(str(chain));cross_row=cross.get(str(chain));summary['surface']=surface_row['summary'] if surface_row else None;summary['crossview']=cross_row['summary'] if cross_row else None
   if surface_row is None or cross_row is None:seed_complete=False
   result['arms'][key]=dict(status='complete_depth_and_chain' if surface_row is None or cross_row is None else 'complete_with_surface_crossview',summary=summary,checkpoint_sha256=digest(required[1]),oracle_sha256=digest(oracle/'evaluation.json'),chain_sha256=digest(chain/'evaluation.json'),actual_LR_and_scene_reference_sequence_verified=True,scene_reference_exposure_sha256=exposure,ray_sequence_note='Per-rayxy notlogged; ray samplerstatelesssource/seed/cachefixed, scene/reference sequence independentlyreplayed')
  if len(protocols)==2:
   ignore={'neck_kernel','neck_hidden','began'};assert {k:v for k,v in protocols[1].items() if k not in ignore}=={k:v for k,v in protocols[3].items() if k not in ignore};assert records[1]==records[3]
  if seed_complete and len(protocols)==2:paired.append(seed)
 result['comparable_complete_seeds']=paired
 if paired==[17,29,43]:
  result['all_three_pairs_comparable']=True;means={}
  for kernel in (1,3):
   rows=[result['arms'][f'{seed}/{kernel}']['summary'] for seed in paired];m={k:float(np.mean([r[k] for r in rows])) for k in ['oracle_mean_view_rmse_mm','chain_overlap_mean_view_rmse_mm','gt_depth_coverage']}
   for kind in ['fscore','precision','recall']:m[kind]={str(t):float(np.mean([r[kind][str(t)] for r in rows])) for t in (1,2,5)}
   m['surface']={k:float(np.mean([r['surface'][k] for r in rows])) for k in ['gradient_mae_mm_on_covered_pairs','false_step_fraction_on_covered_smooth','normal_angle_mean_deg_on_covered','adjacent_pair_coverage','normal_coverage']};m['crossview']={k:float(np.mean([r['crossview'][k] for r in rows])) for k in ['mean_absolute_consistency_mm_on_covered','rmse_consistency_mm_on_covered','prediction_coverage']};means[str(kernel)]=m
  result['equal_seed_mean']=means
 for p in [OUT/'spatial_depth_pair_report_v2.json',ROOT/'docs/experiments/calibrated_mouse_next12h_spatial_depth_20261025_v2.json']:p.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 print(json.dumps(result))
if __name__=='__main__':main()
