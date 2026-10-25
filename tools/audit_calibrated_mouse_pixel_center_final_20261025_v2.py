"""Read-only final execution and original-native probability audit of FP32 pixel pair."""
from pathlib import Path
import datetime as dt,hashlib,json,math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
import torch
from mousegeoforge.semseg_training import BalancedSchedule,load_items,digest
from audit_mouse_semseg_native_probability_20261025_v3 import audit
OUT=ROOT/'output/calibrated_mouse_resume8h_20261025';BASE=OUT/'pixel_center_pair_v2'
def main():
 target=OUT/'pixel_center_final_audit_v2.json';assert not target.exists()
 registration=json.loads((OUT/'pixel_center_final_audit_registration_v2.json').read_text())
 for f,h in registration['source_sha256'].items():assert digest(ROOT/f)==h,f
 formal=json.loads((OUT/'pixel_center_pair_registration_v2.json').read_text())
 for f,h in formal['source_sha256'].items():assert digest(ROOT/f)==h,f
 complete=json.loads((BASE/'completion.json').read_text());assert complete['status']=='completed' and complete['full_paired_endpoint']==6000
 rows=load_items(ROOT/'output/calibrated_mouse_10h_20261025/semseg_items.json',ROOT,verify=False);schedule=BalancedSchedule(rows,4,17)
 data_hash=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest();protocols={};arms={};exposures={}
 for policy in ['legacy','center_exact']:
  folder=BASE/f'{policy}_006000';checkpoint=folder/'step_006000.pt';state=torch.load(checkpoint,map_location='cpu',weights_only=False);protocols[policy]=state['protocol'];protocol=state['protocol']
  assert state['step']==protocol['steps']==6000 and protocol['label_policy']==policy and protocol['precision']=='fp32' and protocol['data_sha256']==data_hash
  assert protocol['parent_sha256']==formal['parent_sha256'] and protocol['global_batch']==4 and protocol['seed']==17
  assert all(n.startswith('head.') for n in state['model']) and all(n.startswith('head.') for n in state['ema'])
  assert all(enabled==n.startswith('head.') for n,enabled in state['requires_grad'].items())
  assert len(state['metrics'])==6000 and all(torch.isfinite(v).all() for v in state['model'].values()) and all(torch.isfinite(v).all() for v in state['ema'].values())
  exposure=[]
  for i,record in enumerate(state['metrics']):
   expected=[rows[j]['id'] for j in schedule.at(i+18000)];assert record['step']==i+1 and record['local_ids']==record['global_ids']==expected
   assert all(rows[j]['split']=='train' for j in schedule.at(i+18000));exposure.extend(expected)
   assert not record['blocks'] and math.isfinite(record['loss']) and math.isfinite(record['grad_norm'])
   expected_lr=1e-5*(.5+.25*(1+math.cos(math.pi*i/6000)))*min(1.,(i+1)/100)
   assert all(math.isclose(lr,expected_lr,rel_tol=1e-12,abs_tol=1e-16) for lr in record['lrs'].values())
  for endpoint in range(500,6001,500):
   segment=BASE/f'{policy}_{endpoint:06d}';receipt=json.loads((segment/'completion.json').read_text());assert receipt['passed'] and receipt['frozen_unchanged'] and not receipt['engineering_only'] and not receipt['fixture'] and receipt['segment_endpoint']==endpoint
   assert json.loads((segment/'protocol.json').read_text())==protocol
  exposures[policy]=hashlib.sha256(json.dumps(exposure).encode()).hexdigest()
  native_path=OUT/f'pixel_center_{policy}_native_probability_audit_v2.json';native=audit(BASE/(policy+'_dev'),native_path)
  assert native['denominator']==879 and native['checkpoint_sha256']==digest(checkpoint)
  arms[policy]=dict(checkpoint_sha256=digest(checkpoint),native_audit_sha256=digest(native_path),native_groups=native['groups'],steps=6000,TRAIN_examples=24000,all_segments_encoder_frozen=True,exact_actual_schedule_and_LR=True,head_parameters_finite=True)
 assert exposures['legacy']==exposures['center_exact'];assert {k:v for k,v in protocols['legacy'].items() if k!='label_policy'}=={k:v for k,v in protocols['center_exact'].items() if k!='label_policy'}
 result=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),passed=True,source_sha256=registration['source_sha256'],registration_sha256=digest(OUT/'pixel_center_pair_registration_v2.json'),campaign_completion_sha256=digest(BASE/'completion.json'),arms=arms,exposure_sha256=exposures,protocol_difference_only_label_policy=True,encoder_unfreeze=False,no_model_promotion=True,limits=['Single exploratory reusedDEVseed17; no independent replication, checkpoint/threshold selection or promotion','Native masks replay exact FP32 probability>.5; original native GT label hashes checked; real manual vs synthetic skin remain distinct','Fixed6000 is an exposure budget, not evidence of TRAIN plateau or permission to unfreeze'])
 target.write_text(json.dumps(result,indent=2)+'\n');(ROOT/'docs/experiments/calibrated_mouse_pixel_center_final_audit_20261025_v2.json').write_bytes(target.read_bytes());print(json.dumps({'passed':True,'groups':{k:v['native_groups'] for k,v in arms.items()}}))
if __name__=='__main__':main()
