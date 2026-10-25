"""Report fixed shared endpoint, actual LRs/exposures and native original-label audits."""
from pathlib import Path
import datetime as dt,json,math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from mousegeoforge.semseg_training import digest,load_items,BalancedSchedule
OUT=ROOT/'output/calibrated_mouse_next12h_20261025';BASE=OUT/'semseg_adapter_pair_v1'
def main():
 reg=json.loads((OUT/'registrations/semseg_adapter_pair_v1.json').read_text());result=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),registration_sha256=digest(OUT/'registrations/semseg_adapter_pair_v1.json'),comparable=False,arms={},limitations=reg['limitations'])
 cf=BASE/'completion.json'
 if cf.exists():
  campaign=json.loads(cf.read_text());result['campaign']=campaign;endpoint=campaign['full_paired_endpoint'];rows=load_items(OUT/'semseg_roi_coarse_cache_v1/items.json',ROOT);schedule=BalancedSchedule(rows,4,17);protocols={};exposures={}
  for policy in ['control','adapter']:
   folder=BASE/f'{policy}_{endpoint:06d}';ev=BASE/(policy+'_dev/evaluation.json');audit=OUT/f'semseg_adapter_native_{policy}_v1.json'
   if not endpoint or not all(p.exists() for p in [folder/f'step_{endpoint:06d}.pt',ev,audit]):result['arms'][policy]={'status':'missing_or_partial'};continue
   ck=torch.load(folder/f'step_{endpoint:06d}.pt',map_location='cpu',weights_only=False);p=ck['protocol'];assert ck['step']==endpoint and len(ck['metrics'])==endpoint and p['parent_sha256']==reg['parent_sha256'] and p['head_phase_exposure_offset']==24000 and p['adapter_mode']==policy and p['adapter_parameters']==262144
   assert all(not enabled for name,enabled in ck['requires_grad'].items() if name.startswith('encoder.'));observed=[]
   for step,record in enumerate(ck['metrics']):
    ids=[rows[i]['id'] for i in schedule.at(step+24000)];assert record['step']==step+1 and record['local_ids']==record['global_ids']==ids
    expected={'head':3e-6};expected.update({'adapters':1e-5} if policy=='adapter' else {});assert record['lrs'].keys()==expected.keys()
    for owner,peak in expected.items():assert abs(record['lrs'][owner]-peak*min(1,(step+1)/300)*(.1+.9*.5*(1+math.cos(math.pi*step/3000))))<1e-18
    assert math.isfinite(record['loss']) and math.isfinite(record['grad_norm']);observed.append(ids)
   a=json.loads(audit.read_text());evaluation=json.loads(ev.read_text());assert len(evaluation['ema'])==879 and evaluation['checkpoint_sha256']==digest(folder/f'step_{endpoint:06d}.pt');assert a['denominator']==879 and a['checkpoint_sha256']==evaluation['checkpoint_sha256'] and a['evaluation_receipt_sha256']==digest(ev);assert {r['id'] for r in a['rows']}=={r['id'] for r in evaluation['ema']};assert all(r['native_original']==r['evaluation_saved_native'] for r in a['rows'])
   groups={}
   for key in ['all','real/True','real/False','synthetic/True','synthetic/False']:
    rs=[r for r in evaluation['ema'] if key=='all' or f"{r['domain']}/{r['bottom']}"==key];groups[key]=dict(images=len(rs),**{m:sum(r['native'][m] for r in rs)/len(rs) for m in ['iou','dice','boundary_f1']})
   result['arms'][policy]=dict(status='complete',groups=groups,actual_all_LRs_and_exposures_verified=True,checkpoint_sha256=digest(folder/f'step_{endpoint:06d}.pt'),native_audit_sha256=digest(audit),encoder_frozen=True);protocols[policy]=p;exposures[policy]=observed
  if len(protocols)==2:
   assert {k:v for k,v in protocols['control'].items() if k!='adapter_mode'}=={k:v for k,v in protocols['adapter'].items() if k!='adapter_mode'};assert exposures['control']==exposures['adapter'];result['comparable']=True;result['full_registered_endpoint_completed']=endpoint==3000 and campaign['status']=='completed';result['adapter_minus_control']={g:{m:result['arms']['adapter']['groups'][g][m]-result['arms']['control']['groups'][g][m] for m in ['iou','dice','boundary_f1']} for g in result['arms']['control']['groups']}
 for file in [OUT/'semseg_adapter_pair_report_v1.json',ROOT/'docs/experiments/calibrated_mouse_next12h_semseg_adapter_20261025.json']:file.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result))
if __name__=='__main__':main()
