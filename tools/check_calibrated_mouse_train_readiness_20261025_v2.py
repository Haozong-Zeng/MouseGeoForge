"""Prospective TRAIN-only nonregression screen and finalFULL native numerical repeat."""
from pathlib import Path
import argparse,datetime as dt,hashlib,json,math,os,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from mousegeoforge.semseg_training import digest,BalancedSchedule
OUT=ROOT/'output/calibrated_mouse_next12h_20261025';BASE=OUT/'semseg_roi_pair_v2'
GROUPS=['all','real/True','real/False','synthetic/True','synthetic/False'];METRICS=['iou','dice','boundary_f1']

def frozen_model(factory):
 with torch.inference_mode(False):
  return factory().eval().requires_grad_(False)

def parameter_versions(model):
 return {n:(id(p),p._version) for n,p in model.named_parameters()}

def rows_checked(rows,expected):
 assert [r['id'] for r in rows]==expected and len(set(expected))==256
 for r in rows:
  assert r['domain'] in ['real','synthetic'] and isinstance(r['bottom'],bool)
  assert np.isfinite(r['loss']) and r['loss']>=0
  for key in METRICS:assert np.isfinite(r['native'][key]) and 0<=r['native'][key]<=1
 for d in ['real','synthetic']:
  for b in [True,False]:assert sum((r['domain'],r['bottom'])==(d,b) for r in rows)==64

def assess(windows,expected,tolerances):
 assert len(windows)==4
 for rows in windows:rows_checked(rows,expected)
 groups={};failed=[]
 for group in GROUPS:
  slices=[rows if group=='all' else [r for r in rows if f"{r['domain']}/{r['bottom']}"==group] for rows in windows];checks={}
  for metric in METRICS:
   values=[float(np.mean([r['native'][metric] for r in rows])) for rows in slices];t=tolerances[group][metric]['tolerance'];early=float(np.mean(values[:2]));late=float(np.mean(values[2:]));minimum_jump=float(np.min(np.diff(values)));passed=late-early>=-t-1e-12 and values[-1]-values[0]>=-t-1e-12 and minimum_jump>=-2*t-1e-12
   checks[metric]=dict(values=values,tolerance=t,late_minus_early=late-early,final_minus_first=values[-1]-values[0],minimum_consecutive_delta=minimum_jump,passed=passed)
   if not passed:failed.append(group+'/'+metric)
  losses=[float(np.mean([r['loss'] for r in rows])) for rows in slices];early_loss=float(np.mean(losses[:2]));cap=max(2*early_loss,early_loss+.02);passed=max(losses)<=cap+1e-12;checks['loss_blowup']=dict(values=losses,early_mean=early_loss,cap=cap,passed=passed)
  if not passed:failed.append(group+'/loss_blowup')
  groups[group]=checks
 return dict(passed=not failed,failed=failed,groups=groups)

def compare_repeat(reference,actual,expected):
 rows_checked(reference,expected);rows_checked(actual,expected);maxima={m:0. for m in METRICS};lossdiff=0.
 for a,b in zip(reference,actual):
  assert (a['id'],a['domain'],a['bottom'],a['group'])==(b['id'],b['domain'],b['bottom'],b['group'])
  for metric in METRICS:
   delta=abs(a['native'][metric]-b['native'][metric]);maxima[metric]=max(maxima[metric],delta);assert delta<=1e-5,(a['id'],metric,delta)
  delta=abs(a['loss']-b['loss']);lossdiff=max(lossdiff,delta);assert delta<=1e-6+1e-5*abs(a['loss']),(a['id'],'loss',delta)
 return dict(native_max_abs_difference=maxima,loss_max_abs_difference=lossdiff)

def registrations():
 path=OUT/'registrations/semseg_readiness_sources_v2.json';sources=json.loads(path.read_text());design_path=OUT/'registrations/semseg_readiness_design_v1.json';assert digest(design_path)==sources['design_sha256'];design=json.loads(design_path.read_text())
 for p,h in sources['source_sha256'].items():assert digest(ROOT/p)==h
 assert digest(ROOT/design['items_path'])==design['items_sha256']
 for p,h in design['calibration_input_sha256'].items():assert digest(ROOT/p)==h
 return design,sources

def repeat():
 design,sources=registrations();dest=OUT/'semseg_readiness_FULL_GPU_repeat_v2';assert not dest.exists();dest.mkdir();checkpoint=BASE/'full_006000/step_006000.pt';ck=torch.load(checkpoint,map_location='cpu',weights_only=False);assert ck['step']==6000 and ck['protocol']['crop_mode']=='full' and ck['protocol']['precision']=='fp32'
 from train_mouse_semseg_roi_20261025_v2 import parser,make_encoder,MouseSemseg,evaluate_rows,gpu_guard,seed_all
 args=parser().parse_args(['evaluate','--crop-mode','full','--label-policy','center_exact','--precision','fp32','--device','cuda','--items',str(ROOT/design['items_path'])]);torch.set_num_threads(4)
 import cv2
 cv2.setNumThreads(1);seed_all(17);items=json.loads((ROOT/design['items_path']).read_text())['items'];byid={r['id']:r for r in items};rows=[byid[i] for i in design['TRAIN_monitor_ids']];assert all(r['split']=='train' for r in rows)
 for row in rows:
  for k in ['image','label']:assert digest(ROOT/row[k])==row[k+'_sha256']
 with gpu_guard(args,0):
  model=frozen_model(lambda:MouseSemseg(make_encoder(args.checkpoint)).cuda());params=dict(model.named_parameters());buffers=dict(model.named_buffers());assert all(n.startswith('head.') for n in ck['model']) and all(n.startswith('head.') for n in ck['ema'])
  for name,value in ck['ema'].items():params[name].copy_(value)
  for name,value in ck['buffers'].items():buffers[name].copy_(value)
  versions=parameter_versions(model);actual=evaluate_rows(model,rows,args,torch.device('cuda'),'M',None);assert versions==parameter_versions(model)
 reference_path=BASE/'full_006000/train_monitor_006000.json';comparison=compare_repeat(json.loads(reference_path.read_text()),actual,design['TRAIN_monitor_ids']);(dest/'actual_TRAIN_monitor.json').write_text(json.dumps(actual,indent=2)+'\n');receipt=dict(passed=True,utc=dt.datetime.now(dt.timezone.utc).isoformat(),optimization_updates=0,TRAIN_only=True,checkpoint_sha256=digest(checkpoint),reference_monitor_sha256=digest(reference_path),actual_monitor_sha256=digest(dest/'actual_TRAIN_monitor.json'),encoder_release_sha256=digest(args.checkpoint),parameters_unchanged=True,all_parameters_frozen=True,fp32=True,comparison=comparison,design_sha256=sources['design_sha256'],source_sha256=sources['source_sha256']);(dest/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))

def report():
 design,sources=registrations();items=json.loads((ROOT/design['items_path']).read_text())['items'];byid={r['id']:r for r in items};assert all(byid[i]['split']=='train' for i in design['TRAIN_monitor_ids']);result=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),design_sha256=sources['design_sha256'],historical_gate_failures_unchanged=True,optimization_updates=0,no_DEV_reads=True,arms={},FULL_ready_for_trial_design=False,adaptation_job_registered=False,scope=design['gate_meaning'],limits=design['limitations'])
 for arm in ['full','roi']:
  paths=[BASE/f'{arm}_{s:06d}/train_monitor_{s:06d}.json' for s in design['future_window']]
  if not all(p.exists() for p in paths):result['arms'][arm]=dict(status='missing_future_window',missing=[str(p.relative_to(OUT)) for p in paths if not p.exists()]);continue
  assert all(dt.datetime.fromtimestamp(p.stat().st_mtime,dt.timezone.utc)>=dt.datetime.fromisoformat(design['registered_utc']) for p in paths)
  assessment=assess([json.loads(p.read_text()) for p in paths],design['TRAIN_monitor_ids'],design['tolerances'][arm]);checkpoint=BASE/f'{arm}_006000/step_006000.pt'
  if not checkpoint.exists():result['arms'][arm]=dict(status='missing_final_checkpoint',assessment=assessment);continue
  ck=torch.load(checkpoint,map_location='cpu',weights_only=False);assert ck['step']==6000 and len(ck['metrics'])==6000;assert ck['protocol']['crop_mode']==arm and ck['protocol']['roi_warmup']==600 and ck['protocol']['roi_cosine_floor']==.1;assert all(not enabled or n.startswith('head.') for n,enabled in ck['requires_grad'].items());assert all(n.startswith('head.') for n in ck['model'])
  for owner in ['model','ema','buffers']:
   assert all(torch.isfinite(t).all() for t in ck[owner].values())
  schedule=BalancedSchedule(items,4,17)
  for row in ck['metrics']:
   step=row['step']-1;lr=1e-5*min(1,(step+1)/600)*(.1+.9*.5*(1+math.cos(math.pi*step/6000)));assert all(abs(v-lr)<1e-16 for v in row['lrs'].values());expected=[items[i]['id'] for i in schedule.at(step+18000)];assert row['local_ids']==row['global_ids']==expected and all(byid[i]['split']=='train' for i in expected)
  result['arms'][arm]=dict(status='TRAIN_window_passed' if assessment['passed'] else 'TRAIN_window_failed',assessment=assessment,checkpoint_sha256=digest(checkpoint),monitor_sha256={str(p.relative_to(OUT)):digest(p) for p in paths},actual6000LRs_and_TRAIN_exposure_verified=True,finite_checkpoint_frozen_encoder=True)
 repeat_path=OUT/'semseg_readiness_FULL_GPU_repeat_v2/receipt.json'
 if repeat_path.exists():
  value=json.loads(repeat_path.read_text());assert value['passed'] and value['design_sha256']==sources['design_sha256'];assert value['checkpoint_sha256']==digest(BASE/'full_006000/step_006000.pt');result['FULL_numerical_repeat']=dict(status='passed',receipt_sha256=digest(repeat_path),comparison=value['comparison'])
 else:result['FULL_numerical_repeat']=dict(status='missing_or_failed',not_inferred_from_engineering_training_smoke=True)
 result['FULL_ready_for_trial_design']=result['arms'].get('full',{}).get('status')=='TRAIN_window_passed' and result['FULL_numerical_repeat']['status']=='passed'
 result['next']='Separatefinitepairedadaptationprotocol/GPUgradientrestorechecks stillmandatory; noautomaticlaunch' if result['FULL_ready_for_trial_design'] else 'Noadaptationfromthisparent withoutregisteredrequirements; no earlier/ROI/DEV-selectedfallback'
 for p in [OUT/'semseg_readiness_v2.json',ROOT/'docs/experiments/calibrated_mouse_next12h_readiness_20261025_v2.json']:p.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result))

def main():
 p=argparse.ArgumentParser();p.add_argument('--repeat',action='store_true');args=p.parse_args();repeat() if args.repeat else report()
if __name__=='__main__':main()
