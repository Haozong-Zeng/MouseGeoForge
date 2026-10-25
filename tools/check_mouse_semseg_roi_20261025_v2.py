"""Finite engineering checks for the separately versioned label-center engine."""
from pathlib import Path
import argparse,datetime as dt,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from mousegeoforge.semseg_training import digest
OUT=ROOT/'output/calibrated_mouse_next12h_20261025'
ENGINE='tools/train_mouse_semseg_roi_20261025_v2.py'
PARENT=ROOT/'output/calibrated_mouse_10h_20261025/semseg_campaign_s17_v1/M_018000/step_018000.pt'
def differences(a,b,exact=False):
 maxima={}
 def recurse(x,y,path):
  if isinstance(x,torch.Tensor):
   assert x.shape==y.shape and x.dtype==y.dtype,path
   diff=float((x.to(torch.float64)-y.to(torch.float64)).abs().max()) if x.numel() else 0.;maxima[path]=diff
   if exact or not x.is_floating_point() or path.endswith('.step'):assert torch.equal(x,y),path
   else:torch.testing.assert_close(x,y,atol=1e-5 if path.startswith('optimizer') else 1e-6,rtol=1e-4 if path.startswith('optimizer') else 1e-5,msg=path)
  elif isinstance(x,dict):
   assert x.keys()==y.keys(),path
   for k in x:recurse(x[k],y[k],path+'.'+str(k))
  elif isinstance(x,(list,tuple)):
   assert len(x)==len(y),path
   for i,(u,v) in enumerate(zip(x,y)):recurse(u,v,path+'.'+str(i))
  else:assert x==y,path
 for key in ['model','ema','optimizer','requires_grad','buffers']:recurse(a[key],b[key],key)
 assert a['step']==b['step']==4
 return {key:max((v for k,v in maxima.items() if k.startswith(key)),default=0.) for key in ['model','ema','optimizer','buffers']}
def run(device,base):
 assert not (base/'receipt.json').exists();base.mkdir(parents=True,exist_ok=True)
 commands=[]
 def launch(name,policy,stop=None,resume=None,expected_reject=False):
  directory=base/name
  args=[sys.executable,ENGINE,'smoke','--device',device,'--precision','fp32','--side','64' if device=='cpu' else '512','--global-batch','4','--micro-batch','1','--workers','2','--steps','4','--save-every','2','--eval-every','500','--arm','M','--crop-mode',policy,'--label-policy','center_exact','--run-dir',str(directory)]
  if device=='cpu':args+=['--fixture']
  else:args+=['--items',str(OUT/'semseg_roi_coarse_cache_v1/items.json'),'--parent',str(PARENT)]
  if stop:args+=['--stop-after',str(stop)]
  if resume:args+=['--resume',str(resume)]
  began=time.monotonic()
  with (base/(name+'.log')).open('w') as log:result=subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=240,env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
  commands.append(dict(command=args,returncode=result.returncode,seconds=time.monotonic()-began))
  if expected_reject:
   assert result.returncode!=0 and 'Resume protocol mismatch' in (base/(name+'.log')).read_text();return None
  assert result.returncode==0,(name,result.returncode)
  receipt=json.loads((directory/'completion.json').read_text());assert receipt['passed'] and receipt['frozen_unchanged']
  assert all(r['head_nonzero'] and not r['blocks'] for r in receipt['gradient_checks'])
  if device=='cuda':assert receipt['feature_receipt']['actual_weights']
  return directory/f'step_{receipt["segment_endpoint"]:06d}.pt'
 full=launch('roi_full4','roi');part=launch('roi_part2','roi',2);resumed=launch('roi_resumed4','roi',resume=part);legacy=launch('full_full4','full')
 a=torch.load(full,map_location='cpu',weights_only=False);b=torch.load(resumed,map_location='cpu',weights_only=False);comparison=differences(a,b,exact=device=='cpu')
 launch('wrong_crop_rejected','full',resume=part,expected_reject=True)
 receipt=dict(passed=True,utc=dt.datetime.now(dt.timezone.utc).isoformat(),device=device,engineering_only=True,actual_weights=device=='cuda',source_sha256={f:digest(ROOT/f) for f in [ENGINE,'mousegeoforge/semseg_roi_20261025.py','tools/check_mouse_semseg_roi_20261025_v2.py']},resume_4_vs_2_plus_2_max_abs=comparison,CPU_exact=device=='cpu',GPU_tolerance='model/EMA atol1e-6 rtol1e-5; optimizer atol1e-5 rtol1e-4; counters exact',strict_cross_crop_resume_rejected=True,head_gradients_nonzero=True,encoder_frozen=True,commands=commands)
 (base/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--device',choices=['cpu','cuda'],required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();print(json.dumps(run(args.device,args.output)))
