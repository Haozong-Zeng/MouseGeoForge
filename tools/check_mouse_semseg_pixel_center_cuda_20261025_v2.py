"""Bounded engineering diagnosis of v1 CUDA smoke; no formal scientific training."""
from pathlib import Path
import datetime as dt, json, os, subprocess, sys, time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from mousegeoforge.semseg_training import digest
from check_mouse_semseg_pixel_center_20261025 import differences
import torch
OUT=ROOT/'output/calibrated_mouse_resume8h_20261025';BASE=OUT/'pixel_center_cuda_diagnostic_v2'
ENGINE='tools/train_mouse_semseg_pixel_center_20261025.py'
PARENT=ROOT/'output/calibrated_mouse_10h_20261025/semseg_campaign_s17_v1/M_018000/step_018000.pt'
def main():
 assert not BASE.exists();BASE.mkdir()
 reg=json.loads((OUT/'pixel_center_cuda_diagnostic_registration_v2.json').read_text())
 for f,h in reg['source_sha256'].items():assert digest(ROOT/f)==h,f
 results={}
 def launch(name,precision='fp32',stop=None,resume=None,strict=False):
  args=[ENGINE,'smoke','--device','cuda','--precision',precision,'--side','512','--global-batch','4','--micro-batch','1','--workers','2','--steps','4','--save-every','2','--eval-every','500','--arm','M','--label-policy','center_exact','--run-dir',str(BASE/name),'--items',str(ROOT/'output/calibrated_mouse_10h_20261025/semseg_items.json'),'--parent',str(PARENT)]
  if stop:args+=['--stop-after',str(stop)]
  if resume:args+=['--resume',str(resume)]
  command=[sys.executable,*args]
  if strict:command=[sys.executable,'-c',"import runpy,sys,torch;torch.use_deterministic_algorithms(True);sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name='__main__')",*args]
  began=time.monotonic()
  with (BASE/(name+'.log')).open('w') as log:r=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=180,env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','CUBLAS_WORKSPACE_CONFIG':':4096:8'})
  results[name]=dict(returncode=r.returncode,seconds=time.monotonic()-began,command=command,log_sha256=digest(BASE/(name+'.log')))
  if strict:return
  assert r.returncode==0,(name,r.returncode)
  receipt=json.loads((BASE/name/'completion.json').read_text());assert receipt['passed'] and receipt['frozen_unchanged'] and receipt['feature_receipt']['actual_weights']
  return BASE/name/f'step_{receipt["segment_endpoint"]:06d}.pt'
 launch('strict_determinism_probe',strict=True)
 repeat=launch('bf16_repeat_full4',precision='bf16')
 full=launch('fp32_full4');part=launch('fp32_part2',stop=2);resumed=launch('fp32_resumed4',resume=part);repeat32=launch('fp32_repeat_full4')
 def compare(name,left,right):
  a=torch.load(left,map_location='cpu',weights_only=False);b=torch.load(right,map_location='cpu',weights_only=False)
  maxima={k:max(float((a[k][n].double()-b[k][n].double()).abs().max()) for n in a[k]) for k in ['model','ema','buffers']}
  try:checks=differences(a,b);passed=True;error=None
  except AssertionError as e:checks=None;passed=False;error=str(e)
  results[name]=dict(passed=passed,max_abs=maxima,strict_original_tolerances=checks,error=error,left_sha256=digest(left),right_sha256=digest(right),same_exposure=a['metrics'][-1]['local_ids']==b['metrics'][-1]['local_ids'])
 compare('bf16_repeatability',OUT/'pixel_center_pair_v1/actual_GPU_smoke/center_full4/step_000004.pt',repeat)
 compare('fp32_resume',full,resumed);compare('fp32_repeatability',full,repeat32)
 result=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),engineering_only=True,formal_updates=0,registration_sha256=digest(OUT/'pixel_center_cuda_diagnostic_registration_v2.json'),results=results,limits='Short actual-weight engineering controls only; no tolerance relaxation or original method mutation')
 (BASE/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in results.items() if 'passed' in v}))
if __name__=='__main__':main()
