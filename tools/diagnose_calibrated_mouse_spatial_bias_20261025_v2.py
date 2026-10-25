"""Finite TRAIN cached-feature GPU diagnosis; original/frozen softmax-null bias, strict tolerances."""
from pathlib import Path
import copy,datetime as dt,io,json,os,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
import numpy as np
import torch
from mousegeoforge.models.calibrated_mouse_depth_spatial_20261025 import CalibratedRayDepthSpatial
from mousegeoforge.models.calibrated_mouse_depth_roi import supervised_loss
from mousegeoforge.semseg_training import seed_all,digest
from train_calibrated_mouse_depth_spatial_20261025 import registered_lr
from calibrated_mouse_data_20261025 import gpu_lock
OUT=ROOT/'output/calibrated_mouse_next12h_20261025';CACHE=ROOT/'output/calibrated_mouse_resume8h_20261025/roi_cache_v1';OLD=ROOT/'output/calibrated_mouse_10h_20261025'

def differences(a,b):
 maximum={};failures=[]
 def walk(x,y,path):
  if isinstance(x,torch.Tensor):
   assert x.shape==y.shape and x.dtype==y.dtype;x=x.detach().cpu();y=y.detach().cpu();delta=float((x.double()-y.double()).abs().max()) if x.numel() else 0.;maximum[path]=delta
   exact=not x.is_floating_point() or path.endswith('.step');atol,rtol=(1e-5,1e-4) if path.startswith('optimizer') else (1e-6,1e-5)
   good=torch.equal(x,y) if exact else torch.allclose(x,y,atol=atol,rtol=rtol)
   if not good:failures.append(dict(path=path,max_abs=delta,exact=exact,atol=atol,rtol=rtol))
  elif isinstance(x,dict):
   assert x.keys()==y.keys()
   for key in x:walk(x[key],y[key],path+'.'+str(key))
  elif isinstance(x,(list,tuple)):
   assert len(x)==len(y)
   for i,(u,v) in enumerate(zip(x,y)):walk(u,v,path+'.'+str(i))
  else:assert x==y
 for key in ['model','ema','optimizer']:walk(a[key],b[key],key)
 return dict(passed=not failures,failures=failures,max_abs_by_tensor=maximum)

def main():
 dest=OUT/'spatial_bias_GPU_diagnostic_v2';assert not dest.exists();dest.mkdir();reg=json.loads((OUT/'registrations/spatial_bias_GPU_diagnostic_v2.json').read_text())
 for f,h in reg['source_sha256'].items():assert digest(ROOT/f)==h
 plan=json.loads((OUT/'plan.json').read_text());cutoff=dt.datetime.fromisoformat(plan['optimization_cutoff_utc']);assert dt.datetime.now(dt.timezone.utc)<cutoff
 torch.set_num_threads(4);manifest=json.loads((OLD/'data_manifest.json').read_text());index=next(i for i,s in enumerate(manifest['scenes']) if s['split']=='train');file=CACHE/f'{index:06d}.pt';receipt=json.loads((CACHE/'cache_receipt.json').read_text());assert digest(file)==next(r['sha256'] for r in receipt['files'] if r['scene_index']==index);item=torch.load(file,map_location='cpu',weights_only=False)
 results={};updates=0
 with gpu_lock():
  features=item['features'].cuda().float();k=item['K'].cuda().float();e=item['E'].cuda().float();affine=item['feature_affines'].cuda().float();bounds=torch.tensor(manifest['capture_aabb_mm'],device='cuda',dtype=torch.float32);y,x=np.where(item['body'][0].numpy());pick=np.linspace(0,len(x)-1,16).astype(int);xy=torch.tensor(np.stack([x[pick],y[pick]],-1),device='cuda',dtype=torch.float32);target=item['depth'][0][y[pick],x[pick]].cuda().float()
  def forward(m):return m(m.encode(features),xy,0,k,e,bounds,feature_affines=affine)
  def factory(kernel,freeze):
   seed_all(17);m=CalibratedRayDepthSpatial(neck_kernel=kernel).cuda();m.score[4].bias.requires_grad_(not freeze);ema=copy.deepcopy(m);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01,foreach=False);return m,ema,opt
  def state(m,ema,opt):return dict(model=m.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict())
  for kernel in (1,3):
   for freeze in (False,True):
    key=f'k{kernel}_bias'+('frozen' if freeze else 'trainable');gradients=[];m,ema,opt=factory(kernel,freeze)
    with torch.no_grad():
     initial=forward(m)[0];m.score[4].bias.add_(1.);shifted=forward(m)[0];invariance=float((initial-shifted).abs().max());m.score[4].bias.sub_(1.)
    # Independent freshfactory, so invariance probe doesnotchange traininginitialstate.
    m,ema,opt=factory(kernel,freeze)
    def update(model,average,optimizer,step):
     nonlocal updates
     assert dt.datetime.now(dt.timezone.utc)<cutoff;optimizer.zero_grad(set_to_none=True);z,_,valid=forward(model);loss=supervised_loss(z,target,valid);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True)
     bias_grad=None if model.score[4].bias.grad is None else float(model.score[4].bias.grad);gradients.append(dict(step=step+1,bias_grad=bias_grad,loss=float(loss.detach())))
     for group in optimizer.param_groups:group['lr']=registered_lr(step,6000)
     optimizer.step();updates+=1
     with torch.no_grad():
      for p,q in zip(model.parameters(),average.parameters()):q.lerp_(p,.005)
    snapshot=None
    for step in range(4):
     update(m,ema,opt,step)
     if step==1:
      stream=io.BytesIO();torch.save(state(m,ema,opt),stream);stream.seek(0);snapshot=torch.load(stream,map_location='cuda',weights_only=False);torch.save(snapshot,dest/(key+'_step2.pt'))
    full=state(m,ema,opt);torch.save(full,dest/(key+'_full4.pt'))
    restored,restored_ema,restored_opt=factory(kernel,freeze);restored.load_state_dict(snapshot['model']);restored_ema.load_state_dict(snapshot['ema']);restored_opt.load_state_dict(snapshot['optimizer'])
    for step in (2,3):update(restored,restored_ema,restored_opt,step)
    resumed=state(restored,restored_ema,restored_opt);torch.save(resumed,dest/(key+'_resumed4.pt'));resume_diff=differences(full,resumed)
    repeated,repeated_ema,repeated_opt=factory(kernel,freeze)
    for step in range(4):update(repeated,repeated_ema,repeated_opt,step)
    repeated_state=state(repeated,repeated_ema,repeated_opt);torch.save(repeated_state,dest/(key+'_repeat4.pt'));repeat_diff=differences(full,repeated_state)
    with torch.no_grad():zfull=forward(m)[0];zresume=forward(restored)[0];zrepeat=forward(repeated)[0]
    results[key]=dict(softmax_common_bias_plus1_forward_max_mm=invariance,resume=resume_diff,independent_repeat=repeat_diff,prediction_resume_max_mm=float((zfull-zresume).abs().max()),prediction_repeat_max_mm=float((zfull-zrepeat).abs().max()),bias_gradients=gradients)
 result=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),engineering_only=True,actual_GPU=True,TRAIN_scene=index,actual_cache_sha256=digest(file),registered_maximum_updates=40,actual_updates=updates,results=results,tolerance='UNCHANGED:modelEMAatol1e-6rtol1e-5;optimizeratol1e-5rtol1e-4;countersexact',no_formal_training=True,no_GT_scale_fit=True,source_sha256=reg['source_sha256']);assert updates==40;(dest/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({key:dict(resume_pass=r['resume']['passed'],repeat_pass=r['independent_repeat']['passed'],resume_failures=r['resume']['failures'],repeat_failures=r['independent_repeat']['failures'],invariance_mm=r['softmax_common_bias_plus1_forward_max_mm'],prediction_resume_mm=r['prediction_resume_max_mm']) for key,r in results.items()}))
if __name__=='__main__':main()
