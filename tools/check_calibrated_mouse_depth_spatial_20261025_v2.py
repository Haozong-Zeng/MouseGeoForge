"""Bounded actual-released-cache identity/gradient/restore checks; no scientific result."""
from pathlib import Path
import argparse
import contextlib
import copy
import datetime as dt
import io
import json
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
import numpy as np
import torch
from mousegeoforge.models.calibrated_mouse_depth_roi import CalibratedRayDepthROI,supervised_loss
from mousegeoforge.models.calibrated_mouse_depth_spatial_20261025_v2 import CalibratedRayDepthSpatialV2 as CalibratedRayDepthSpatial
from mousegeoforge.semseg_training import seed_all,digest,write_json
from train_calibrated_mouse_depth_spatial_20261025_v2 import registered_lr
from calibrated_mouse_data_20261025 import gpu_lock
OUT=ROOT/'output/calibrated_mouse_next12h_20261025';OLD=ROOT/'output/calibrated_mouse_10h_20261025';CACHE=ROOT/'output/calibrated_mouse_resume8h_20261025/roi_cache_v1'
def before_cutoff():
 plan=json.loads((OUT/'plan.json').read_text());assert dt.datetime.now(dt.timezone.utc)<dt.datetime.fromisoformat(plan['optimization_cutoff_utc']),'Original optimization cutoff'
def compare(a,b,exact):
 maxima={}
 def visit(x,y,name):
  if isinstance(x,torch.Tensor):
   assert x.shape==y.shape and x.dtype==y.dtype,name
   x=x.detach().cpu();y=y.detach().cpu()
   d=float((x.detach().double()-y.detach().double()).abs().max()) if x.numel() else 0;maxima[name]=d
   if exact or not x.is_floating_point() or name.endswith('.step'):assert torch.equal(x,y),name
   else:torch.testing.assert_close(x,y,atol=1e-5 if name.startswith('optimizer') else 1e-6,rtol=1e-4 if name.startswith('optimizer') else 1e-5,msg=name)
  elif isinstance(x,dict):
   assert x.keys()==y.keys(),name
   for k in x:visit(x[k],y[k],name+'.'+str(k))
  elif isinstance(x,(list,tuple)):
   assert len(x)==len(y),name
   for i,(u,v) in enumerate(zip(x,y)):visit(u,v,name+'.'+str(i))
  else:assert x==y,name
 for key in ['model','ema','optimizer']:visit(a[key],b[key],key)
 return {key:max((v for n,v in maxima.items() if n.startswith(key)),default=0) for key in ['model','ema','optimizer']}
def run(device,output):
 before_cutoff();assert not output.exists();output.mkdir(parents=True);torch.set_num_threads(4)
 manifest=json.loads((OLD/'data_manifest.json').read_text());index=next(i for i,s in enumerate(manifest['scenes']) if s['split']=='train');file=CACHE/f'{index:06d}.pt';receipt=json.loads((CACHE/'cache_receipt.json').read_text());entry=next(r for r in receipt['files'] if (ROOT/r['path']).resolve()==file.resolve());assert digest(file)==entry['sha256']
 item=torch.load(file,map_location='cpu',weights_only=False);features=item['features'].to(device).float();k=item['K'].to(device).float();e=item['E'].to(device).float();affine=item['feature_affines'].to(device).float();bounds=torch.tensor(manifest['capture_aabb_mm'],device=device,dtype=torch.float32)
 view=0;y,x=np.where(item['body'][view].numpy());pick=np.linspace(0,len(x)-1,16).astype(int);xy=torch.tensor(np.stack([x[pick],y[pick]],-1),device=device,dtype=torch.float32);target=item['depth'][view][y[pick],x[pick]].to(device).float();feature_before=features.clone();assert not any(v.requires_grad for v in [features,k,e,affine,bounds])
 results={};initial_states={}
 lock=gpu_lock() if device=='cuda' else contextlib.nullcontext()
 with lock:
  for kernel in (1,3):
   seed_all(17);model=CalibratedRayDepthSpatial(neck_kernel=kernel).to(device);ema=copy.deepcopy(model);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01,foreach=False)
   seed_all(17);base=CalibratedRayDepthROI().to(device)
   for name,value in base.state_dict().items():assert torch.equal(value,model.state_dict()[name])
   initial_states[kernel]={name:value.detach().cpu().clone() for name,value in base.state_dict().items()}
   with torch.no_grad():
    encoded=model.encode(features);torch.testing.assert_close(encoded,base.encode(features),atol=0,rtol=0)
    a=base(base.encode(features),xy,view,k,e,bounds,feature_affines=affine)[0];b=model(encoded,xy,view,k,e,bounds,feature_affines=affine)[0];torch.testing.assert_close(a,b,atol=0,rtol=0)
   gradients=[]
   def update(candidate,opt,average,step):
    before_cutoff();opt.zero_grad(set_to_none=True);prediction,_,valid=candidate(candidate.encode(features),xy,view,k,e,bounds,feature_affines=affine);loss=supervised_loss(prediction,target,valid);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(candidate.parameters(),10.,error_if_nonfinite=True)
    for group in opt.param_groups:group['lr']=registered_lr(step,6000)
    assert float(candidate.neck[2].weight.grad.norm())>0
    assert candidate.score[4].bias.grad is None and not candidate.score[4].bias.requires_grad
    if step:assert float(candidate.neck[0].weight.grad.norm())>0
    gradients.append(dict(step=step+1,first_neck_norm=float(candidate.neck[0].weight.grad.norm()),last_neck_norm=float(candidate.neck[2].weight.grad.norm()),lr=opt.param_groups[0]['lr']))
    opt.step()
    with torch.no_grad():
     for p,q in zip(candidate.parameters(),average.parameters()):q.lerp_(p,.005)
   snapshot=None
   for step in range(4):
    update(model,optimizer,ema,step)
    if step==1:
     stream=io.BytesIO();torch.save(dict(model=model.state_dict(),ema=ema.state_dict(),optimizer=optimizer.state_dict()),stream);stream.seek(0);snapshot=torch.load(stream,map_location=device,weights_only=False)
   restored=CalibratedRayDepthSpatial(neck_kernel=kernel).to(device);restored.load_state_dict(snapshot['model']);restored_ema=copy.deepcopy(restored);restored_ema.load_state_dict(snapshot['ema']);restored_opt=torch.optim.AdamW(restored.parameters(),lr=.001,weight_decay=.01,foreach=False);restored_opt.load_state_dict(snapshot['optimizer'])
   for step in (2,3):update(restored,restored_opt,restored_ema,step)
   differences=compare(dict(model=model.state_dict(),ema=ema.state_dict(),optimizer=optimizer.state_dict()),dict(model=restored.state_dict(),ema=restored_ema.state_dict(),optimizer=restored_opt.state_dict()),exact=device=='cpu')
   results[str(kernel)]=dict(neck_parameters=sum(p.numel() for p in model.neck.parameters()),base_initial_forward_exact=True,gradient_receipts=gradients,uninterrupted4_vs_2plus2_max_abs=differences)
  for name,value in initial_states[1].items():assert torch.equal(value,initial_states[3][name])
 assert torch.equal(features,feature_before) and digest(file)==entry['sha256']
 result=dict(passed=True,utc=dt.datetime.now(dt.timezone.utc).isoformat(),engineering_only=True,score_bias_trainable=False,device=device,actual_released_cached_features=True,online_encoder=False,TRAIN_scene_index=index,actual_cache_file_sha256=digest(file),roi_cache_receipt_sha256=digest(CACHE/'cache_receipt.json'),base_initial_weights_identical=True,feature_values_and_file_unchanged=True,results=results,CPU_exact=device=='cpu',GPU_tolerance='model/EMA atol1e-6 rtol1e-5;optimizeratol1e-5rtol1e-4;countersexact',source_sha256={f:digest(ROOT/f) for f in ['mousegeoforge/models/calibrated_mouse_depth_spatial_20261025_v2.py','mousegeoforge/models/calibrated_mouse_depth_spatial_20261025.py','tools/train_calibrated_mouse_depth_spatial_20261025_v2.py','tools/check_calibrated_mouse_depth_spatial_20261025_v2.py']})
 write_json(output/'receipt.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','cuda'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.device,a.output)
