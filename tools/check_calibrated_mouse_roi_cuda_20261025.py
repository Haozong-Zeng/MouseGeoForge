"""Actual CUDA forward/gradient identity and repeated-backward numerical control."""
import pathlib,json,sys,datetime
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from mousegeoforge.models.calibrated_mouse_depth import CalibratedRayDepth
from mousegeoforge.models.calibrated_mouse_depth_roi import CalibratedRayDepthROI
from mousegeoforge.semseg_training import seed_all,digest
from calibrated_mouse_data_20261025 import gpu_lock
OLD=ROOT/'output/calibrated_mouse_10h_20261025';OUT=ROOT/'output/calibrated_mouse_resume8h_20261025'
def main():
 torch.set_num_threads(4);seed_all(17);rows=[]
 with gpu_lock():
  saved=torch.load(OLD/'depth_D1_6000_s17_v2/step_006000.pt',map_location='cuda',weights_only=False);a=CalibratedRayDepth().cuda();b=CalibratedRayDepthROI().cuda();a.load_state_dict(saved['ema']);b.load_state_dict(saved['ema']);bounds=torch.tensor(saved['protocol']['capture_aabb_mm'],device='cuda',dtype=torch.float32)
  for index in [123,460,515]:
   item=torch.load(OLD/'depth_features'/f'{index:06d}.pt',map_location='cpu',weights_only=False);assert item['scene']['split']=='train';f=item['features'].cuda().float();k=item['K'].cuda().float();e=item['E'].cuda().float();v=1;y,x=np.where(item['body'][v].numpy());xy=torch.tensor(np.stack([x,y],-1)[::max(1,len(x)//64)][:64],device='cuda').float()
   def run(model,roi):
    model.zero_grad(set_to_none=True);kwargs={'feature_affines':torch.eye(3,device='cuda').repeat(3,1,1)} if roi else {};z,_,_=model(model.encode(f),xy,v,k,e,bounds,**kwargs);z.mean().backward();return z.detach().clone(),{n:p.grad.detach().clone() for n,p in model.named_parameters()}
   za,ga=run(a,False);zr,gr=run(a,False);zb,gb=run(b,True)
   norm=max(float(x.abs().max()) for x in ga.values());diff=lambda first,second:max(float((first[n]-second[n]).abs().max()) for n in first)
   row={'scene':index,'forward_max_difference_mm':float((za-zb).abs().max()),'same_original_repeat_forward_mm':float((za-zr).abs().max()),'gradient_relative_linf':diff(ga,gb)/max(1e-12,norm),'same_original_repeat_gradient_relative_linf':diff(ga,gr)/max(1e-12,norm)};rows.append(row)
 passed=all(x['forward_max_difference_mm']<=1e-4 and x['gradient_relative_linf']<=1e-4 for x in rows)
 result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':passed,'rows':rows,'forward_tolerance_mm':1e-4,'gradient_relative_linf_tolerance':1e-4,'source_sha256':digest(pathlib.Path(__file__)),'long_training_identity':'Separate FULL6000 vs parentD1 NOT exact; compare fresh matched FULL/ROI pairs, do not reuse original D1 as bitwise paired baseline'}
 p=OUT/'roi_cuda_identity_control_v1.json';assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));assert passed
if __name__=='__main__':main()
