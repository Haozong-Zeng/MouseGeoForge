"""Known-camera RGB96plane and mask-hull finite baselines; no learned weights/GT forward."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from mousegeoforge.models.calibrated_mouse_depth import ray_bounds,project_world
from calibrated_mouse_data_20261025 import points
from train_calibrated_mouse_depth_roi_20261025 import cloud_metrics
OLD=ROOT/'output/calibrated_mouse_10h_20261025';PARENT=ROOT/'output/calibrated_mouse_resume8h_20261025';OUT=ROOT/'output/calibrated_mouse_next12h_20261025'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sweep(rgb,masks,k,e,bounds,reference,xy,planes=96):
    """No targets enter forward. xy/native K sample pixel centres; z is camera Z."""
    h,w=rgb.shape[-2:];origin,directions,near,far,valid=ray_bounds(xy,k[reference],e[reference],bounds)
    z=near[:,None]+(far-near)[:,None]*torch.linspace(0,1,planes,device=xy.device,dtype=xy.dtype)
    world=origin+directions[:,None]*z[...,None]
    grid=(2*(xy+.5)/xy.new_tensor([w,h])-1)[None,:,None]
    color=F.grid_sample(rgb[reference:reference+1],grid,align_corners=False)[0,:,:,0].T
    errors=[];visibilities=[];hull=torch.ones_like(z,dtype=torch.bool)
    for source in range(len(rgb)):
        if source==reference:continue
        uv,source_z=project_world(world,k[source],e[source]);g=(2*(uv+.5)/xy.new_tensor([w,h])-1)[None]
        sample=F.grid_sample(rgb[source:source+1],g,align_corners=False)[0].permute(1,2,0)
        inside=(source_z>0)&(uv[...,0]>=0)&(uv[...,0]<=w-1)&(uv[...,1]>=0)&(uv[...,1]<=h-1)
        fg=F.grid_sample(masks[source:source+1,None].float(),g,mode='nearest',align_corners=False)[0,0]>.5
        hull &= inside&fg
        errors.append((sample-color[:,None]).abs().mean(-1)*inside);visibilities.append(inside)
    count=torch.stack(visibilities).sum(0);cost=torch.stack(errors).sum(0)/count.clamp_min(1);cost=torch.where(count>0,cost,torch.inf)
    best,idx=cost.min(-1);photo_valid=valid&torch.isfinite(best)
    photo=torch.where(photo_valid,z.gather(1,idx[:,None])[:,0],0.)
    occupied=valid&hull.any(-1);first=hull.to(torch.int8).argmax(-1);hull_z=torch.where(occupied,z.gather(1,first[:,None])[:,0],0.)
    return dict(rgb=photo,hull=hull_z,photo_valid=photo_valid,hull_valid=occupied,photo_cost=best,photo_exact_tie_planes=(cost==best[:,None]).sum(-1))
def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['smoke','evaluate']);args=parser.parse_args();torch.set_num_threads(4);cv2.setNumThreads(1)
    dest=OUT/('classical_actual_smoke_v1' if args.command=='smoke' else 'classical_baselines_v1')
    if dest.exists():raise FileExistsError(dest)
    dest.mkdir();registration=OUT/'registrations/classical_baselines_v1.json';reg=json.loads(registration.read_text())
    for f,h in reg['source_sha256'].items():assert sha(ROOT/f)==h
    manifest=json.loads((OLD/'data_manifest.json').read_text());dev=[(i,s) for i,s in enumerate(manifest['scenes']) if s['split']=='dev'];assert len(dev)==144
    if args.command=='smoke':dev=dev[:1]
    fixed=PARENT/'roi_pipeline_s17_roi_v1';bounds=torch.tensor(manifest['capture_aabb_mm'],device='cuda',dtype=torch.float32)
    rows=[];inputs={str((OLD/'data_manifest.json').relative_to(ROOT)):sha(OLD/'data_manifest.json')};began=time.monotonic()
    with torch.inference_mode():
      for count,(index,scene) in enumerate(dev,1):
        rgb=[];masks=[];k=[];e=[];target=[];body=[]
        for v,src in enumerate(scene['views']):
            image=ROOT/src['image'];truth=ROOT/src['truth'];pred=fixed/f'scene{index:06d}'/f'view{v}.npz'
            assert sha(image)==src['image_sha256'] and sha(truth)==src['truth_sha256']
            for f in [image,truth,pred]:inputs[str(f.relative_to(ROOT))]=sha(f)
            raw=cv2.cvtColor(cv2.imread(str(image)),cv2.COLOR_BGR2RGB).astype(np.float32)/255
            rgb.append(cv2.blur(raw,(5,5)))
            with np.load(pred,allow_pickle=False) as a:masks.append(a['mask']);k.append(a['K']);e.append(a['E'])
            with np.load(truth,allow_pickle=False) as a:target.append(a['depth']);body.append(a['body'].astype(bool));assert np.array_equal(k[-1],a['K'].astype(np.float32)) and np.array_equal(e[-1],a['E'].astype(np.float32))
        rgb=torch.from_numpy(np.stack(rgb)).permute(0,3,1,2).cuda();mask_tensor=torch.from_numpy(np.stack(masks)).cuda();kt=torch.from_numpy(np.stack(k)).cuda();et=torch.from_numpy(np.stack(e)).cuda()
        result={kind:dict(views=[],cloud=[]) for kind in ['rgb','hull']};gt_cloud=[]
        scene_dir=dest/f'scene{index:06d}';scene_dir.mkdir()
        for v in range(3):
            ys,xs=np.where(masks[v]);xy=torch.from_numpy(np.stack([xs,ys],-1)).cuda().float();values={kind:[] for kind in ['rgb','hull']};ties=[]
            for chunk in xy.split(512):
                a=sweep(rgb,mask_tensor,kt,et,bounds,v,chunk)
                for kind in values:values[kind].append(a[kind].cpu().numpy())
                ties.append(a['photo_exact_tie_planes'].cpu().numpy())
            gt_cloud.append(points(target[v],body[v],k[v],e[v]))
            for kind in values:
                depth=np.zeros_like(target[v]);depth[ys,xs]=np.concatenate(values[kind]) if values[kind] else []
                assert np.isfinite(depth).all();support=masks[v]&(depth>0);overlap=support&body[v];res=depth[overlap]-target[v][overlap]
                result[kind]['views'].append(dict(view=v,bottom=v==0,predicted_mask_pixels=int(masks[v].sum()),valid_depth_pixels=int(support.sum()),gt_foreground_pixels=int(body[v].sum()),gt_depth_coverage=float(overlap.sum()/max(1,body[v].sum())),rmse_mm_on_overlap=float(np.sqrt(np.mean(res.astype(np.float64)**2))) if len(res) else None,missing_as_zero_sentinel_rmse=float(np.sqrt(np.mean((depth[body[v]]-target[v][body[v]]).astype(np.float64)**2)))))
                result[kind]['cloud'].append(points(depth,support,k[v],e[v]));np.savez_compressed(scene_dir/f'{kind}_view{v}.npz',depth_mm=depth,mask=masks[v],K=k[v],E=e[v])
        truth=np.concatenate(gt_cloud)
        for kind in result:result[kind]['pointcloud']=cloud_metrics(np.concatenate(result[kind].pop('cloud')),truth)
        rows.append(dict(index=index,id=scene['id'],source=scene['source'],methods=result))
        if count%12==0 or args.command=='smoke':print('CLASSICAL',count,len(dev),'seconds',round(time.monotonic()-began,1),flush=True)
    summary={}
    for kind in ['rgb','hull']:
        views=[v for s in rows for v in s['methods'][kind]['views']];clouds=[s['methods'][kind]['pointcloud'] for s in rows];errors=[v['rmse_mm_on_overlap'] for v in views if v['rmse_mm_on_overlap'] is not None]
        summary[kind]=dict(scenes=len(rows),views=len(views),gt_depth_coverage=float(np.mean([v['gt_depth_coverage'] for v in views])),overlap_nonempty_views=len(errors),rmse_mm_on_overlap=float(np.mean(errors)) if errors else None,cloud_fscore={str(t):float(np.mean([c['fscore'][str(t)] for c in clouds])) for t in (1,2,5)})
    for f,h in inputs.items():assert sha(ROOT/f)==h
    report=dict(completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),engineering_only=args.command=='smoke',registration_sha256=sha(registration),summary=summary,scenes=rows,input_sha256=inputs,seconds=time.monotonic()-began,limitations=reg['limitations'])
    (dest/'completion.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if args.command=='evaluate':(OUT/'classical_baselines_report_v1.json').write_text(json.dumps({k:v for k,v in report.items() if k not in ['scenes','input_sha256']},indent=2)+'\n')
    print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
