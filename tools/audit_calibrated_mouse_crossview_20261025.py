"""CPU directed cross-view depth consistency on GT co-visible native rays."""
from pathlib import Path
import argparse,json,sys,datetime
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from mousegeoforge.semseg_training import digest
OUT=ROOT/'output/calibrated_mouse_resume8h_20261025';OLD=ROOT/'output/calibrated_mouse_10h_20261025'
def world(xy,z,k,e):
 rays=np.c_[xy,np.ones(len(xy))]@np.linalg.inv(k).T;return (rays*z[:,None]-e[:3,3])@e[:3,:3]
def project(points,k,e):
 camera=points@e[:3,:3].T+e[:3,3];uvh=camera@k.T;safe=np.where(np.abs(uvh[:,2])>1e-12,uvh[:,2],1);return uvh[:,:2]/safe[:,None],camera[:,2]
def bilinear(values,valid,uv):
 h,w=values.shape;inside=np.isfinite(uv).all(-1)&(uv[:,0]>=0)&(uv[:,0]<w-1)&(uv[:,1]>=0)&(uv[:,1]<h-1);safe=np.where(np.isfinite(uv),uv,0);x=np.clip(np.floor(safe[:,0]),0,w-2).astype(int);y=np.clip(np.floor(safe[:,1]),0,h-2).astype(int);dx=safe[:,0]-x;dy=safe[:,1]-y
 samples=[values[y,x],values[y,x+1],values[y+1,x],values[y+1,x+1]];ok=inside&valid[y,x]&valid[y,x+1]&valid[y+1,x]&valid[y+1,x+1]&np.isfinite(np.stack(samples)).all(0)&(np.stack(samples)>0).all(0)
 result=samples[0]*(1-dx)*(1-dy)+samples[1]*dx*(1-dy)+samples[2]*(1-dx)*dy+samples[3]*dx*dy
 return result,ok
def direction(gt_ref,body_ref,gt_src,body_src,pred_ref,mask_ref,pred_src,mask_src,kr,er,ks,es):
 body=body_ref&np.isfinite(gt_ref)&(gt_ref>0);y,x=np.where(body);xy=np.stack([x,y],-1);gt_uv,gt_z=project(world(xy,gt_ref[y,x],kr,er),ks,es);sample,inside=bilinear(gt_src,body_src,gt_uv);eligible=(gt_z>0)&inside&(np.abs(gt_z-sample)<=1.)
 count=int(eligible.sum());selected=np.where(eligible&mask_ref[y,x]&np.isfinite(pred_ref[y,x])&(pred_ref[y,x]>0))[0]
 uv,z=project(world(xy[selected],pred_ref[y[selected],x[selected]],kr,er),ks,es);source,ok=bilinear(pred_src,mask_src,uv);ok &= z>0;residual=z[ok]-source[ok];absolute=np.abs(residual)
 result={'candidate_GT_reference_pixels':len(x),'GT_covisible_pixels':count,'covered_predictions':len(absolute),'prediction_coverage':len(absolute)/max(1,count),'absolute_error_sum_mm':float(absolute.sum()),'squared_error_sum_mm2':float((residual**2).sum()),'mean_absolute_consistency_mm_on_covered':float(absolute.mean()) if len(absolute) else None,'rmse_consistency_mm_on_covered':float(np.sqrt(np.mean(residual**2))) if len(absolute) else None,'p95_consistency_mm_on_covered':float(np.percentile(absolute,95)) if len(absolute) else None,'GT_visibility_closure_error_sum_mm':float(np.abs(gt_z[eligible]-sample[eligible]).sum())}
 for t in [1,2,5]:result[f'consistent_le{t}mm_pixels']=int((absolute<=t).sum())
 return result
def aggregate(rows):
 keys=['candidate_GT_reference_pixels','GT_covisible_pixels','covered_predictions','absolute_error_sum_mm','squared_error_sum_mm2','GT_visibility_closure_error_sum_mm',*[f'consistent_le{t}mm_pixels' for t in [1,2,5]]];s={k:sum(r[k] for r in rows) for k in keys};s.update(directed_view_pairs=len(rows),prediction_coverage=s['covered_predictions']/max(1,s['GT_covisible_pixels']),mean_absolute_consistency_mm_on_covered=s['absolute_error_sum_mm']/max(1,s['covered_predictions']),rmse_consistency_mm_on_covered=float(np.sqrt(s['squared_error_sum_mm2']/max(1,s['covered_predictions']))),GT_mean_visibility_closure_error_mm=s['GT_visibility_closure_error_sum_mm']/max(1,s['GT_covisible_pixels']))
 for t in [1,2,5]:s[f'fraction_consistent_le{t}mm_over_GT_covisible']=s[f'consistent_le{t}mm_pixels']/max(1,s['GT_covisible_pixels'])
 return s
def main():
 p=argparse.ArgumentParser();p.add_argument('--pipeline',action='append');p.add_argument('--max-scenes',type=int);p.add_argument('--output',type=Path,required=True);args=p.parse_args();assert not args.output.exists();manifest=json.loads((OLD/'data_manifest.json').read_text())['scenes'];indices=[i for i,s in enumerate(manifest) if s['split']=='dev']
 if args.max_scenes:indices=[indices[int(i)] for i in np.linspace(0,len(indices)-1,args.max_scenes)]
 names=args.pipeline or [f'roi_pipeline_s{s}_{m}_v1' for s in [17,29,43] for m in ['full','roi']]
 result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':digest(Path(__file__)),'manifest_sha256':digest(OLD/'data_manifest.json'),'scene_indices':indices,'pipelines':{},'rules':{'GT_covisibility_depth_tolerance_mm':1,'bilinear_support':'all4sourcepixels validbody andpositivefinite depth; no boundary mixing','reference_support':'all native GTbody rays, no subsampling','coverage_denominator':'GTcovisible rays, includes all predicted misses and out-of-frame projections','prediction_consistency':'project predicted reference 3D into source, compare projected cameraZ with source predicted depth, no fitting','consistency_cutoffs_mm':[1,2,5]},'limits':['Consistency is not accuracy: a shared wrong plane can be perfectly consistent','GT co-visibility is diagnostic selection, not an inference input','Synthetic opaque skin excludes fur/glass/refraction, six correlated poses','No threshold/checkpoint/model selection or scale alignment']}
 for name in names:
  directory=OUT/name;rows=[];receipts=[]
  for index in indices:
   scene=manifest[index];gt=[];pred=[]
   for v,view in enumerate(scene['views']):
    p=ROOT/view['truth'];assert digest(p)==view['truth_sha256']
    with np.load(p) as z:gt.append({k:z[k].copy() for k in ['depth','body','K','E']})
    p=directory/f'scene{index:06d}/view{v}.npz'
    with np.load(p) as z:pred.append({k:z[k].copy() for k in ['depth_mm','mask','K','E']})
    for key in ['K','E']:np.testing.assert_allclose(pred[-1][key],gt[-1][key],atol=1e-4,rtol=0)
    receipts.append({'scene_index':index,'view':v,'prediction_sha256':digest(p)})
   for ref in range(3):
    for src in range(3):
     if ref==src:continue
     g,h,a,b=gt[ref],gt[src],pred[ref],pred[src];m=direction(g['depth'],g['body'],h['depth'],h['body'],a['depth_mm'],a['mask'],b['depth_mm'],b['mask'],g['K'],g['E'],h['K'],h['E']);rows.append(dict(scene=scene['id'],index=index,source=scene['source'],rig=scene['rig'],condition=scene['condition'],reference=ref,source_view=src,**m))
  result['pipelines'][name]={'summary':aggregate(rows),'by_source':{s:aggregate([x for x in rows if x['source']==s]) for s in ['climb','moseq']},'directions':rows,'prediction_receipts':receipts,'evaluation_sha256':digest(directory/'evaluation.json')};print('CROSSVIEW',name,json.dumps(result['pipelines'][name]['summary']),flush=True)
 args.output.write_text(json.dumps(result,indent=2)+'\n');print('CROSSVIEW_COMPLETE',args.output,flush=True)
if __name__=='__main__':main()
