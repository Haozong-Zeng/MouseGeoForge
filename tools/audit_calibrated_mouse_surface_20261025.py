"""CPU native-pixel surface audit; geometry/cameras fixed, no fitting or filtering."""
from pathlib import Path
import argparse,json,datetime,sys
from functools import lru_cache
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from mousegeoforge.semseg_training import digest
OUT=ROOT/'output/calibrated_mouse_resume8h_20261025';OLD=ROOT/'output/calibrated_mouse_10h_20261025'
def surface_metrics(prediction,mask,target,body,k):
 p=np.asarray(prediction,dtype=np.float64);t=np.asarray(target,dtype=np.float64);b=np.asarray(body,dtype=bool)&np.isfinite(t)&(t>0);m=np.asarray(mask,dtype=bool)&np.isfinite(p)&(p>0)
 errors=[];eligible_pairs=covered_pairs=smooth_pairs=covered_smooth=false_steps=0
 for first,second in [((slice(None),slice(None,-1)),(slice(None),slice(1,None))),((slice(None,-1),slice(None)),(slice(1,None),slice(None)))]:
  dz=t[second]-t[first];eligible=b[first]&b[second]&(np.abs(dz)<=20);covered=eligible&m[first]&m[second];eligible_pairs+=int(eligible.sum());covered_pairs+=int(covered.sum());delta=p[second]-p[first]
  errors.extend(np.abs(delta[covered]-dz[covered]).tolist());smooth=eligible&(np.abs(dz)<=1);covered_s=smooth&m[first]&m[second];smooth_pairs+=int(smooth.sum());covered_smooth+=int(covered_s.sum());false_steps+=int((covered_s&(np.abs(delta)>5)).sum())
 # Upper-left, right and down pixels form a fixed local camera-space triangle.
 eligible=b[:-1,:-1]&b[:-1,1:]&b[1:,:-1]&(np.abs(t[:-1,1:]-t[:-1,:-1])<=20)&(np.abs(t[1:,:-1]-t[:-1,:-1])<=20)
 y,x=np.where(eligible);inv=np.linalg.inv(np.asarray(k,dtype=np.float64))
 def normal(z):
  def point(xx,yy):return np.stack([xx,yy,np.ones_like(xx)],-1)@inv.T*z[yy,xx,None]
  center=point(x,y);n=np.cross(point(x+1,y)-center,point(x,y+1)-center);length=np.linalg.norm(n,axis=-1);return n/np.maximum(length[:,None],1e-12),length
 gt,length=normal(t);gt_valid=length>1e-12;normal_eligible=int(gt_valid.sum());pr,plength=normal(p);covered=m[y,x]&m[y,x+1]&m[y+1,x]&(plength>1e-12)&np.isfinite(pr).all(-1)&gt_valid
 angle=np.degrees(np.arccos(np.clip((pr[covered]*gt[covered]).sum(-1),-1,1)));covered_normals=int(covered.sum());penalty_sum=float(angle.sum())+180*(normal_eligible-covered_normals)
 return {'gt_eligible_adjacent_pairs':eligible_pairs,'covered_adjacent_pairs':covered_pairs,'adjacent_pair_coverage':covered_pairs/max(1,eligible_pairs),'gradient_mae_mm_on_covered_pairs':float(np.mean(errors)) if errors else None,'gradient_absolute_error_sum_mm':float(np.sum(errors)),'gt_smooth_pairs':smooth_pairs,'covered_gt_smooth_pairs':covered_smooth,'false_steps_gt_le1_pred_gt5_mm':false_steps,'false_step_fraction_on_covered_smooth':false_steps/max(1,covered_smooth),'gt_eligible_normals':normal_eligible,'covered_normals':covered_normals,'normal_coverage':covered_normals/max(1,normal_eligible),'normal_angle_mean_deg_on_covered':float(angle.mean()) if len(angle) else None,'normal_angle_sum_deg':float(angle.sum()),'normal_angle_mean_deg_missing180':penalty_sum/max(1,normal_eligible),'normal_missing180_sum_deg':penalty_sum}
def aggregate(rows):
 counts=['gt_eligible_adjacent_pairs','covered_adjacent_pairs','gradient_absolute_error_sum_mm','gt_smooth_pairs','covered_gt_smooth_pairs','false_steps_gt_le1_pred_gt5_mm','gt_eligible_normals','covered_normals','normal_angle_sum_deg','normal_missing180_sum_deg'];s={k:sum(x[k] for x in rows) for k in counts}
 s.update(views=len(rows),adjacent_pair_coverage=s['covered_adjacent_pairs']/max(1,s['gt_eligible_adjacent_pairs']),gradient_mae_mm_on_covered_pairs=s['gradient_absolute_error_sum_mm']/max(1,s['covered_adjacent_pairs']),normal_coverage=s['covered_normals']/max(1,s['gt_eligible_normals']),normal_angle_mean_deg_on_covered=s['normal_angle_sum_deg']/max(1,s['covered_normals']),normal_angle_mean_deg_missing180=s['normal_missing180_sum_deg']/max(1,s['gt_eligible_normals']),false_step_fraction_on_covered_smooth=s['false_steps_gt_le1_pred_gt5_mm']/max(1,s['covered_gt_smooth_pairs']))
 return s
@lru_cache(maxsize=432)
def truth(path):
 with np.load(path) as z:return z['depth'].copy(),z['body'].copy(),z['K'].copy()
def main():
 p=argparse.ArgumentParser();p.add_argument('--pipeline',action='append');p.add_argument('--output',type=Path,required=True);args=p.parse_args();assert not args.output.exists()
 manifest=json.loads((OLD/'data_manifest.json').read_text())['scenes'];names=args.pipeline or [f'roi_pipeline_s{s}_{m}_v1' for s in [17,29,43] for m in ['full','roi']]
 result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'schema':'native-calibrated-surface-audit-v1','source_sha256':digest(Path(__file__)),'pipelines':{},'rules':{'GT_adjacent_depth_jump_max_mm':20,'GT_smooth_depth_jump_max_mm':1,'predicted_false_step_min_mm_exclusive':5,'normal':'oriented upper-left/right/down camera-space triangle using original native K','coverage':'all eligible GT pairs/triangles including prediction misses','missing_normal_penalty_deg':180,'normal_angles_on_covered_also_reported':True,'no_pixel_or_3D_scale_alignment':True},'limits':['Synthetic opaque skin surrogate only, excludes fur and glass','Pixel gradient and triangle normal audit is resolution-dependent, not independent real surface validation','Conditional errors exclude mask/ray misses; coverage and missing180 penalty retain them separately','DEV reused for diagnosis; no threshold/model/checkpoint selection from this audit']}
 for name in names:
  directory=OUT/name;evaluation=json.loads((directory/'evaluation.json').read_text());assert evaluation['full_scene_failure_denominator']==144;byid={s['id']:s for s in evaluation['scenes']};rows=[]
  for index,scene in enumerate(manifest):
   if scene['split']!='dev':continue
   assert scene['id'] in byid
   for view,v in enumerate(scene['views']):
    target_path=ROOT/v['truth'];assert digest(target_path)==v['truth_sha256'];t,b,k=truth(str(target_path));pred_path=directory/f'scene{index:06d}/view{view}.npz'
    with np.load(pred_path) as z:
     np.testing.assert_allclose(z['K'],k,atol=1e-4,rtol=0);metrics=surface_metrics(z['depth_mm'],z['mask'],t,b,k)
    rows.append(dict(scene=scene['id'],index=index,source=scene['source'],rig=scene['rig'],condition=scene['condition'],view=view,bottom=view==0,prediction_sha256=digest(pred_path),**metrics))
  assert len(rows)==432
  result['pipelines'][name]={'evaluation_receipt_sha256':digest(directory/'evaluation.json'),'summary':aggregate(rows),'by_source_view':{f'{source}/{v}':aggregate([x for x in rows if x['source']==source and x['view']==v]) for source in ['climb','moseq'] for v in range(3)},'views':rows}
  print('SURFACE_AUDIT',name,json.dumps(result['pipelines'][name]['summary']),flush=True)
 args.output.write_text(json.dumps(result,indent=2)+'\n');print('SURFACE_COMPLETE',args.output,flush=True)
if __name__=='__main__':main()
