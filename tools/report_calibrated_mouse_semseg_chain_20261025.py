"""Fixed-depth mask replacement comparison; baseline P/R replay and partial-safe report."""
from pathlib import Path
import argparse,datetime as dt,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from mousegeoforge.semseg_training import digest
from calibrated_mouse_data_20261025 import OUT,points
from evaluate_calibrated_mouse_semseg_chain_20261025 import NEW,RESUME,ANCHOR,verify_registration
from evaluate_calibrated_mouse_spatial_pipeline_20261025 import cloud_metrics

def anchor_metrics():
 reg=verify_registration();dest=NEW/'semseg_chain_anchor_metrics_v1.json';assert not dest.exists();manifest=json.loads((OUT/'data_manifest.json').read_text());old=json.loads((ANCHOR/'evaluation.json').read_text());byid={s['id']:s for s in old['scenes']};rows=[];inputs={}
 for index,scene in enumerate(manifest['scenes']):
  if scene['split']!='dev':continue
  predicted=[];truth=[]
  for view,source in enumerate(scene['views']):
   path=ANCHOR/f'scene{index:06d}/view{view}.npz';inputs[str(path.relative_to(ROOT))]=digest(path)
   assert digest(ROOT/source['truth'])==source['truth_sha256'];inputs[source['truth']]=source['truth_sha256']
   with np.load(path) as z:p,m,k,e=z['depth_mm'].copy(),z['mask'].copy(),z['K'].copy(),z['E'].copy()
   with np.load(ROOT/source['truth']) as z:t,b=z['depth'].copy(),z['body'].astype(bool)
   predicted.append(points(p,m&(p>0),k,e));truth.append(points(t,b,k,e))
  metrics=cloud_metrics(np.concatenate(predicted),np.concatenate(truth));expected=byid[scene['id']]['pointcloud']['fscore']
  assert all(abs(metrics['fscore'][str(t)]-expected[str(t)])<=1e-12 for t in (1,2,5)),(scene['id'],metrics,expected)
  rows.append(dict(id=scene['id'],pointcloud=metrics))
 assert len(rows)==144
 result=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),passed=True,old_fscore_replayed_atol=1e-12,scene_denominator=144,view_denominator=432,old_evaluation_sha256=digest(ANCHOR/'evaluation.json'),source_sha256=reg['source_sha256'],inputs_sha256=inputs,scenes=rows)
 dest.write_text(json.dumps(result,indent=2)+'\n');print('ANCHOR_METRICS_REPLAYED',len(rows))

def summary(evaluation):
 scenes=evaluation['scenes'];assert len(scenes)==144 and evaluation['full_scene_failure_denominator']==144;views=[v for s in scenes for v in s['views']];assert len(views)==432
 result=dict(mask_iou=evaluation['summary']['mean_iou'],gt_depth_coverage=evaluation['summary']['gt_depth_coverage'],overlap_nonempty_views=sum(v['rmse_mm_on_overlap'] is not None for v in views),overlap_mean_view_rmse_mm=float(np.mean([v['rmse_mm_on_overlap'] for v in views if v['rmse_mm_on_overlap'] is not None])),missing_zero_sentinel_rmse_mm=float(np.mean([v['rmse_mm_missing_as_zero'] for v in views])))
 for kind in ['precision','recall','fscore']:result[kind]={str(t):float(np.mean([s['pointcloud'][kind][str(t)] for s in scenes])) for t in (1,2,5)}
 return result

def report():
 reg=verify_registration();data=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),registration_sha256=digest(NEW/'registrations/semseg_chain_v1.json'),all_three_complete=False,paired_full_roi_complete=False,arms={},limitations=reg['limitations'])
 anchor=json.loads((ANCHOR/'evaluation.json').read_text());path=NEW/'semseg_chain_anchor_metrics_v1.json'
 if path.exists():
  extra=json.loads(path.read_text());assert extra['passed'] and extra['old_evaluation_sha256']==digest(ANCHOR/'evaluation.json');assert [s['id'] for s in extra['scenes']]==[s['id'] for s in anchor['scenes']]
  for dst,src in zip(anchor['scenes'],extra['scenes']):dst['pointcloud']=src['pointcloud']
  data['arms']['anchor']=dict(status='complete',summary=summary(anchor),evaluation_sha256=digest(ANCHOR/'evaluation.json'),extra_metrics_sha256=digest(path))
 else:data['arms']['anchor']=dict(status='missing_anchor_P_R_replay')
 for arm in ['full','roi']:
  directory=NEW/f'semseg_chain_{arm}_v1';path=directory/'evaluation.json'
  if not path.exists():data['arms'][arm]=dict(status='missing_or_partial');continue
  e=json.loads(path.read_text());assert e['depth_checkpoint_sha256']==reg['depth_checkpoint_sha256'] and e['fixed_depth_common_replay_passed'];assert [s['id'] for s in e['scenes']]==[s['id'] for s in anchor['scenes']]
  data['arms'][arm]=dict(status='complete',summary=summary(e),evaluation_sha256=digest(path),segmentation_lineage=e['segmentation_lineage'],max_common_depth_difference_mm=max(v['fixed_depth_common_replay']['maximum_abs_difference_mm'] for s in e['scenes'] for v in s['views']))
 surface_path=NEW/'semseg_chain_surface_v1.json';cross_path=NEW/'semseg_chain_crossview_v1.json';surface=json.loads(surface_path.read_text()).get('pipelines',{}) if surface_path.exists() else {};cross=json.loads(cross_path.read_text()).get('pipelines',{}) if cross_path.exists() else {}
 for arm,value in data['arms'].items():
  directory=ANCHOR if arm=='anchor' else NEW/f'semseg_chain_{arm}_v1';s=surface.get(str(directory));c=cross.get(str(directory));value['surface']=s['summary'] if s else None;value['crossview']=c['summary'] if c else None
 data['paired_full_roi_complete']=all(data['arms'][a]['status']=='complete' for a in ['full','roi']);data['all_three_complete']=all(v['status']=='complete' and v['surface'] is not None and v['crossview'] is not None for v in data['arms'].values())
 if data['paired_full_roi_complete']:
  a=data['arms']['full']['summary'];b=data['arms']['roi']['summary'];data['roi_minus_full']={k:b[k]-a[k] for k in ['mask_iou','gt_depth_coverage','overlap_mean_view_rmse_mm']}
  for kind in ['precision','recall','fscore']:data['roi_minus_full'][kind]={str(t):b[kind][str(t)]-a[kind][str(t)] for t in (1,2,5)}
 for path in [NEW/'semseg_chain_report_v1.json',ROOT/'docs/experiments/calibrated_mouse_next12h_semseg_chain_20261025.json']:path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
 print(json.dumps(data))

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--anchor-only',action='store_true');args=parser.parse_args();anchor_metrics() if args.anchor_only else report()
if __name__=='__main__':main()
