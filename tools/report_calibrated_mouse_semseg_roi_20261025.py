"""Refreshable native-original full-denominator paired ROI segmentation report."""
from pathlib import Path
import datetime as dt
import json
import numpy as np
import hashlib
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mousegeoforge.semseg_training import BalancedSchedule
import torch
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/calibrated_mouse_next12h_20261025';BASE=OUT/'semseg_roi_pair_v1'
def main():
 campaign=json.loads((BASE/'completion.json').read_text()) if (BASE/'completion.json').exists() else {}
 result=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),campaign=campaign,comparable=False,arms={},no_DEV_selection=True,no_formal_convergence_claim=True)
 for arm in ['full','roi']:
  f=BASE/(arm+'_dev/evaluation.json')
  if not f.exists():result['arms'][arm]=dict(status='missing_or_partial');continue
  data=json.loads(f.read_text());rows=data['ema'];assert len(rows)==879 and len({r['id'] for r in rows})==879
  groups={}
  for name,rr in [('all',rows),*[(f'{domain}/{bottom}',[r for r in rows if r['domain']==domain and r['bottom']==bottom]) for domain in ['real','synthetic'] for bottom in [True,False]]]:
   groups[name]=dict(images=len(rr),**{key:float(np.mean([r['native'][key] for r in rr])) for key in ['iou','dice','boundary_f1']})
  result['arms'][arm]=dict(status='complete',groups=groups)
 if campaign.get('status')=='completed' and all(a['status']=='complete' for a in result['arms'].values()):
  assert campaign['full_paired_endpoint']==6000
  full=json.loads((BASE/'full_006000/protocol.json').read_text());roi=json.loads((BASE/'roi_006000/protocol.json').read_text());assert {k:v for k,v in full.items() if k!='crop_mode'}=={k:v for k,v in roi.items() if k!='crop_mode'}
  items=OUT/'semseg_roi_coarse_cache_v1/items.json';rows=json.loads(items.read_text())['items'];schedule=BalancedSchedule(rows,4,17);exposures={}
  for arm in ['full','roi']:
   path=BASE/f'{arm}_006000/step_006000.pt';ck=torch.load(path,map_location='cpu',weights_only=False);assert ck['step']==6000 and len(ck['metrics'])==6000
   for record in ck['metrics']:
    expected=[rows[i]['id'] for i in schedule.at(record['step']-1+18000)];assert record['local_ids']==record['global_ids']==expected
   exposures[arm]=hashlib.sha256(json.dumps([r['local_ids'] for r in ck['metrics']]).encode()).hexdigest();result['arms'][arm]['checkpoint_sha256']=sha(path)
  assert exposures['full']==exposures['roi'];result['actual_exposure_verified']=exposures
  a=json.loads((BASE/'full_dev/evaluation.json').read_text())['ema'];b=json.loads((BASE/'roi_dev/evaluation.json').read_text())['ema'];assert [(r['id'],r['domain'],r['bottom']) for r in a]==[(r['id'],r['domain'],r['bottom']) for r in b]
  result['comparable']=True;result['roi_minus_full']={g:{k:result['arms']['roi']['groups'][g][k]-result['arms']['full']['groups'][g][k] for k in ['iou','dice','boundary_f1']} for g in result['arms']['full']['groups']}
 (OUT/'semseg_roi_pair_report_v1.json').write_text(json.dumps(result,indent=2)+'\n');(ROOT/'docs/experiments/calibrated_mouse_next12h_semseg_roi_20261025.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
