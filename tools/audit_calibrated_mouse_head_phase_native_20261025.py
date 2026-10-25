"""Audit available complete phase evaluations, preserving partial/failed jobs."""
from pathlib import Path
import json,sys,datetime
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from audit_mouse_semseg_native_20261025_v2 import audit
from mousegeoforge.semseg_training import digest
OUT=ROOT/'output/calibrated_mouse_resume8h_20261025';RUN=OUT/'semseg_head_phase_v1'
def read(p):return json.loads(p.read_text()) if p.exists() else None
def main():
 result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'campaign_completion':read(RUN/'completion.json'),'audited':{},'unavailable':[],'native_original_labels':True,'DEV_selected_stopping_or_checkpoint':False,'source_sha256':digest(Path(__file__))}
 for arm in ['L','M','F','U']:
  directory=RUN/f'{arm}_dev';p=OUT/f'head_phase_{arm}_native_audit_v1.json'
  if not (directory/'evaluation.json').exists():result['unavailable'].append(arm);continue
  if p.exists():j=json.loads(p.read_text());value={k:v for k,v in j.items() if k!='rows'}
  else:value=audit(directory,p)
  result['audited'][arm]=value
 completion=result['campaign_completion'];result['F_U_completed']=bool(completion and completion.get('F_U_completed'));result['partial']=not all(a in result['audited'] for a in ['L','M'])
 (OUT/'head_phase_native_report_v1.json').write_text(json.dumps(result,indent=2)+'\n');(ROOT/'docs/experiments/calibrated_mouse_head_phase_native_20261025.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'available':list(result['audited']),'unavailable':result['unavailable'],'partial':result['partial']}))
if __name__=='__main__':main()
