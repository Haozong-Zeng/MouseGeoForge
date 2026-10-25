"""Exact original TRAIN convergence-rule audit; no retrospective rule changes."""
from pathlib import Path
import json,datetime,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from mousegeoforge.semseg_training import digest
OLD=ROOT/'output/calibrated_mouse_10h_20261025';OUT=ROOT/'output/calibrated_mouse_resume8h_20261025'
def main():
 path=OLD/'semseg_campaign_s17_v1/progress.json';j=json.loads(path.read_text());result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':digest(Path(__file__)),'progress_sha256':digest(path),'common_endpoint':j['common_endpoint'],'recorded_converged':j['converged'],'rule':'last5 TRAIN endpoint metric spans; loss<=1percent mean, IoU and boundaryF1<=.002, all/each stratum and BOTH arms','arms':{}}
 for arm,history in j['history'].items():
  last=history[-5:];failures=[];checks=[]
  for group in last[0]['metrics']:
   for metric in ['loss','iou','boundary_f1']:
    values=[x['metrics'][group][metric] for x in last];span=max(values)-min(values);tol=.01*max(1e-8,float(np.mean(values))) if metric=='loss' else .002
    row={'group':group,'metric':metric,'values':values,'span':span,'tolerance':tol,'passes':span<=tol};checks.append(row)
    if not row['passes']:failures.append(row)
  result['arms'][arm]={'last5_steps':[x['step'] for x in last],'failed_rule_count':len(failures),'failed_rules':failures,'all_checks':checks}
 result['both_pass']=all(v['failed_rule_count']==0 for v in result['arms'].values());assert result['both_pass']==j['converged'];result['encoder_unfreeze_admitted']=False
 p=OUT/'head_plateau_audit_v1.json';assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'both_pass':result['both_pass'],'failed':{a:v['failed_rules'] for a,v in result['arms'].items()}}))
if __name__=='__main__':main()
