"""Compact resume-cycle evidence and deadline report; no training mutation."""
from pathlib import Path
import json,datetime
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/calibrated_mouse_resume8h_20261025'
OLD=ROOT/'output/calibrated_mouse_10h_20261025'
def read(p):
 return json.loads(p.read_text()) if p.exists() else None
def main():
 plan=read(OUT/'plan.json');state=read(OUT/'state.json');queue=read(OUT/'queue.json')
 report={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'plan':plan,'state':state,'queue':[{'id':j['id'],'status':state.get('jobs',{}).get(j['id'],{}).get('status','pending')} for j in queue], 'evaluations':{},'limits':['Fixed known K/E, synchronized visible skin-surrogate surfaces only','No GT scale alignment; no independent real millimeter accuracy claim','Six correlated synthetic source poses, no independent unseen-real-pose proof','Single mouse semantic union and bbox are not multi-mouse instance detection','Original L/M 18000 endpoint failed formal plateau; encoder adaptation remains gated']}
 for key,path in {'L':OUT/'L_dev_v1/evaluation.json','M':OUT/'M_dev_v1/evaluation.json','full_chain':OUT/'pipeline_D1_M_dev_v1/evaluation.json','controls':OLD/'controls_report.json'}.items():
  value=read(path)
  if value:
   report['evaluations'][key]={'path':str(path),'keys':list(value),'summary':{k:v for k,v in value.items() if k in ['summary','aggregate','metrics','held_source_frame_pairs','breakdown']}}
 out=OUT/'report.json';tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n');tmp.replace(out)
 p=ROOT/'docs/experiments/calibrated_mouse_resume8h_20261025.json';p.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
 print(json.dumps({'active_job':state.get('active_job'),'completed_current_jobs':sum(x['status']=='completed' for x in report['queue']),'evaluations':list(report['evaluations'])}))
if __name__=='__main__':main()
