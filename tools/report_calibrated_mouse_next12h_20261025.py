"""Compact refreshable lifecycle report; no model calls or selections."""
from pathlib import Path
import datetime as dt
import json
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/calibrated_mouse_next12h_20261025'
def read(p,default=None):return json.loads(p.read_text()) if p.exists() else default
def main():
    plan=read(OUT/'plan.json');state=read(OUT/'state.json',{});queue=read(OUT/'queue.json',[]);now=dt.datetime.now(dt.timezone.utc)
    result=dict(utc=now.isoformat(),plan=plan,state=state,active=read(OUT/'active_child.json',{}),pending=[j['id'] for j in queue if j['id'] not in state.get('jobs',{})],completion=read(OUT/'completion.json'),evidence_available=[p.name for p in OUT.glob('*') if p.name.endswith(('report_v1.json','feasibility_v1.json','findings_v1.json'))],limits='Real released3D sparse correlated proxy and anatomical landmarks are not visible-surface depthGT. No protectedTEST, no automatic promotion or retroactive prior gate pass.')
    for p in [OUT/'report.json',ROOT/'docs/experiments/calibrated_mouse_next12h_20261025.json']:
        tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(p)
    print(json.dumps(dict(utc=result['utc'],pending=result['pending'],active=result['active'].get('job'),completion=bool(result['completion']))))
if __name__=='__main__':main()
