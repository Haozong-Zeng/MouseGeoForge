import json,datetime as dt,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=root/'output/bottom_tail_fix_20261003';out.mkdir(exist_ok=True)
def win(p):return '\\\\wsl.localhost\\Ubuntu'+str(p).replace('/','\\')
name=sys.argv[1] if len(sys.argv)>1 else 'audit'
script=sys.argv[2] if len(sys.argv)>2 else 'audit_bottom_tail_20261003.py'
state=out/'state.json'
if not state.exists():state.write_text(json.dumps(dict(stage='ready',active_pid=None,active_job=None,updated_utc=dt.datetime.now(dt.timezone.utc).isoformat())))
job=dict(id=name,root=win(root),output=win(out),blend=win(root/'BlenderPyMouseProjects/Cylinder/PyMouseStudio-RealCapture-EarStudy-20261003/DELIVERY/PyMouseStudio-EarStudy-HeadRaise-Static.blend'),script=win(root/'BlenderScripts'/script),timeout_seconds=900,deadline_utc=(dt.datetime.now(dt.timezone.utc)+dt.timedelta(minutes=30)).isoformat())
(out/f'job_{name}.json').write_text(json.dumps(job,indent=2));print(out/f'job_{name}.json')
