import datetime as dt
import hashlib
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
out=root/'output/pymousestudio_ear_cycle_20261003';out.mkdir(exist_ok=False)
project=root/'BlenderPyMouseProjects/Cylinder/PyMouseStudio-RealCapture-EarStudy-20261003';project.mkdir(exist_ok=False)
source=root/'BlenderPyMouseProjects/Cylinder/PyMouseStudio-RealCapture-Replica-20261002/DELIVERY'
def win(p):return '\\\\wsl.localhost\\Ubuntu'+str(p).replace('/','\\')
sources={p:str(source/f'PyMouseStudio-RealCapture-{n}-Static.blend') for p,n in [('climb','Climb'),('quadruped','Quadruped'),('headraise','HeadRaise')]}
hashes={p:hashlib.sha256(Path(s).read_bytes()).hexdigest() for p,s in sources.items()}
plan=dict(start_utc='2026-10-03T07:58:00+00:00',deadline_utc='2026-10-03T08:58:00+00:00',heartbeat_minutes=10,automation_id='pymousestudio',scope='Ear-only material study. No pose/camera/light/glass/coating/other material changes. New candidate files only; prior delivery immutable.',sources=sources,source_sha256=hashes,project_directory=str(project),reference_root=str(root/'output/pymousestudio_realism_cycle_20261002/references'),reference_groups={'climb':'FR2/frame_000649','quadruped':'FR2/frame_001104','headraise':'AZR2L2/frame_000129'},heldout='FR2/frame_001429: after selection only, never tune; GR2/UR2/UR3 forbidden',protocol=['Audit actual ear shader links, face assignments, evaluated geometry, projected ear ROI before changes.','Separate brightness/roughness/SSS/thin-tissue transmission hypotheses. No emission or global exposure fixes. Gray capture fit, not measured spectral physiology.','Native1080x1200 fixed-seed128sample Cycles OptiX final comparisons. Crops plus full-view checks, at least two views during screening and three poses for finalist.','Compare evaluated skin and bone/camera matrices, immutable source hashes; single shared GPU lock and owned PID timeouts.','Last10minutes package accepted candidate or explicit no-improvement result, copy Windows Downloads, pause automation. Do not extend.'],windows_export='C:/Users/55395/Downloads/PyMouseStudio-EarStudy-20261003')
(out/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2))
(out/'state.json').write_text(json.dumps(dict(stage='initialized',active_pid=None,active_job=None,best_candidate=None,automation_id='pymousestudio',updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),next_action='Run audit_baseline job then inspect audit.json and CF renders with actual visual tool. Launch independent material probes only after audit. Deadline08:58UTC; newoutput/newproject, previousdeliveryreadonly.'),ensure_ascii=False,indent=2))
job=dict(id='audit_baseline',root=win(root),output=win(out),blend=win(Path(sources['climb'])),script=win(root/'BlenderScripts/audit_ear_cycle_20261003.py'),timeout_seconds=360,deadline_utc=plan['deadline_utc'])
(out/'job_audit.json').write_text(json.dumps(job,indent=2))
print(json.dumps(plan,ensure_ascii=False,indent=2))
