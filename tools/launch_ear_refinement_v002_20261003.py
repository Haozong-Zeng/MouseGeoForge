import datetime as dt,json,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1];base=root/'output/pymousestudio_ear_cycle_20261003'
m=json.loads((base/'probe_v001/manifest.json').read_text());assert m['complete']
review=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),actually_viewed=['ear_compare_camC.jpg','ear_compare_camF.jpg'],decision='Prefer b_albedo060 for now; not final before pose validation.',observations=['Gray.60 yields visibly brighter rim and bowl closer to real references while retaining cavity shadow.','Gray.30 improves baseline but remains darker than real in C/F.','At gray.30, increased specular and SSS have relatively small visual effect; no demonstrated advantage yet.','30percent translucency makes current lighting response darker/flatter, not adopted.','Ear shape/orientation differs from real; brightness comparison is qualitative, not pixel alignment.'],next='Independent gray.45/.75 bounds; gray.60+SSS and mild15percent translucent controls; then three-pose validation.')
(base/'probe_v001/review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2))
def win(p):return '\\\\wsl.localhost\\Ubuntu'+str(p).replace('/','\\')
job=dict(id='probe_v002',root=win(root),output=win(base),blend=m['source'],script=win(root/'BlenderScripts/refine_ear_materials_v002_20261003.py'),timeout_seconds=720,deadline_utc='2026-10-03T08:58:00+00:00')
(base/'job_probe_v002.json').write_text(json.dumps(job,indent=2))
state=json.loads((base/'state.json').read_text(encoding='utf-8-sig'));assert state['active_pid'] is None
state.update(best_candidate=m['candidates']['b_albedo060']['blend'],updated_utc=review['utc'],latest_review=str(base/'probe_v001/review.json'),next_action='v001 CF crops actually viewed with realrefs. Preliminaryprefergray.60; low.30stilldark; translucency.3gray.3darker, spec/SSSlittlebenefit. v002 now4independentCF128probes:gray.45,.75;gray.60SSS.20scale.35;gray.60trans.15color.6. WaitownedPID/readmanifestthenassemblecomparev001b.60+v002+baseline+real,actualvisualreview. Selectsimplebestnotjustwhite;validate3staticposesCFplusbottomAandbackE, save3newclones, unchangedskin/bones/cameras/lights/non-earmats, original3SHA. WindowsEarStudy/index alreadyfirstCFcropboard; updateonlynewexportnotoldDELIVERY. Current08:13~,deadline08:58, last10minwrap/pauseautomation. Originalshapeproblemsoutofscope.')
(base/'state.json').write_text(json.dumps(state,ensure_ascii=False,indent=2))
subprocess.run([sys.executable,str(root/'tools/launch_realism_job.py'),str(base/'job_probe_v002.json')],check=True)
