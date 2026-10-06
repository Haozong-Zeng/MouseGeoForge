import datetime as dt,json,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=root/'output/pymousestudio_ear_cycle_20261003'
audit=json.loads((out/'audit_baseline/audit.json').read_text());assert audit['complete']
def win(p):return '\\\\wsl.localhost\\Ubuntu'+str(p).replace('/','\\')
job=dict(id='probe_v001',root=win(root),output=win(out),blend=audit['baseline_blend'],script=win(root/'BlenderScripts/probe_ear_materials_20261003.py'),timeout_seconds=900,deadline_utc='2026-10-03T08:58:00+00:00')
(out/'job_probe_v001.json').write_text(json.dumps(job,indent=2))
state=json.loads((out/'state.json').read_text(encoding='utf-8-sig'));state.update(updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),baseline_blend=audit['baseline_blend'],ear_audit=str(out/'audit_baseline/audit.json'),next_action='Audit baseline done actualnativeCF realreference andsyntheticF visuallyreviewed. Ear slot4 Replica_BalancedSkin_4:Principled gray.12 rough.8 spec.08 SSS.025scale.15, no translucentnode. First2audit failed onlyonnameassert(no changes);fixedconfirmedslotnaminghistory. v001 five independent candidates CF128:gray.3,.6; .3rough.5spec.25; .3+translucentmix.3color.6; .3SSS.15scale.3neutralradius. Each scene copied frombaseline,geometry/object/bone/non-earmaterialsnapshot exactsame. NEXT waitownedprobePID completion/readmanifest, build ear crop comparison withaudit.rois andrealreferences(actualrealearC[260,245,395,340],F[770,270,975,385]);actualvisualinspect,choosepromisingcandidate notjustbrightness. Then refine ifnecessary andvalidate3poses+side/backear+bottom. ExportnewWindowsEarStudydirectory, notoldDELIVERY. Deadline08:58UTC;last10minwrap/pauseautomation. AutomationpymousestudioACTIVEevery10min.')
(out/'state.json').write_text(json.dumps(state,ensure_ascii=False,indent=2))
subprocess.run([sys.executable,str(root/'tools/launch_realism_job.py'),str(out/'job_probe_v001.json')],check=True)
