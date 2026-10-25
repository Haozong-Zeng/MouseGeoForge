"""CPU-only evidence index; preserves incomplete/failed jobs and separates physical claims."""
from pathlib import Path
import argparse,datetime as dt,hashlib,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/calibrated_mouse_next12h_20261025'
REFRESH=['tools/report_calibrated_mouse_semseg_roi_20261025_v2.py','tools/check_calibrated_mouse_train_readiness_20261025_v2.py','tools/report_calibrated_mouse_semseg_chain_20261025.py','tools/report_calibrated_mouse_spatial_depth_20261025.py']
PATHS=dict(plan='plan.json',state='state.json',queue='queue.json',completion='completion.json',core='report.json',real='real_reference_feasibility_v1.json',classical='classical_baseline_findings_v1.json',dannce='dannce_holdout_audit_v1.json',semseg='semseg_roi_pair_report_v2.json',readiness='semseg_readiness_v2.json',chain='semseg_chain_report_v1.json',spatial='spatial_depth_pair_report_v1.json',semseg_failure='semseg_roi_schedule_failure_v1.json',readiness_failure='semseg_readiness_implementation_failure_v1.json',readiness_cancelled='semseg_readiness_cancelled_queue_v1.json',spatial_fixture_failure='spatial_CPU_test_import_failure_v1.json')

def sha_bytes(raw):return hashlib.sha256(raw).hexdigest()

def read_snapshot(path):
 raw=path.read_bytes();data=json.loads(raw,parse_constant=lambda value:(_ for _ in ()).throw(ValueError('NonfiniteJSON '+value)))
 return data,dict(path=str(path),sha256=sha_bytes(raw),bytes=len(raw))

def components(data):
 result={}
 real=data.get('real')
 if real:
  assert real['protected_session_model_calls']==0;pan=real['panomouse'];assert pan['dev_sessions']==['S03','S11']
  result['real_reference']=dict(status='complete_feasibility_only',panomouse={k:pan[k] for k in ['dev_sessions','timepoints','views','visible_body_reference_observations','original_synthetic_volume_ray_intersection_fraction','released_body_point_inside_synthetic_volume_fraction']},independent_real_surface_mm_verified=False,real_inference_deployment_verified=False,training_admission=False)
 else:result['real_reference']=dict(status='missing')
 classical=data.get('classical')
 if classical:
  assert classical['all144scene_ids_match'] and classical['views_per_method']==432 and not classical['gt_scale_fit']
  for row in classical['classical'].values():assert row['scenes']==144 and row['views']==432
  result['classical']=dict(status='complete_synthetic_diagnostic',methods=classical['classical'],conditional_error_must_be_read_with_coverage=True)
 else:result['classical']=dict(status='missing')
 dannce=data.get('dannce')
 if dannce:
  split=dannce['loader_path_replay'];assert split['train_test_overlap']==split['test_count'] and split['disjoint_test_count']==0 and not dannce['stored_render_scores']['heldout_accuracy_evidence']
  result['DANNCE_GS']=dict(status='complete_provenance_audit',train_count=split['train_count'],test_count=split['test_count'],overlap=split['train_test_overlap'],source_directory_currently_exists=dannce['current_source_path']['exists'],independent_heldout_geometry_proven=False,historical_actual_exposure_replayed=False,dense_surface_GT=False,depth_semantics='alpha-normalized expectedcameraZ, notfirstopaque-surface')
 else:result['DANNCE_GS']=dict(status='missing')
 semseg=data.get('semseg')
 if semseg and semseg.get('comparable'):
  assert semseg['campaign']['status']=='completed' and semseg['campaign']['full_paired_endpoint']==6000
  for row in semseg['arms'].values():assert row['status']=='complete' and row['groups']['all']['images']==879
  result['segmentation']=dict(status='complete_pair',weights='EMA',native_original_GT=True,arms={a:r['groups'] for a,r in semseg['arms'].items()},roi_minus_full=semseg['roi_minus_full'],old_failed_attempt_not_eligible=True)
 else:result['segmentation']=dict(status='missing_or_partial',comparable=False)
 readiness=data.get('readiness')
 if readiness:
  assert readiness['historical_gate_failures_unchanged'] and readiness['no_DEV_reads'];arms=readiness['arms'];complete=all(arms.get(a,{}).get('status') in ['TRAIN_window_passed','TRAIN_window_failed'] for a in ['full','roi']) and readiness['FULL_numerical_repeat']['status']=='passed'
  result['TRAIN_readiness']=dict(status='assessment_complete' if complete else 'missing_or_partial',arm_status={a:r['status'] for a,r in arms.items()},FULL_ready_for_trial_design=readiness['FULL_ready_for_trial_design'],numerical_repeat=readiness['FULL_numerical_repeat'],convergence_claim=False,model_quality_acceptance=False,automatic_unfreeze=False,adaptation_job_registered=readiness['adaptation_job_registered'])
 else:result['TRAIN_readiness']=dict(status='missing',FULL_ready_for_trial_design=False,automatic_unfreeze=False)
 chain=data.get('chain')
 if chain:
  complete=chain.get('all_three_complete',False)
  if complete:assert all(r['status']=='complete' and r['surface'] is not None and r['crossview'] is not None for r in chain['arms'].values())
  result['mask_replacement_chain']=dict(status='complete_with_surface_crossview' if complete else 'missing_or_partial',matched_new_FULL_ROI_complete=chain.get('paired_full_roi_complete',False),fixed_depth='oldseed17ROI6000gradient0EMA andoriginalMfeaturecrop',arms=chain['arms'],roi_minus_full=chain.get('roi_minus_full'),overlap_error_is_conditional=True,missing_zero_is_sentinel_not_valid_point_error=True,oldM_inverse_operator_differs_from_new_pair=True)
 else:result['mask_replacement_chain']=dict(status='missing')
 spatial=data.get('spatial')
 if spatial and spatial.get('all_three_pairs_comparable'):
  assert spatial['comparable_complete_seeds']==[17,29,43] and len(spatial['arms'])==6 and spatial['equal_seed_mean'] is not None
  assert all(r['status']=='complete_with_surface_crossview' for r in spatial['arms'].values())
  result['spatial_depth']=dict(status='complete_three_pairs',equal_seed_mean=spatial['equal_seed_mean'],oracle_and_predicted_mask_chain_separate=True,parameter_count_matched_but_hidden_width_differs=True,independent_real_mm_claim=False)
 else:
  assert not spatial or spatial.get('equal_seed_mean') is None
  result['spatial_depth']=dict(status='missing_or_partial',equal_seed_mean=None,comparable_complete_seeds=spatial.get('comparable_complete_seeds',[]) if spatial else [])
 return result

def worker_snapshot(saved):
 if not saved:return dict(status='no_saved_identity',matches=False)
 required=['pid','start_ticks','pgid','boot_id','command']
 if any(k not in saved for k in required):return dict(status='invalid_saved_identity',matches=False)
 try:
  pid=int(saved['pid']);text=Path(f'/proc/{pid}/stat').read_text();fields=text[text.rfind(')')+2:].split();command=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode().strip();boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip();observed=dict(pid=pid,start_ticks=fields[19],pgid=os.getpgid(pid),boot_id=boot,command=command)
  same=all(str(observed[k])==str(saved[k]) for k in required);return dict(status='matching_live_identity' if same else 'identity_mismatch',matches=same,saved=saved,observed=observed)
 except (FileNotFoundError,ProcessLookupError):return dict(status='process_absent',matches=False,saved=saved)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--refresh',action='store_true');args=parser.parse_args();refresh=[]
 if args.refresh:
  for script in REFRESH:
   try:
    p=subprocess.run([sys.executable,str(ROOT/script)],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=180);refresh.append(dict(script=script,returncode=p.returncode,stderr_tail=p.stderr[-2000:]))
   except subprocess.TimeoutExpired:refresh.append(dict(script=script,returncode=None,error='CPUreportrefresh_timeout180s'))
 data={};receipts={};missing={};errors=[]
 for key,name in PATHS.items():
  path=OUT/name
  if path.exists():
   try:data[key],receipts[key]=read_snapshot(path)
   except (ValueError,OSError) as exc:missing[key]=dict(path=name,error=str(exc));errors.append(key+':unreadable')
  else:missing[key]=dict(path=name)
 source_checks=[]
 for path in sorted((OUT/'registrations').glob('*.json')):
  try:reg,receipt=read_snapshot(path);receipts['registration:'+path.name]=receipt
  except (ValueError,OSError) as exc:errors.append('registration:'+path.name+':'+str(exc));continue
  for name,h in reg.get('source_sha256',{}).items():
   try:observed=sha_bytes((ROOT/name).read_bytes())
   except OSError:observed=None
   match=observed==h;source_checks.append(dict(registration=path.name,path=name,expected=h,observed=observed,matches=match))
   if not match:errors.append('source_pin:'+name)
 try:science=components(data)
 except (AssertionError,KeyError,TypeError) as exc:science=dict(status='component_consistency_error',detail=str(exc));errors.append('component_consistency:'+str(exc))
 plan=data.get('plan',{});expected=dict(start_utc='2026-10-25T19:50:28+00:00',optimization_cutoff_utc='2026-10-26T07:20:28+00:00',hard_finish_utc='2026-10-26T07:50:28+00:00')
 if any(plan.get(k)!=v for k,v in expected.items()):errors.append('immutable_deadline_mismatch')
 state=data.get('state',{});queue=data.get('queue',[]);jobs=state.get('jobs',{});counts={}
 for row in queue:
  status=jobs.get(row['id'],{}).get('status','pending_or_active');counts[status]=counts.get(status,0)+1
 workers={}
 for name in ['controller_process','watchdog_process','guardian_process','active_child']:
  path=OUT/(name+'.json')
  if path.exists():
   value,receipt=read_snapshot(path);receipts[name]=receipt;workers[name]=worker_snapshot(value.get('identity') if name=='active_child' else value)
 failures={key:dict(receipt=receipts[key],details={k:v for k,v in data[key].items() if k in ['failure','reason','cause','attempt','stage','status','actual_GPU_repeat_run','original_source_sha256']}) for key in ['semseg_failure','readiness_failure','readiness_cancelled','spatial_fixture_failure'] if key in data}
 complete_statuses=['complete_feasibility_only','complete_synthetic_diagnostic','complete_provenance_audit','complete_pair','assessment_complete','complete_with_surface_crossview','complete_three_pairs'];required_complete=len(science)==7 and all(r.get('status') in complete_statuses for r in science.values()) if 'component_consistency_error' not in str(science.get('status')) else False
 all_success=bool(queue) and all(jobs.get(j['id'],{}).get('status')=='completed' for j in queue)
 result=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),report_source_sha256=sha_bytes(Path(__file__).read_bytes()),immutable_window=expected,available_evidence_checks_passed=not errors,verification_errors=errors,required_evidence_complete=required_complete,all_scheduled_jobs_completed_successfully=all_success,job_counts=counts,current_active_job=state.get('active_job'),components=science,known_failed_or_cancelled_attempts=failures,source_pin_count=len(source_checks),source_pin_checks=source_checks,input_JSON_receipts=receipts,missing_inputs=missing,refresh_results=refresh,worker_identity_snapshot=workers,completion_receipt=data.get('completion'),queue_closed_file_exists=(OUT/'QUEUE_CLOSED').exists(),global_limits=['No independentrealvisible-surfacemm/hardware-sync/gaugeproof; releasedanatomicaltriangulatedjoints notskinGT','Onlysixcorrelatedsyntheticposes/reusedDEV; no significance/unseenrealpose/instancecount/modelpromotionclaim','Oracledepthsupport vsautomaticmaskchain andoverlaperror vscoverage/P_R mustremainseparate','LowpointcloudF1/normal/gradient/crossviewcanconflict; no aggregatescalaraccuracywinner','TRAINreadinessisnotconvergence, qualityacceptanceorautomaticencoderunfreeze; historicalfailedgatesretained'],lifecycle='ThisCPUreport doesnotstopworkers, closequeue, orpauseheartbeat. Separatelifecycleidentityverification/safestop/automationpause stillrequired atoriginaldeadlines.')
 for p in [OUT/'evidence_status_v1.json',ROOT/'docs/experiments/calibrated_mouse_next12h_evidence_status_20261025.json']:
  tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');tmp.replace(p)
 print(json.dumps(dict(available_evidence_checks_passed=not errors,required_evidence_complete=required_complete,all_scheduled_jobs_completed_successfully=all_success,source_pins=len(source_checks),job_counts=counts,verification_errors=errors)))
 if errors or any(r.get('returncode')!=0 for r in refresh):raise SystemExit(2)
if __name__=='__main__':main()
