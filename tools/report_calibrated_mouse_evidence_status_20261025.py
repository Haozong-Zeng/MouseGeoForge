"""Read-only evidence integration. Refreshes CPU reporters only when requested.

Primary native labels, all registered seeds and failure denominators are explicit.
Does not choose checkpoints, thresholds, models or training stopping conditions.
"""
from pathlib import Path
import argparse,datetime as dt,hashlib,json,math,os,statistics,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/calibrated_mouse_resume8h_20261025';OLD=ROOT/'output/calibrated_mouse_10h_20261025'

def digest_bytes(b):return hashlib.sha256(b).hexdigest()
class Snapshot:
    def __init__(self):self.inputs={}
    def read(self,path):
        path=Path(path)
        if not path.exists():return None
        b=path.read_bytes();value=json.loads(b)
        self.inputs[str(path.relative_to(ROOT))]=dict(sha256=digest_bytes(b),bytes=len(b))
        return value

def original_native(data):
    assert data['denominator']==879
    result={}
    for arm in ['L','M']:
        a=data['arms'][arm];assert len(a['rows'])==879 and len({r['id'] for r in a['rows']})==879
        assert set(a['groups'])=={'all','real/True','real/False','synthetic/True','synthetic/False'}
        assert a['groups']['all']['images']==879
        assert sum(a['groups'][k]['images'] for k in a['groups'] if k!='all')==879
        for g in a['groups'].values():assert 0<=g['native_original_iou']<=1
        result[arm]=dict(checkpoint_sha256=a['checkpoint_sha256'],groups={g:{k:v for k,v in metrics.items() if k in ['images','native_original_iou','native_original_boundary_f1_tol1']} for g,metrics in a['groups'].items()})
    return dict(status='complete',truth='Original native labels; legacy GT roundtrip excluded',arms=result)

def full_chain(data,expected_scenes=144):
    assert data['oracle_mask_inference'] is False and data['calibration_fixed'] is True and data['gt_scale_alignment'] is False
    assert data['independent_real_mm_accuracy_claim'] is False and data['multiple_mouse_instance_detection_claim'] is False
    scenes=data['scenes'];views=[v for s in scenes for v in s['views']]
    assert len(scenes)==data['full_scene_failure_denominator']==expected_scenes and len(views)==3*expected_scenes
    overlap=[v['rmse_mm_on_overlap'] for v in views if v['rmse_mm_on_overlap'] is not None]
    assert overlap
    return dict(status='complete',scenes=len(scenes),views=len(views),nonempty_overlap_views=len(overlap),summary=data['summary'],
        mean_view_overlap_RMSE_mm=statistics.mean(overlap),overlap_metric_excludes_missed_foreground=True,
        mean_view_missing_sentinel_zero_RMSE_mm=statistics.mean(v['rmse_mm_missing_as_zero'] for v in views),
        sentinel_metric_is_not_valid_point_physical_error=True,no_GT_scale_alignment=True,independent_real_mm_accuracy=False)

def matched_roi(data):
    pairs=data['pairs'];assert set(pairs)=={'17','29','43'},'All three registered seeds required'
    for p in pairs.values():
        assert p['comparable'] and p['same_recorded_scene_view_exposure']
        for mode in ['full','roi']:
            c=p['chain'][mode];assert c['scene_denominator']==144 and c['view_denominator']==432
            assert math.isfinite(c['mean_view_overlap_rmse_mm']) and c['mean_view_overlap_rmse_mm']>=0
            assert all(0<=c['pointcloud_fscore'][str(t)]<=1 for t in [1,2,5])
    return dict(status='complete',registered_seeds=[17,29,43],per_seed={seed:{k:p[k] for k in ['comparable','oracle','chain']} for seed,p in pairs.items()},
        mean_of_three_seed_mean_view_overlap_RMSE_mm={m:statistics.mean(p['chain'][m]['mean_view_overlap_rmse_mm'] for p in pairs.values()) for m in ['full','roi']},
        mean_of_three_seed_chain_F1_at5mm={m:statistics.mean(p['chain'][m]['pointcloud_fscore']['5'] for p in pairs.values()) for m in ['full','roi']},
        oracle_support_is_separate_from_predicted_mask_chain=True,three_seeds_share_same_correlated_DEV_data=True)

def adapter_status(completion,progress,audited):
    if completion and completion.get('head_converged') is False:
        return 'protocol_contradiction_F_U_without_gate' if any(a in audited for a in ['F','U']) else 'not_admitted_TRAIN_gate'
    if completion and completion.get('F_U_completed'):
        return 'complete' if all(a in audited for a in ['F','U']) else 'completed_campaign_but_native_audit_pending'
    if any(a in audited for a in ['F','U']):return 'partial_pair_do_not_claim_comparable'
    if completion:return 'not_completed_or_budget_not_admitted'
    return 'pending_TRAIN_gate' if not progress.get('converged') else 'gate_passed_optional_budget_check_pending'

def audited_pipelines(data,expected_count,count_key,per_pipeline_count,scope):
    assert len(data['pipelines'])==expected_count,'Missing registered pipelines'
    for value in data['pipelines'].values():assert value['summary'][count_key]==per_pipeline_count,'Incomplete denominator'
    return dict(status='complete',metrics_scope=scope,rules=data['rules'],pipelines={k:v['summary'] for k,v in data['pipelines'].items()})

def summarize_optional_pair(data):
    return dict(status='complete_comparable' if data.get('comparable') else 'not_complete_comparable',data=data)

def process_identity(saved):
    try:
        pid=saved['pid'];stat=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()
        actual=dict(pid=pid,start_ticks=stat[19],pgid=os.getpgid(pid),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),command=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode().strip())
        return 'verified_running' if actual==saved else 'identity_mismatch'
    except (FileNotFoundError,ProcessLookupError):return 'not_running'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--refresh',action='store_true');args=parser.parse_args();refresh=[]
    if args.refresh:
        for tool in ['report_calibrated_mouse_roi_pair_20261025.py','report_calibrated_mouse_roi_gradient_20261025.py','audit_calibrated_mouse_head_phase_native_20261025.py','report_calibrated_mouse_geometric_failures_20261025.py','report_calibrated_mouse_pixel_center_pair_20261025.py','report_calibrated_mouse_resume8h_20261025.py']:
            with (OUT/(tool+'.refresh.log')).open('w') as log:
                try:r=subprocess.run([sys.executable,str(ROOT/'tools'/tool)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=180);refresh.append(dict(tool=tool,returncode=r.returncode))
                except subprocess.TimeoutExpired:refresh.append(dict(tool=tool,status='timeout'))
    snap=Snapshot();plan=snap.read(OUT/'plan.json');state=snap.read(OUT/'state.json');queue=snap.read(OUT/'queue.json');now=dt.datetime.now(dt.timezone.utc)
    report=dict(schema='calibrated-mouse-evidence-status-v1',utc=now.isoformat(),window={k:plan[k] for k in ['start_utc','optimization_cutoff_utc','hard_finish_utc']},
        optimization_seconds_remaining=max(0,(dt.datetime.fromisoformat(plan['optimization_cutoff_utc'])-now).total_seconds()),
        cycle_status=state['status'],cycle_completion=snap.read(OUT/'completion.json'),workers={},queue=[],components={},verification_errors=[],refresh=refresh,
        scope=['LOCAL only; fixed known K/E, native camera-Z mm, no GT scale fitting','Real masks manual; synthetic visible opaque skin excludes fur/glass','Six correlated synthetic poses; no independent real-mm/unseen-real-pose claim','Union mask/bbox is not a multi-mouse instance detector','Reused DEV diagnostics do not select thresholds, checkpoints or model promotion'])
    for f in ['controller_process.json','watchdog_process.json','guardian_process.json','active_child.json']:
        saved=snap.read(OUT/f);report['workers'][f]=process_identity(saved.get('identity',saved)) if saved else 'identity_unavailable'
    for job in queue:
        status=state.get('jobs',{}).get(job['id'],{}).get('status','pending')
        if job['id']==state.get('active_job') and report['workers']['active_child.json']=='verified_running':status='verified_running'
        report['queue'].append(dict(id=job['id'],status=status,optimization=job.get('optimization',True)))
    def component(name,path,fn):
        try:
            data=snap.read(path)
            report['components'][name]=fn(data) if data is not None else {'status':'not_available'}
        except (AssertionError,KeyError,ValueError,TypeError) as e:
            report['components'][name]={'status':'verification_failed_no_claim','error':repr(e)};report['verification_errors'].append(dict(component=name,error=repr(e)))
    component('original18000_semseg_primary',OUT/'native_metric_audit_v1.json',original_native)
    component('originalM_D1_full_chain',OUT/'pipeline_D1_M_dev_v1/evaluation.json',full_chain)
    component('matched_FULL_ROI',OUT/'roi_pair_report_v1.json',matched_roi)
    component('surface_quality_diagnostic',OUT/'surface_quality_audit_v1.json',lambda d:audited_pipelines(d,6,'views',432,'GT-body eligible pairs; covered errors and missing penalties separate'))
    component('crossview_diagnostic',OUT/'crossview_audit_v1.json',lambda d:audited_pipelines(d,6,'directed_view_pairs',864,'GT co-visible support; consistency is not accuracy'))
    component('head_phase_native',OUT/'head_phase_native_report_v1.json',lambda d:dict(status='partial' if d['partial'] else 'complete_L_M',audited={k:v['groups'] for k,v in d['audited'].items()},unavailable=d['unavailable']))
    progress=snap.read(OUT/'semseg_head_phase_v1/progress.json') or {};completion=snap.read(OUT/'semseg_head_phase_v1/completion.json');native=snap.read(OUT/'head_phase_native_report_v1.json') or {}
    report['head_phase']=dict(common_endpoint=progress.get('common_endpoint'),TRAIN_converged=progress.get('converged'),completion_status=completion.get('status') if completion else 'pending',optional_encoder_pair=adapter_status(completion,progress,native.get('audited',{})))
    component('ROI_gradient_pair',OUT/'roi_gradient_pair_report_v1.json',lambda d:dict(status='complete_three_oracle_pairs' if d['completed_pairs']==3 and all(p['comparable'] for p in d['pairs'].values()) else 'partial_or_not_comparable',data=d))
    component('pixel_center_pair',OUT/'pixel_center_pair_report_v1.json',summarize_optional_pair)
    for name,file in [('gradient_surface','roi_gradient_surface_audit_v1.json'),('gradient_crossview','roi_gradient_crossview_audit_v1.json')]:
        component(name,OUT/file,lambda d,key=name:audited_pipelines(d,3,'views' if key=='gradient_surface' else 'directed_view_pairs',432 if key=='gradient_surface' else 864,'Exploratory ROI gradient-loss followup'))
    def empty(d):
        qa=snap.read(OUT/'empty_background_v1/qa.json');visual=snap.read(OUT/'empty_background_v1/formal_visual_review.json')
        assert qa and qa['passed'] and visual and visual['passed'];assert d['full_denominator']==d['unique_rgb_hashes']==len(d['views'])==36 and d['synthetic_only'] and not d['independent_real_background_claim']
        return dict(status='complete',denominator=36,summary=d['summary'],false_positive_frame_counts={k:sum(bool(v[k]) for v in d['views']) for k in ['any_foreground','component_ge20px','component_ge100px']},native_visual_review='representative12views, not all36',no_TRAIN_admission=True,independent_real_background_claim=False)
    component('empty_background_diagnostic',OUT/'empty_background_v1/M_common_endpoint_eval/evaluation.json',empty)
    def real(d):
        receipt=snap.read(OUT/'admitted_real_calibration_execution_v2.json');assert receipt and receipt['status']=='passed' and receipt['full_audit_sha256']==snap.inputs[str((OUT/'admitted_real_calibration_audit_v2.json').relative_to(ROOT))]['sha256']
        return dict(status='metadata_audit_complete_REAL3D_not_validated',images=d['images'],split_counts=d['split_counts'],source_frame_groups=len(d['frame_groups']),synchronization=d['synchronization'],units=d['units'],limits=d['limitations'])
    component('real_calibration_provenance',OUT/'admitted_real_calibration_audit_v2.json',real)
    component('inverse_resampling_exploratory',OUT/'inverse_resampling_audit_v1.json',lambda d:dict(status='complete_diagnostic_only',groups=d['groups'],primary_masks_unchanged=d['primary_prediction_masks_unchanged'],precision=d['probability_precision']))
    component('GT_label_resize_diagnostic',OUT/'label_resampling_control_v1.json',lambda d:dict(status='complete_GT_operator_only_NOT_model_ceiling',groups=d['groups'],limits=d['limitations']))
    pins=[]
    for path in sorted(OUT.glob('*registration*.json')):
        value=snap.read(path);mapping=value.get('source_sha256',{})
        if isinstance(mapping,dict):
            for source,h in mapping.items():
                p=ROOT/source;status='match' if p.exists() and digest_bytes(p.read_bytes())==h else 'mismatch_or_missing'
                pins.append(dict(registration=path.name,path=source,status=status))
    report['registered_source_pin_check']=dict(entries=len(pins),mismatches=[p for p in pins if p['status']!='match'],scope='source_sha256 dictionaries in top-level registration JSONs; archived/data contracts remain separate')
    report['input_snapshots']=snap.inputs;report['source_sha256']=digest_bytes(Path(__file__).read_bytes());report['available_evidence_checks_passed']=not report['verification_errors'] and not report['registered_source_pin_check']['mismatches']
    report['all_registered_jobs_completed_successfully']=all(j['status']=='completed' for j in report['queue'])
    if report['head_phase']['optional_encoder_pair'].startswith('protocol_contradiction'):
        report['available_evidence_checks_passed']=False;report['verification_errors'].append({'component':'encoder_admission','error':'F/U artifacts exist despite failed TRAIN gate'})
    target=OUT/'evidence_status_v1.json';temp=target.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(target)
    (ROOT/'docs/experiments/calibrated_mouse_evidence_status_20261025.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'verified':report['available_evidence_checks_passed'],'head':report['head_phase'],'components':{k:v['status'] for k,v in report['components'].items()},'errors':report['verification_errors'],'pin_mismatches':report['registered_source_pin_check']['mismatches']}))
if __name__=='__main__':main()
