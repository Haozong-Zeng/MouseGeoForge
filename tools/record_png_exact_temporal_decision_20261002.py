"""Freeze exact-input temporal outcome; plan a bounded official context-teacher pilot."""
import datetime as dt,json
from run_teacher_cycle_dlc_20261001 import OUT,write_state

def main():
    result=json.loads((OUT/'temporal_joint_states_pngexact_v1/summary.json').read_text())
    qa=json.loads((OUT/'png_exact_clips_v1/verification.json').read_text())
    peaks=json.loads((OUT/'temporal_paw_peaks_pngexact_v1/manifest.json').read_text())
    sensitivity=json.loads((OUT/'temporal_paw_peaks_pngexact_v1/encoding_sensitivity.json').read_text())
    assert qa['passed'] and qa['manual_frames_checked']==84
    assert result['sparse_framewise_parity']==dict(acceptance_mismatches=0,max_error_difference_px=0.0)
    assert all(q['peak_max_distance_px']==0 and q['ratio_max_difference']==0 for p in peaks for q in p['sparse_peak_parities'])
    report=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),result=result,
        verification=dict(manual_png_byte_exact=84,all3seed_sparse_peaks_exact=True,static_vs_dense_framewise_exact=True,
            originals_modified=False,already_rectified_videos_rectified_again=False),encoding_sensitivity=sensitivity,
        conclusions=[
            'Inputconfound resolved without parameter changes: all84raw-remap/PNG/letterbox/FFV1checks exact; all3seed sparsepeak positions/ratios exact; denseindependentjoint vs staticjoint has0acceptancemismatches,0pxerrordifference.',
            'PNG-exact temporal pilot adds6acceptedpoints(39->45/94),stress adds4(58->62/117). All97commonacceptedpoints unchanged. Severeerrors>100px persist1pilot and3stress. No evidence of correctingidentity with this fixedposthocmotioncost.',
            'Additional H264CRF16encoding changes3817/59904nativehighest-pawpeaks by>50px(~6.37%). This is pairedinput sensitivity,not GTaccuracy or independent samplecount. Preserve PNG-exact path for scientific comparisons.',
            'Visually inspected exactstressAZR2L2f1169 heldA/F:temporal resolves some previouslymissingforepaw evidence but retains contralateralhindpaw error. Do notcall morefilledpoints universallybetter.',
            'Stop addingposthocgates/motionweight sweeps. Existing2D teachers still sharesemanticerrors. A bounded learnedimage-context teacher is a materially different next hypothesis, not moreepochs on unchangedmodel.'
        ],
        implementation_evidence=dict(package='lightning-pose2.4.2 installedisolated',
            class_name='lightning_pose.models.heatmap_tracker_mhcrnn.HeatmapTrackerMHCRNN',
            input='B,5,3,H,W; native +/-2context frames; singleframe and multiframe heatmap heads',
            native_predict='selects higherconfidence head perpoint; must evaluate SF,MF,anddefaultselected separately becauseconfidence isnotcalibrated',
            dataset='HeatmapDataset(do_context=True), get_context_img_paths parses FIRSTdigitsequence instem; must use frame_000064.png under session/camera dirs, not session-digit-prefixed filenames',
            caution='Dataset silently substitutescenterifneighborfilemissing; must preverify everyinteriorneighbor and explicitlyrecord boundaryclamps. Missingannotation isunknown, notinvisible-negative.'),
        next_actions=[
            'Prepare output-onlyPNG-exact5frame context data for existing approved train/developmentvalidationcenters, excludingall24groups andnewSEM01entiregroup. Originalrawcamera gets one existingK/Dremap only. Same704x768letterbox and centerannotation order/NaNmask. Noholdoutimages. Verify centerbyteparity and no excluded frame anywhere incontexts.',
            'Use actual installedLPHeatmapTrackerMHCRNN andKLloss. Smoke-test Bx5x3x768x704, targetorder,missinggradientmask,boundaryclamps,and GPUmemory. Do not installnewframeworks or editLP package. Ifsmoke fails, recordfailure before training.',
            'If smoke passes,start ONE bounded supervisedseed17contextteacher: same7bodyids, source split, ImageNetResNet50initialization,30epochs ceiling and60minwallcap/globaldeadline, initialbatch4 withmemorycheck. SharedGPUlock,checkpoint/log/deadlineguards. No productioncheckpointoverwrite.',
            'Evaluate nativeSFhead,MFhead,defaultconfidence-selection separately; additionallyreplaceall5frameswithcenter atsamecheckpoint to isolate contextual input contribution. Comparebaselineonlyonidenticalremainingmanualcohort(newSEM01excluded). SFbranchjointlytrained differsfromoldstaticteacher, so do notclaimpuretrainingablation.',
            'Judge mean/P90,severeerrors,manualrelativeidentityconfusions andcoverage, not smoothness. Noautomaticallyrelabeleddata or strong3DGT. Stop newexperimentsat07:21:52UTC andpauseautomationatdeadline.'
        ],locked_holdout_opened=False,original_labels_or_calibration_changed=False,pseudo_gt_released=False,
        visual_review='C:/Users/55395/Downloads/MouseGeoForge-TeacherTemporal-20261001/temporal_joint_states_pngexact_v1')
    p=OUT/'heartbeat_20261002_0340_decision.json';p.write_text(json.dumps(report,indent=2))
    write_state(stage='png_exact_temporal_validated_no_identity_gain',pid=None,active_run=None,mode='idle_between_heartbeats',job_deadline_utc=None,
        latest_decision=str(p),latest_result=str(OUT/'temporal_joint_states_pngexact_v1/summary.json'),next_action=report['next_actions'][0],pseudo_gt_released=False)
    print(p)

if __name__=='__main__':main()
