"""Record verified live start and completed baseline decisions, without PID mutation."""
import datetime as dt
import fcntl
import json
from pathlib import Path
from bottom_anchor_common_20261002 import ROOT,OUT,PRIOR,write_state


def main():
    run=OUT/'superanimal_quadruped_hrnet_v1'
    state=json.loads((OUT/'state.json').read_text())
    assert state['active_run']==run.name and Path(f'/proc/{state["pid"]}').exists()
    proof=json.loads((run/'initialization_verified.json').read_text())
    first=json.loads((run/'first_gpu_step.json').read_text())
    assert proof['backbone_tensors_exact']==1830 and proof['missing_gradient_exactly_zero']
    assert first['device']=='cuda:0' and first['optimizer_state_entries']>0
    assert first['input_shape']==[8,3,768,704]
    plan=json.loads((OUT/'plan.json').read_text())
    with (ROOT/plan['gpu_lock']).open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:pass
        else:raise AssertionError('Training PID present but sharedGPUlock notheld')
    baseline=json.loads((OUT/'dlc_150to200_v1/dataset_manifest.json').read_text())
    current=json.loads((run/'dataset_manifest.json').read_text())
    for key in ['entries','old_indices','context_manifest_indices','train_indices','validation_indices']:
        assert current[key]==baseline[key]
    report=json.loads((OUT/'extended_teacher_review_v2/summary.json').read_text())
    decision=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        prior_STOP_preserved=(PRIOR/'STOP').is_file(),cohort_same_exact=True,
        completed_baselines=['DLC200','LP60'],
        baseline_metrics={k:{g:report['models'][k][g] for g in ['all','bottom','side','swap_diagnostic']}
            for k in ['DLC150_best135','DLC200_best200','LP30_best24','LP60_best45']},
        DLC_decision='NoResNet300 now: latevalidationerror oscillates21.6-22.4inputpx; pawsource mean35.60to34.48, >100px90to89. Marginalgains; keep200checkpoint andmove torequestedQuadruped.',
        LP_decision='Completed60epochs; insufficientcontinuedbenefit toextend100blindly. Preserveoldbest24 andextendedbest45 separately.',
        visual_review=dict(files=['learning_curves.png','DLC_gain_AZR2L2_844_camC.jpg','DLC_regression_AZR2L2_1169_camA.jpg'],
            observations=['AZR2L2f844camC:200corrects collapsedhindpawoutputs seenat135.',
                'AZR2L2f1169camA:hindpaw17/18swap reappearsat200 after135 hadcorrectidentity. Aggregateimprovementdoesnotmeanallcasesimprove.'],
            directory='C:/Users/55395/Downloads/MouseGeoForge-BottomAnchor-20261002/extended_teacher_review_v2'),
        live_run=run.name,pid=state['pid'],shared_gpu_lock_held=True,initialization=proof,first_gpu_step=first,
        optimization='OfficialQuadruped1830tensorbackbone, freshlyinitialized7pointhead; AdamW1e-4,150epochs,decays100/130x0.3;batch8;noGTROI;sameaugmentationandmissinglabelpolicy.',
        caveat='Notpurearchitectureablation:pretraining,backbone,heatmaphead,optimizationdiffer. Reuseddevelopmentandsingleseed; bottom-viewgeneralizationnotguaranteed.',
        next='MonitoractiveQuadruped; savecheckpointatlocal/globaldeadline. Afterstageexportpairedsource/inputmetrics,pawlargeerrors,identitydiagnostics,camera/sessionandactualvisuals. DoNOTrestartoldDLC/LPqueue.',
        pseudo_gt_released=False,locked_holdout_accessed=False,global_deadline_utc=plan['deadline_utc'])
    path=OUT/'heartbeat_20261002_1035_decision.json';assert not path.exists()
    path.write_text(json.dumps(decision,indent=2))
    write_state(latest_decision=str(path),latest_paired_review=str(OUT/'extended_teacher_review_v2/summary.json'),
        convergence_decision='DLC200 marginaladditionalgain;LP60notsustained. OfficialQuadrupedHRNettrainingactivewithverifiedweightsandmaskedmissinglabels.',
        next_action=decision['next'])
    print(json.dumps(dict(decision=str(path),pid=state['pid'],epoch=state['epoch'],verified=True)))


if __name__=='__main__':main()
