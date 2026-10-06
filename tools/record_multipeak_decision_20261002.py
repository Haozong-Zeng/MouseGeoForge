"""Summarize frozen decoding experiments and predeclare next development check."""
import datetime as dt,json
from run_teacher_cycle_dlc_20261001 import ROOT,OUT,write_state

def main():
    multi=json.loads((OUT/'multipeak_geometry_v1/summary.json').read_text())
    joint=json.loads((OUT/'joint_paw_assignment_v1/summary.json').read_text())
    transitions={};used=set()
    for cohort in ['pilot','stress1']:
        rows=json.loads((OUT/f'joint_paw_assignment_v1/{cohort}_observations.json').read_text());used.update((r['session'],r['frame']) for r in rows)
        transitions[cohort]=dict(both_accepted=sum(r['joint'] is not None and r['merged_top3'] is not None for r in rows),
            actual_improved_over1px=sum(r['joint'] is not None and r['merged_top3'] is not None and r['joint']<r['merged_top3']-1 for r in rows),
            rejected_not_corrected=sum(r['joint'] is None and r['merged_top3'] is not None for r in rows),
            promoted=[dict(session=r['session'],frame=r['frame'],camera=r['camera'],id=r['id'],joint_error_px=r['joint']) for r in rows if r['joint'] is not None and r['merged_top3'] is None])
    plan=json.loads((OUT/'plan.json').read_text());m=json.loads((OUT/'dlc_dataset_manifest.json').read_text())
    excluded={(r['session'],r['frame']) for r in json.loads((ROOT/plan['exclusion_manifest']).read_text())['cases']}
    excluded.add(('AZR2L2',584)) # Newly suspected manual semantic groupSEM01; historical metrics unchanged.
    remaining=sorted({(e['session'],e['frame']) for e in m['entries'] if e['role']=='validation'}-used-excluded)
    predeclared=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),groups=[dict(session=s,frame=f) for s,f in remaining],
        settings_source='multipeak_geometry_v1/settings.json',primary='merged_top1',comparators=['weighted_robust_floor4','merged_top3','joint'],
        policy='Freeze all settings. Paw-only actualLOO, per-keypoint coverage and severeerrors. Existing teacher development data, NOT independenttest; only new to geometry pilot. No images/labels from lockedholdout.',
        exclusion='Existing24reviewgroups plus newSEM01wholeframe; exclude before evaluation. No historical metric changes.',train_allowed=False)
    (OUT/'remaining_development_geometry_predeclaration.json').write_text(json.dumps(predeclared,indent=2))
    report=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),multipeak=multi,joint_assignment=joint,transitions=transitions,
        conclusions=[
            'Naive multi-peak increases accepted paws but substantially worsens tails of manual2D error. Stress top1coverage39.3%,P9030.81px vs top3coverage61.5%,P90138.07px. Same accepted46observations nearly unchanged, so new coverage carries new failures.',
            'Joint uniqueness reduces stress accepted72->58 and P90138.07->33.52px mainly by rejecting14points, not correcting common predictions. No>1px improvement on any commonly accepted point in either cohort.',
            'Joint inference also promotes a previously unknown pilot point to157.29px error(AZR2L2f129camDk15); visibly lands on body instead of annotated paw. Do not use uniqueness as proof of identity.',
            'Conservative mergedtop1 stays a diagnostic candidate, not acceptedpseudoGT: severe errors remain and coverage is only~39% of manual paw observations. Comparison to oldconsensus changes candidate merging/noise/gates, not merely topK.',
            'Heatmap oracle recoverability did NOT translate to real automatic correction. Stop expandingnaive top3/joint into densevideo. Broaden frozen development validation before further changes.'
        ],visual_review='Windows Downloads/MouseGeoForge-TeacherTemporal-20261001/joint_paw_assignment_v1_review_v2. Older gain-named panels had zeroactualgain and are explicitly superseded.',
        next_actions=[
            'Use remaining_development_geometry_predeclaration.json for one frozen evaluation on remainingdevelopmentframegroups; cacheallteacherpredictionsalreadyavailable. Minimum3views/actualLOO and same thresholds; no tuning.',
            'Compare conservative mergedtop1 to uniform weighted framewise and optional unchangedtop3/joint on same pawobservations. Report per-camera/per-keypointcoverage,severeerrors and source-groupcounts; this is NOT independentanimaltest.',
            'Audit newly suspected semanticlabels separately; SEM01nowexcludedfromnewgeometryevaluationonly, originalcoordinates/historicalresultsunchanged. No training or productionlabels rewritten.',
            'If conservative reconstruction still fails broadly, prioritize evidence on anatomicalidentity/observability and teacher supervision rather than additional ad-hoc gates or smoothing. Continue untildeadlinewithboundedusefulwork, no newexperimentafter07:21:52UTC.'
        ],next_group_count=len(remaining),locked_holdout_opened=False,original_labels_or_calibration_changed=False,pseudo_gt_released=False)
    p=OUT/'heartbeat_20261002_0205_decision.json';p.write_text(json.dumps(report,indent=2))
    write_state(stage='multipeak_and_joint_routes_not_accepted',pid=None,active_run=None,mode='idle_between_heartbeats',job_deadline_utc=None,
        latest_decision=str(p),latest_result=str(OUT/'joint_paw_assignment_v1/summary.json'),next_action=report['next_actions'][0],pseudo_gt_released=False)
    print('remaining_development_groups',len(remaining))

if __name__=='__main__':main()
