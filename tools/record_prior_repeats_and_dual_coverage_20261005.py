"""Persist repeat limits and next controlled dual-TRAIN inclusion hypothesis."""
import datetime as dt,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';DEST=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005')


def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def main():
    next_action=(
        'Completed bounded_prior_viewstress_v1, bounded_prior_selector_seed_repeats_v1 (session23084 exited), visualrepeatreview, dual_train_candidate_audit_v1. Noactivejobs. Read latest docs/singleview_gated_fusion_geometry_direction_20261005.md and Windows bounded_prior_repeat_review_v1/index.html. '
        'Stress recomputesfeatures from24validationcachedscenes. Full/reverse/duplicates exact, V1/V2/allinvalid exactoriginalfallback. Targetcentric3 views meansfixedtarget+2lowestphysicalIDs. Geometryreal28.33/bottom41.20 vsfull26.32/39.70 andbase33.25/48.46. All7leaveonecameraouts realbetterthansameobservedbaseline; coverage explicitlyreported. NeverchoosebestsubsetonVAL. stressscript/results/predictions/points saved. '
        'Additionalselector-training seeds29/43 ONLYnewheadinitialization/schedule; frozenbaseline17 andcandidatesfixed. Botharmssameinit/schedule800. Notindependentencoder,split,videoorbaseline seedrepeat. seed29unary/geometryVALreal29.292/27.852bottom39.886/42.215;seed43real31.141/28.006bottom38.414/42.122. seed17real30.438/26.324bottom41.197/39.704. Mean+-sampleSDrealunary30.290+-.933 geometry27.394+-.930;bottomunary39.833+-1.392 geometry41.347+-1.424. Geometryoverallbenefit3/3butbottomworse2/3, notpromoted. All3dualunaryperqueryerrors25.657,24.682,24.779 worsebase23.299. Visualrow15 all3collapse15/16ontosamevisiblepaw. Donotaddsimilarheadsorblindlyincreasefailedgeometrytraining. '
        'Recovered existingdualTRAIN42images21scenes in dual_train_candidate_audit_v1/train_inputs.pt, CPU/GPU replaymax<2e-5. Fieldscompatiblepackedtrain fromlocal_candidate_selector_pilot_v1: inputs{semantic[N6,5,256],localFP16[N6,5,6,256],scalars[N6,5,6,9],geometryzeros,candidate_xy[N6,5,6,2],baseline_xy[N6,5,2],eligiblefalse},base_xy[N6,23,2],boxes,logits,gt_xy,point_valid,gt_boxes,box_valid,instance_valid,pixels,metadata{scene,view,kindsynthetic,roletrain,source_row,image_index}. NOtarget_prob/train_valid yet. Thisis42alreadyexistingTRAINnotnewrender. Validationdual32 in dual_unary_diagnostic_v1/inputs.pt, scenesdisjoint butNOTindependentvideo. Native/source/script hashes inresults. '
        'TRAINdual84GT84pred67TP; valid420nose/footpts baseline22.235, GT-assistedbestof6oracle11.645, prior-admissibleoracle15.033. VAL32 dualbase23.299/alloracle12.119/priororacle15.445,64GT64pred45TP. Oracleonlypostforwarddiagnostic usesGT(notprediction/notnewannotation). Nativeprior candidates allowedbaseline or logmass+2>.001, score>=.5 query only. Correctcandidatesoftenexist; learning/instanceassignment gap, notonlyresolution. '
        'NEXT one controlled DATA inclusion experiment, notarchitecturevariant: independent per-query unary PriorCandidateSelector bound2, frozenbaseline17/candidates. Compare singlecohort-only vs mixed existingdualTRAIN42, sameinitialseed17 and800steps/optimizer/lr/CEsigma16. Botharms mustuse identicalper-queryscore>=.5 gate (noothercamera requirement,noGTcount) and recompute GTsofttargets/train_valid afterfrozenforwardboxmatching; oldpackedtrain_valid includesmultivieweligibility soDO NOTreuse. All6queries retained, noGTidentity inforward, no realcurve/fake3D. Singlecontrol4realbottom+4otherreal+8synsingle eachstep. Mixedkeepfirst14sameimages(4+4+6) andreplace last2synsingle with2dualTRAIN: exactly2/16=12.5%dualimagebudget, prerecord. UseSOURCE originalseed17schedule forcommonindices, independentdual RNG seed17042 predeclared so otherimage scheduleunchanged. Equal-image CE averagesvalidpointsoverqueries, so dualimage10pointsnotdoubletotalweight. NoVALID selection, fixedfinal800; TRAIN42 small~38effectivepasses dual, watchoverfitting. Originalheadunary independentVAL/dualpreds servehistory butmatchednewgatecontrolrequired. '
        'Evaluate singleVAL80+dualVAL32 afterbothtraining; originalbaseline vsnewcontrol vsmixed. Reportreal/bottom/synsingle/syndual allpoints andgatedPCK, changedharm, unchangedboxcounts, rawallconfidencequeryoverlays/crossinstance-nearerdiagnostic. Tail/backunchanged. Failurecanmeanneedbox/mask-conditionalassignment, notjuststeps; deferdecisiontoevidence. DoNOTtraingeometryondoubleunknowncrosscameraIDs. '
        'Timepolicyheartbeat15minACTIVE, trainingstop18:11:41UTC/hard18:41:41UTC;sharedGPUlock output/teacher_temporal_cycle_20261001/gpu_job.lock. Protectannotations24excluded135pendingR12oldmasklockedGR2UR2UR3dirtyworktree/defaultmodel. Do notrerunoldrecorders. Updatecoherentstate.')
    note='''

## 2026-10-05 actual-view stress, selector-seed repeats and dual TRAIN coverage

Recomputed geometry on actual subsets, not sliced full-context tensors. Reversing and exact camera duplicates preserve predictions exactly. V1, V2 and all-invalid support fall back exactly. Fixed target-centric V3 gives real28.33/bottom41.20 versus full26.32/39.70 and baseline33.25/48.46. All seven leave-one-camera-out tests retain real improvement versus the same observed-target baseline. Coverage and missing target views are explicit; no selecting optimal cameras from these numbers.

Additional seeds29/43 change only selector initialization and sample schedule, with frozen original baseline17. Both arms share each seed's initialization/schedule, 800 steps. Geometry improves real mean in all three, but worsens bottom versus unary in two of three:

| Selector seed | Unary real | Geometry real | Unary bottom | Geometry bottom |
|---|---:|---:|---:|---:|
|17|30.438|26.324|41.197|39.704|
|29|29.292|27.852|39.886|42.215|
|43|31.141|28.006|38.414|42.122|

Mean ± sample SD: real unary30.290±.933, geometry27.394±.930; bottom unary39.833±1.392, geometry41.347±1.424. Not an external video/split/backbone-seed validation. Visual row15 shows all three geometry heads collapse points15/16 on one paw. Dual unary errors25.657/24.682/24.779 all worsen original23.299. No promotion or per-camera model picking.

Recovered the existing 42 TRAIN dual images (21 scenes) into `dual_train_candidate_audit_v1/train_inputs.pt`, no additional frames/renders or labels. Prediction-only probability/candidate construction first, then GT-only candidate-coverage diagnostic. TRAIN84GT84pred67TP; VAL64GT64pred45TP. Train/validation scenes disjoint, but not independent source videos.

| Dual reliable point error | Original mean | Optimistic best-of-six oracle | Prior-admissible oracle |
|---|---:|---:|---:|
|TRAIN420 points|22.235|11.645|15.033|
|VAL320 points|23.299|12.119|15.445|

Oracles use GT post hoc and are not achieved inference or annotations. Correct candidates often exist, motivating one controlled TRAIN-data inclusion experiment before new architectures: fixed unary prior-bound2 head,800steps, independent predicted-query score>=.5 gate in both arms. Compare original 4real-bottom+4other-real+8single-synthetic schedule with same first14 samples plus2 existing dual images (12.5% image budget), using preregistered independent dual RNG17042. Equal-image loss, recompute supervision masks (old masks include multiview eligibility), no GT count or identities in forward. The small dual set is heavily reused; do not assume more steps solves generalization. Evaluate single/dual development subsets and raw instance errors, no VAL-selected checkpoint.

Windows report: `C:/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005/bounded_prior_repeat_review_v1/index.html`. Original labels/locked sources/default model unchanged.
'''
    doc=ROOT/'docs/singleview_gated_fusion_geometry_direction_20261005.md';text=doc.read_text(encoding='utf-8')
    if '## 2026-10-05 actual-view stress, selector-seed repeats and dual TRAIN coverage' not in text:doc.write_text(text+note,encoding='utf-8')
    shutil.copy2(doc,DEST/doc.name)
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='dual_train_inclusion_control_next',active_job=None,active_pid=None,updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),next=next_action)
    for key in ['bounded_prior_actual_view_stress','bounded_prior_three_selector_seed_review']:
        if key not in state['completed']:state['completed'].append(key)
    write(CYCLE/'state.json',state);plan=json.loads((CYCLE/'plan.json').read_text());plan['next_command']=next_action;write(CYCLE/'plan.json',plan)
    index=DEST/'index.html';page=index.read_text(encoding='utf-8');link='<p><a href="bounded_prior_repeat_review_v1/index.html">最新：真实视角压力、三选择头种子与双鼠训练候选分析</a></p>'
    if link not in page:index.write_text(page+link,encoding='utf-8')
    shutil.copy2(CYCLE/'dual_train_candidate_audit_v1/results.json',DEST/'dual_train_candidate_coverage.json')
    print(state['stage'])


if __name__=='__main__':main()
