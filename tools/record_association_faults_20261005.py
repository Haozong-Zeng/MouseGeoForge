"""Persist counterexample and primary-source-informed association direction."""
import datetime as dt,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];CYCLE=ROOT/'output/singleview_fusion_cycle_20261005'
DEST=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005')


def main():
    r=json.loads((CYCLE/'association_logit_fault_audit_v1/results.json').read_text());assert r['source_unchanged']
    note='''

## 2026-10-05 logit fault injection: unique-query association is unsafe

This is an output fault test, NOT natural missed-detection frequency, RGB occlusion, or new generalization accuracy. Frozen64-image dual predictions retain all6queries and original RGB/boxes/point probabilities. A copied logit tensor preserves one original confident query by a fixed prediction-only rule; all other logits become-20. No GT identity/count chooses the retained query. Evaluation follows the entire forward pass.

| Fixed intervention | Mixed reliable identities, scenes/16 | Different-identity/reliable camera pairs | Fusion-eligible points | Points changed versus local |
|---|---:|---:|---:|---:|
|Highest confidence|11|32/67|320|116|
|Leftmost predicted box|14|39/67|320|127|
|Alternating confidence rank by physical-camera parity|8|16/52|320|140|

Identity audit requires original predicted-box assignment IoU>=0.5; other views are unknown. Original fixed and perturbed-logit operational assignments agree in all192 intervened images. Thus assignment switching does not explain this result. Default-off,1/2-view local fallback, order/duplicates, all-invalid support and unchanged18untargetedpoints pass; input file hash is unchanged. CPU5.16s, zero encoder calls/training.

Concrete visual counterexample: first scene highest-confidence keeps query4 in all four views, but posthoc animal IDs are1,0,1,0. In top view0 the retained box belongs to the brown mouse; composite point17 jumps onto the black mouse. That scene's20 retained reliable points worsen23.322->36.126px versus local, with worst added error95.814px. Across all retained IoU-qualified points the mean21.878->21.881px nearly cancels improvements and harms, so the average obscures the failure. No automatic model promotion.

The feature builder currently gates on the number of finite available camera supports, not an identity test or an absolute geometric acceptance threshold. Strong-looking epipolar evidence and identical query indices do not establish one animal. The single-subject development improvement remains scoped to that assumption; this prototype must not be advertised as general multi-mouse fusion.

### Primary-source research and proposed next step

- [Dong et al., MVPose project](https://zju3dv.github.io/mvpose/) separates cross-camera association from pose recovery, combining appearance and geometry with multi-way consistency and allowing missing/false detections without knowing the subject count. This is the closest short-term design reference.
- Inspected the official [geometry affinity code](https://raw.githubusercontent.com/zju3dv/mvpose/master/src/m_utils/geometry.py), [SVT matcher](https://raw.githubusercontent.com/zju3dv/mvpose/master/src/models/matchSVT.py), and [partial-assignment projection](https://raw.githubusercontent.com/zju3dv/mvpose/master/src/models/match_solver.py): same-camera blocks are constrained and matching is not simply equal-index linking. Do not copy its17-human-joint assumptions or scene-wide distance normalization blindly; use our verified camera convention and explicit invalid-camera behavior.
- [MvP](https://arxiv.org/abs/2111.04076) uses hierarchical pose queries and camera-guided projective attention for direct3D prediction. It is a longer-term learned3D-instance route, not a small change to this selector.
- [MVGFormer](https://github.com/XunshanMan/MVGFormer) iterates coarse3D queries,2D projection/refinement and triangulation, with evaluations across camera arrangements/counts. This is relevant to variable-view design but does not provide a single-view triangulation solution; retain a genuine image-only fallback.

Our next step is evidence first, not another pose-head sweep: recover the existing21dualTRAIN scenes' full4view prediction/calibration context, then audit cross-query affinity from corresponding-point epipolar distances and independent appearance features. Report whether frozen semantic tokens merely preserve query-index bias. Same/different-instance labels are permitted only for TRAIN calibration and posthoc held-out development evaluation, never inference routing. Do not force all views into matches; eventual association needs explicit rejection, at-most-one detection per physical camera per group, and multi-view consistency. Appearance evidence on black/brown synthetic mice alone does not prove same-color or real-mouse identity performance. No production integration until association is tested.

Windows visual report: `C:/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005/association_logit_fault_audit_v1/index.html`.
'''
    doc=ROOT/'docs/singleview_gated_fusion_geometry_direction_20261005.md';text=doc.read_text(encoding='utf-8')
    if '## 2026-10-05 logit fault injection:' not in text:doc.write_text(text+note,encoding='utf-8')
    shutil.copy2(doc,DEST/doc.name)
    next_action='''association_logit_fault_audit_v1 COMPLETE, CPU5.16sec, noactivejobs. tools/audit_association_faults_20261005.py +review_association_faults_20261005.py. Source64dualimages unchangedSHA, noencoder/training. Threefixed prediction-only copiedlogit interventions retainhighestconfidence/leftmostbox/alternatingconfidencerank(physicalIDparity); suppressothers=-20, all6queries/boxes/XY/prob/tokens/RGBunchanged. Not naturalocclusion/detectionfrequency. Both fixedoriginal and operationalperturbedscore assignments agree192/192images. IoU>=.5 required for posthoc identity elseunknown.
Results highest/leftmost/alternate mixedidentityscenes11/14/8 of16, mismatchedreliablepairs32of67/39of67/16of52, all320pointsfusioneligibleeach; changedvslocal116/127/140. Defaultoff/order/dup/V1V2/allinvalid/18untargetedexact. Concrete firstscenehighest: keepsq4everycamera but physicalanimal IDs1,0,1,0; visuals topview0 retainedbrown point17jumpsblack,sideview1q4isblack. Firstscene20retainedvalidpoints local23.322->composite36.126px,worstharm95.814px. Overallretained270IoUqualifiedpoints local21.878->21.881mean hides18harms>16px. No generalmultiassociationclaim, no promotion. Windows association_logit_fault_audit_v1/index.html12panels; inspectedfirsttop/side. No newthreshold patched.
Read docs/singleview_gated_fusion_geometry_direction_20261005.md latestsection forsourceURLs. ResearchofficialMVPose DongCVPR2019/T-PAMI2021 geometry+appearance+cycleconsistentpartialmatching; inspected rawsrc/m_utils/geometry.py,src/models/matchSVT.py,match_solver.py. Their human17joints/scene-normalized affinity not blindlyportable. MvP https://arxiv.org/abs/2111.04076 direct3Dhierarchicalqueries/projectiveattention andMVGFormer https://github.com/XunshanMan/MVGFormer projection/refinement/triangulation longerterm. Our current featureeligibility counts finiteavailablecameras, not calibratedidentityverification. Equalqueryindex notsameanimal.
NEXT meaningfulsmallstep: recover full4views for all21 EXISTING dualTRAIN scenes (currentlydual_train_candidate_audit_v1/train_inputs.pt42images2views) and build cross-query affinity diagnostic before any newtraining/fusiondeployment. Source singleview_fusion_inputs_seed17_v1/train.pt1223 hasmetadata expected_instances/record/view; select all21distinctnonreal expected_instances2scenes roletrain, maintain split. Full existing4views via load_scene; no newrenders/actions/VALtrain. Reuse audit_composite_allviews_20261005.py patterns but itsmain currentlyhardcodesvalidation.pt/rolevalidation/OUT anddoesmatching; create clearlyseparateTRAINextraction orparameterizerole withoutoverwriting anyresults. Runtime all6queries/138keypointtokens, calibrationfullbatchcommon_scale andK/Eprojectionverify asprior. Can freshencode84images cheaply underGPUlock, or reuse oldpair plus42new with sourceparity. No GT inforward. Save groups.pt prediction fieldsxy/boxes/objectness_logits/tokens[V6,5,256]/candidate_memory[V6,5,6,256]/candidateXY/mass/prob, projections,targetsseparate,actualRGB/sourcehashes. TRAINandVALsceneIDsdisjointassert.
Then a prediction-only affinity utility over everyconfidentquery fromeveryphysicalview. Initialdiagnostics separate (not weightedVALsearch): median/mean symmetric epipolardistance overfixedreliable5baseXY, and cosine similarity ofcorresponding frozen semantic tokens, and candidate0local-memory descriptors. Guard finite/calibration/pixels, exclude samephysicalcamera. Boxcenter epipolar notprimaryanatomicalpoint. Determine whether semanticfeatures mostly encodequeryindex ratherthanmouseidentity (counterexampleq4switches). TRAINsame/differentinstance labels derived onlyAFTERforward from IoUqualified predictedbox matching, not GTcrops/count/queryselection. Assess pair-ranking/separation and mutualnearest candidates; labelambiguousdetectionsunknown, coverage+FPexplicit. First diagnostic canuse21TRAIN2existingviews withload_itemcameras butfull4betterforcycleconstraints; preserve cameracommon gauge/nativepixels768based.
Do not yet committhresholds or fusebasedonbestVAL. Any fittedcalibration mustuseTRAINonly andfrozenbeforeVAL, no reuseheldoutGR2UR2UR3. Mutualnearest alone cannotrejectsingletonwrongidentity; eventualpartialmatching needs absoluteevidence/rejection plus atmostoneperphysicalcamera/cycleconsistency. Do not force2matches or useGTcount; predicted2naturallyisnotproof2correct. Evaluate natural64VAL and separatedlogitfaults, laterreal/samecolor needed. Learnedgraph/global3Dqueryroute canbeplanned butnotclaimedimplemented/reproduced. Existing defaultoff prototype and labels remainuntouched.
Protect originals24excluded135pendingR12oldmasktailabdomenlockedGR2UR2UR3dirtytree; no productionpromotion, no huge downloads. Deadline remains train18:11:41UTC/hard18:41:41UTC2026-10-05, heartbeatpymousestudio15min, sharedGPUlock; checktime/process beforejob. Savevisuals+hashes/doc/handoff, donotrerunoldrecorders.'''
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='cross_query_affinity_diagnostic_next',active_job=None,active_pid=None,next=next_action,updated_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    state['completed']=list(dict.fromkeys(state['completed']+['association_fault_visual_review','primary_multiview_association_research']))
    (CYCLE/'state.json').write_text(json.dumps(state,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    plan=json.loads((CYCLE/'plan.json').read_text());plan['next_command']=next_action
    (CYCLE/'plan.json').write_text(json.dumps(plan,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    index=DEST/'index.html';page=index.read_text(encoding='utf-8');link='<p><a href="association_logit_fault_audit_v1/index.html">重要反例：漏检故障注入暴露跨鼠错误融合</a></p>'
    if link not in page:index.write_text(page+link,encoding='utf-8')
    print(state['stage'])


if __name__=='__main__':main()
