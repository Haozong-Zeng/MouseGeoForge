"""Evidence-driven handoff: query-biased appearance and partial matching next."""
import datetime as dt,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';DEST=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005')


def main():
    r=json.loads((CYCLE/'cross_query_affinity_diagnostic_v1/results.json').read_text())
    assert r['results']['train']['images']==84 and r['results']['validation']['images']==64
    note='''

## 2026-10-05 cross-query evidence: current token similarity is not re-ID

Recovered all21 existing dual TRAIN scenes' four-camera context:84 fresh independent RGB passes,27.35s total including load, max synthetic projection0.000082px. Existing42view outputs replay within tolerance; TRAIN and development scenes are disjoint. No new rendering/actions/training. `dual_train_allview_affinity_context_v1/groups.pt` preserves predictions and camera geometry separately from evaluation targets.

New `mousegeoforge/association_evidence.py` computes four independent prediction-only channels: corresponding5point epipolar mean/median in input pixels, corresponding semantic-token cosine, and original-coordinate local-memory cosine. It does not match or fuse and accepts no labels/counts. Seven tests pass (analytic geometry, camera/query permutation, duplicate-camera handling, invalid camera, empty/single view and input validation).

| Evidence | TRAIN identity AUC | Development identity AUC | Development same-query AUC | Mutual known correct / known | Unknown / all proposed |
|---|---:|---:|---:|---:|---:|
|Epipolar mean|0.820|0.881|0.383|93/98|64/162|
|Epipolar median|0.822|0.873|0.402|92/99|63/162|
|Semantic cosine|0.433|0.420|0.945|43/106|79/185|
|Local cosine|0.458|0.439|0.844|38/88|58/146|

Identity labels are posthoc box matches with IoU>=0.5. TRAIN135/168 and development97/128 nodes qualify; unknown nodes remain candidates and are included in coverage counts. Both AUCs are diagnostic, with no fitted weights, thresholds or score combinations. A frame split here is not an independent-video test.

The semantic mutual matches select identical query indices on185/185 development edges, despite only43/106 known edges linking the same animal. This is strong query-index bias, not a validated appearance descriptor. The first-scene visual matrices show high semantic similarities on different-animal equal-query pairs. Local point memory also fails, plausibly affected by inaccurate selected positions and view-dependent encoding; that mechanism is an inference, not proven causation. Reject direct cosine as a re-ID gate; do not simply invert the score based on development AUC.

Geometry is useful but not sufficient: mutual nearest still includes5 known wrong edges and64 unknowns. A singleton wrong candidate is trivially mutual. Next test a conservative partial matcher with an absolute cutoff calibrated on TRAIN only, camera uniqueness and clique consistency. Predeclare a1% empirical TRAIN different-identity-pair false-acceptance budget for mean epipolar distance (not a real-world guarantee), freeze before evaluating development/fault injection. Require at least3 distinct cameras before any future fusion. Report rejection and coverage even if it yields no usable groups; no threshold loosening to recover a preferred score.

No production changes, learned association model, or multi-mouse fusion deployment. A future dedicated predicted-ROI appearance descriptor or supervised association branch is a separate hypothesis, requiring same-color/real validation, not merely black/brown discrimination.

Windows report: `C:/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005/cross_query_affinity_diagnostic_v1/index.html`.
'''
    doc=ROOT/'docs/singleview_gated_fusion_geometry_direction_20261005.md';text=doc.read_text(encoding='utf-8')
    if '## 2026-10-05 cross-query evidence:' not in text:doc.write_text(text+note,encoding='utf-8')
    shutil.copy2(doc,DEST/doc.name)
    next_action='''Cross-query affinity diagnostic COMPLETE, noactivejobs. TRAINextraction session42506/PID3771414 exited10:48:32UTC. New dual_train_allview_affinity_context_v1 has21TRAINscenes84freshindependentRGB,27.35stotal,maxprojection8.232e-5px,old42viewparity passes,allviews2queries soexistingwrapperabstains. OriginalTRAIN153single/etcunmodified. Tools prepare_dual_train_allview_affinity_20261005.py imports audit_composite_allviews_20261005.py nowmain(requested_counts=None,source_role=validation) optionaltrain; source-role explicit. Separate outputs immutable. BothTRAIN/VALgroups.pt structure identical. Trainingframes and developmentframesdisjoint, sameoriginalvideos.
Newmodule mousegeoforge/association_evidence.py,7passing tests tests/test_association_evidence.py. pairwise_affinity_evidence(predictions,projections,physical_views,pixels)->nodes(listview_index/view/query/score),evidence matricesNxN(epipolar_mean_px/median_px,semantic_cosine,local_cosine),finite_point_count. Allconfidentqueries>=.5, dedupsphysicalcam exact elseValueError, noGTinputs. Nodescanonicalphysicalcameraorder. CorrespondingKP[0,14,15,16,17], actualpixelgeometry, min3finitepointsnotvisibilityclaim. Semantic cosine perpointnormalizedthenmean; candidate0localmemory same. Samecamera entriesNaN; invalidcalib geometryNaN butappearancecanfinite. No matcher/fusionyet.
tools/diagnose_cross_query_affinity_20261005.py outputcross_query_affinity_diagnostic_v1: evidence.pt list(role,group,scene,nodes,evidence,identity_diagnostic,ious); train/validation_edges.json (allcrossviewpairs,known flag,same_identity,same_query,evidence); rankings.json andresults. Only AFTER feature/ranking forward assign GTbox/score, knownidentityrequiresIoU>=.5;unknownnodesremaincandidate. TRAIN135of168known,VAL97of128known. No threshold/weightsfit. Sourcehashesmodulehashstored.
TRAIN/VAL identityAUC meanepi.820/.881,medianepi.822/.873,semantic.433/.420,local.458/.439. VALsemantic samequeryAUC.945 andall185mutualedges samequery, only43of106knownsameanimal plus79unknown. Local38of88known+58unknown. Do NOT use thesecosines asre-ID or invert basedonVAL. Geometrymean mutual93of98knowncorrect+64unknown (162total),median92of99+63unknown. Geometrynotguarantee, singletonmutualalwayspasses. VisualfirstTRAIN/VALscene matrices reviewed: highsemanticredcrossanimal equalquery; geometry mostlyseparatesbutoverlap. Windowscross_query_affinity_diagnostic_v1/index.html. No newtraining or checkpointpromotion.
NEXT implement conservative prediction-only partial association as separateopt-in utility, no fusionintegrationyet. Predeclared meanepipolardistance channel (not choosebestVAL), empirical1% TRAINdifferent-identity falseacceptance budget fromTRAINknownpairs ONLY (164negative edges). Calibrate largest threshold whose TRAINwrongaccept count<=floor(.01*Nnegative); handle tiesstrictly, record limitedsample/uncertainty, then freezeprotocolhash BEFORE anyVAL/fault results. TRAINpositivecoverage report, neverloosenbasedonVAL. Could enforce mutualnearest perphysicalcamera and absolute cutoff, then conservative groups requiring >=3 distinctphysicalcameras and complete pairwise support(clique), atmostonequerypercamera, reject conflicting/ambiguous overlappinggroups ratherthan forcegroupcount. Do not mistake partialmatching forknownnumbermice. Deterministiccamera/querypermutation andduplicates tests required; noGTn/count/matchtoforward. No automaticfusionorposeedits inthisstage.
Evaluate on natural84TRAIN/64VAL and isolatedlogitfault64x3 using savedgroups; recomputeevidence afterfaultlogits, no additionalencoder needed. Faultselection logic fromtools/audit_association_faults_20261005.py choose(p,views,mode),originalcopiedlogits others=-20. Inspect identitypurity onlyposthocIoUqualifiednodes, unknowncoverage explicit; naturalgroups/same-animalrecall and retainedsupports, mixed-groupfraction, rejectednodes. Evaluate originalfixedGTbox assignment forfaults, sameasoperationalprevious. Neverclaims1%naturalerrorfrom164trainnegatives. Ifno>=3groups, reporthonestlyanddoNOT relaxVALthresholds. Later assessdedicated predictedROIappearance orlearnedassociationTRAIN supervision asdistinctroute, same-color/realneeded. Currentsemantic/localcosine negative, no newhead-onlypose/heatmap retries.
Docs latestsection andpriorprimaryMVPose/MvP/MVGFormer references containreasoning. Preserveprototypesdefaultoff/production/labels/24excluded135pendingR12oldmasktailabdomenlockedGR2UR2UR3dirtytree. Deadline unchanged18:11:41UTCtrain,18:41:41hard2026-10-05,15minheartbeatandsharedGPUlock. Status currentlyready; CPUaffinitynoGPUlockneededbutnewencoder mustlock. SaveWindowsvisuals/sourcehashes/handoff,donotrerunoldrecorders.'''
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='partial_association_calibration_next',active_job=None,active_pid=None,next=next_action,updated_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    state['completed']=list(dict.fromkeys(state['completed']+['dual_train_full4view_context84','association_evidence_7tests','cross_query_affinity_diagnostic','cross_query_affinity_visual_review']))
    (CYCLE/'state.json').write_text(json.dumps(state,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    plan=json.loads((CYCLE/'plan.json').read_text());plan['next_command']=next_action;(CYCLE/'plan.json').write_text(json.dumps(plan,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    index=DEST/'index.html';page=index.read_text(encoding='utf-8');link='<p><a href="cross_query_affinity_diagnostic_v1/index.html">关联证据诊断：当前语义特征偏向query编号，不是动物身份</a></p>'
    if link not in page:index.write_text(page+link,encoding='utf-8')
    print(state['stage'])


if __name__=='__main__':main()
