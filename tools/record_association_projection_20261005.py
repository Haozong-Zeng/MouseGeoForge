"""Record completed association pilot; do not replay older recorders."""
import datetime as dt,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';DEST=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005')
NOTE="""

## 2026-10-05 supervised association projection and conservative partial groups

Completed matched RGB/gray1024->64 bias-free projections (65,536parameters), seeds17/29/43,800steps each. Main encoder/detector/pose frozen, cached predicted-box cropCLS only.16TRAIN fitting scenes plus5internal calibration scenes; outer16development scenes have a disjoint action source but shared render assets. Internal calibration is same source and main pose model already saw it. Unknown nodes enter forward but not loss positives/negatives.7projection tests and7partial-group tests pass. Six fits total20.5seconds (cached small head only).

The near-identical final RGB/gray fitting losses were checked: feature arrays do not alias, max difference0.7492. Multi-positive contrastive fit lower bound0.72492577 is reached by800steps; fit identityAUC reaches1 around100steps while calibration loss worsens later. Do NOT extend training. Retained fixed800endpoints rather than selecting steps on development.

Development RGB identityAUC0.930/0.924/0.927, gray0.735/0.746/0.756. RGB mutual known precision87/91,86/92,85/91, but63/154,62/154,64/155mutual pairs unknown. Gray removes chroma, not brightness; this is not same-color validation.

Calibrated partial grouping on5reservedTRAIN scenes:43knownnegative and43positive cross-camera edges. Predeclared1% empirical negative budget floors to0; strict distance<minimum negative. Mutual nearest, >=3distinct-camera complete cliques, reject overlaps. Unknown nodes retained. All6models plus rawRGB/gray evaluated after freezing thresholds. RGB17/29/43 natural development accepted5/9/10groups:5/7/7fully-known pure,0knownmixed,0/2/3unknown;15/28/32of128nodes assigned. Gray1/2/0groups mostlyunknown. Rawcosine and priorstrictgeometry accepted0groups. Small correlated calibration sample does NOT establish a1% safety bound. Fault tests observed0knownmixed accepted groups but unknowns and very low coverage remain. No pose changed.

Visual review: seed17scene4 brown mouse associated views0/2/3 at q3/q3/q4, overlapped sideview1 rejected. Unknown seed43scene11 topview boxes truncate body portions; visually plausible identity does not override unknown evaluation. Firstscene highest-confidence fault kept different mice across cameras; all four rejected. Curves confirm saturation/calibration deterioration. Illustrations are post-hoc categories, not random samples.

Next: default-off association-gated offline candidate fusion, fixed historicalseed17/800 and frozen threshold. No new training or threshold adjustment. Compare original/current local/unsafe unique-query composite/grouped local/grouped geometry/grouped composite. Keep unassigned nodes and18non-target points unchanged in grouped mode. Evaluate all original GT assignments only after forward, plus assigned/IoU-qualified subsets and unknown coverage. Quantify foot localization, P95/worst harms, not only association accuracy. Negative results stay negative; no production promotion.

Windows report: C:/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005/association_projection_review_v1/index.html
"""
NEXT="""
Completed association_projection_pilot_v1 session49051 exit0. tools/train_association_projection_20261005.py, module models/association_projection.py7tests.65536param sharedlinear1024->64 L2 input/output defaultoff exact, noGT/query/camera/count in forward. Within-scene knownsameIDdifferentcamera positive; differentknownID negative; unknown excluded onlyloss. TRAIN16fit/cal5indices[5,7,12,14,19], stage-specific same-action-source holdout. Seeds17/29/43 RGBgray800steps identicalinit/schedule perseed. Mainmodels frozen. All6fit beforeVAL. Checkpoints rgb_seed17_step0800.pt etc state/seed/arm/step. inputs.pt roles->{features{rgb,gray}[N8,1024],labels,views,queries,scenes}; embeddings.pt keys rgb_seed17_{raw,initial,trained}_{train,validation} etc. curves.json per100 fit/cal only. results.json allcontrols/seeds. FitAUC1 by100,loss.724926 theoreticalfloor.72492577,calibration worsens after100-200. NOextra training. RGBgraynotaliased maxdelta.7492. VALRGB AUC.930/.924/.927 gray.735/.746/.756. MutualRGBknown87/91,86/92,85/91 but63/154,62/154,64/155unknown.

Completed tools/evaluate_learned_partial_association_20261005.py CPU. learned_partial_association_v1/calibration.json frozen5TRAINcal43negative43positive.1%empiricalnegativebudgetfloor0,strictcosdistance<minnegative. RGB17cutoff.19223928451538086,29 .2092090845,43 .2107180953. All6+rawcontrols frozenbeforeVAL/fault. conservative_camera_groups from partial_camera_association.py mutualnearest/>=3distinctphysicalcam cliques/rejectoverlap; noGTcount,unknownnodesretained. RGB17/29/43natural groups5/9/10 pure5/7/7 mixed0 unknown0/2/3 assigned15/28/32of128;rawcos0;gray1/2/0mostlyunknown. Fault3modes0knownmixed butunknownremain+lowcoverage; notcertifiedsafe. results.json summarykeys rgb_seed17_trained/validation/natural etc. details have model/role/mode/group_index/scene/node_indices/node_list/identity_diagnostic/groups[{nodes,identities,status}]/edges. DiagnosticGT fields mustNOT enterforward orfiltergroups.

Reviewed Windows association_projection_review_v1/index.html: curves, actual accepted RGBscene4 brownq3/q3/q4 acrossviews0/2/3 rejectingoverlapview1, unknownRGBscene11partialtopboxes, faultscene0allrejected. 14tests pass. tools/review_association_projection_20261005.py report. Noactivejobs. DoNOTrerunrecorders.

NEXT implement default-off groupedfusion adapter andtests then offlineaudit nolearning. FixedhistoricalRGBseed17final800 + frozen.1922392845cutoff NOTbestVALseed. Recompute prediction-only groups from cropCLS->projection->cosdistance->conservative_camera_groups; doNOT use saveddiagnosticstatusasgate. Adapter receives only physical(view,query) correspondences; rejects invalidindices/duplicatephysicalview/overlappinggroups/<3calibratedphysicalcams/nonconfidentnode. Eachacceptedgroup: subsetviews preservingALL6predictiontensors, copylogits suppressothers-20 keepselectedoriginalconfidentlogit, call BoundedCandidateFusion with localfalsefusiontrue or true/false ortrue/true. CopybackONLYacceptednodes KP=[0,14,15,16,17]. Allunassigned/non-target18points/boxesexactoriginal. NoGTcounts,crops,matches in inference. UnknownidentitiesmustnotbeGTfiltered. V1/V2/off exactoriginal in groupedfusion; independently test local path. Testgrouporder/view-querypermutation,invalidcamerafallback,overlapreject,noinplacemutation.

Sources dual_allscene_association_audit_v1/groups.pt16VAL and dual_train_allview_affinity_context_v1/groups.pt21TRAIN, eachrecord.views/projectionsnp[V3,4]/pixels[W,H]/imagesrelative/evaluation_data. Predictions xy[V6,23,2],boxes[V6,4],logits[V6],tokens[V6,5,256],probability[V6,5,48,48],candidate_xy[V6,5,6,2],candidate_mass[V6,5,6],candidate_memory[V6,5,6,256]. Heads via tools/audit_composite_candidate_cache_20261005.py::load_wrapper CPU, boundedunary/geometry seed17. cross_query_affinity_diagnostic_v1/evidence.pt entriesrole/group,nodes[{view_index,view,query}],GTdiagnosticseparate; cropfeatures mapping inputs.pt nodes sameorder. Runtimegrouping may reconstructnodesfrompredictions conf>=.5; do not hardcode2count8nodesinadapter (pilotdatasetsnaturally8).
Fault choose from audit_association_faults_20261005.py::choose(p,views,mode), modes highest_confidence/leftmost_box/alternating_confidence_rank. Suppressotherslogits-20 andsubsetfeature/nodesBEFOREassociation; notrealocclusion. Compare baseline/currentlocal/unsafecomposite/groupedlocal/groupedgeometry/groupedcomposite. Onlyafterforward fixoriginalGTboxassignment plusoperational faultassignment. Fullallmatchedmetrics+assigned+IoU50knowncoverage/unknown; footP95/worstharm/changedpoints. No cherry onlyaccepted. RGBoverlaysworstharmandacceptedchanges; evaluate geometryovergroupedlocal notonlyoriginal. No freshencoder needed. Ifhurt, recordnegative and investigate geometryuncertainty, notmorefitsteps.

Sourcecorrection: all21TRAINactionCRIM13 021709_A25_Block22_BCfe1_t,all16VAL021309_A29_Block16_BCma1_t, verified37actualposeplanimages. Distinctactionsource butsharedBlenderassets, notreal/samecolor; earlier samevideosclaimwrong. Keepcorrection. ProtectR12oldmask24excluded135pendingtailabdomenlockedGR2UR2UR3dirtytreeproductionoff.
Deadlineunchanged training18:11:41UTC hard18:41:41UTC2026-10-05. Existingheartbeat15minACTIVE, noreset12h. SharedGPUlock output/teacher_temporal_cycle_20261001/gpu_job.lock. Checktime/status/PIDbeforejobs. Pauseofficialautomationatharddeadline. Noactivejobs.
"""

def main():
    doc=ROOT/'docs/singleview_gated_fusion_geometry_direction_20261005.md';content=doc.read_text(encoding='utf-8')
    if '## 2026-10-05 supervised association projection' not in content:doc.write_text(content+NOTE,encoding='utf-8')
    shutil.copy2(doc,DEST/doc.name)
    now=dt.datetime.now(dt.timezone.utc).isoformat();state=json.loads((CYCLE/'state.json').read_text())
    state.update(stage='association_gated_offline_fusion_next',active_job=None,active_pid=None,next=NEXT,updated_utc=now)
    state['completed']=list(dict.fromkeys(state['completed']+['association_projection_7tests','association_projection_convergence_audit','learned_partial_association_calibrated_control','association_projection_visual_review']))
    (CYCLE/'state.json').write_text(json.dumps(state,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    plan=json.loads((CYCLE/'plan.json').read_text());plan['next_command']=NEXT
    (CYCLE/'plan.json').write_text(json.dumps(plan,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    index=DEST/'index.html';page=index.read_text(encoding='utf-8');link='<p><a href="association_projection_review_v1/index.html">关联头RGB/灰度三种子、收敛曲线与拒绝式分组</a></p>'
    if link not in page:index.write_text(page+link,encoding='utf-8')
    print(state['stage'],now)

if __name__=='__main__':main()

