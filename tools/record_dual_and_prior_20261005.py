"""Persist dual OOD failure and bounded-prior matched experiment decisions."""
import datetime as dt,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';DEST=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005')


def write(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def main():
    next_action=(
        'Dual unary diagnostic and bounded-prior matched800x2 COMPLETE; session18009/PID3768126 exited. No active GPUjobs. '
        'Read docs/singleview_gated_fusion_geometry_direction_20261005.md latest sections and Windows dual_unary_diagnostic_v1/index.html and bounded_prior_candidate_pilot_v1/index.html. '
        'Dual diagnostic: SOURCEvalidation112 selected32syntheticdual only offlinecohort, all6queries forward/noGTcount. tools/audit_dual_unary_20261005.py reconstructs48probability using SOURCEtokens/coarse and OLDnativeFP32memory->half->float->48 withfull17pose spatialreadout, no1Bencoder; maxexpectedxyreplay5.96e-7. Local6candidates cached dual_unary_diagnostic_v1/inputs.pt compatiblepacked dictionary. Metadata scene/view/source_row/globalimageindex. Nativefeatures andmodelhashes inresults. '
        '32images64GT64pred45TP atscore.5IoU.5, everyimage2preds. unique-query gate neverupdates; per-query gate activatesbothbutmean23.299->26.063,p95 56.86->74.06,PCK.594->.553. 213/320changed,20improve>16,33harm>16,maxharm134.92. Otherinstance-samename-nearer24->28,7new (diagnosticnotdefinitiveIDGT). Visualrow13view0: q3nosemovesfrombrownmousenosetowardblackmousehead; q4feetmovetobrownmouse. Reject directsingletrainedselector ondual. No originaldatachanged. '
        'Implemented mousegeoforge/models/prior_candidate_selector.py PriorCandidateSelector inheritingLocalCandidateSelector, same22657params/statekeyshape. defaultoffexact; initialexact; baseline+.001, candidate1..5prior scalar0*20 (=logrelative3x3mass), plus residual B*tanh((learnedscore-learnedbaseline)/B), Bpredeclared2. Relativeweakmass<exp(-2) cannotwin; no guaranteeagainstidentity/anatomicalerrors. 7unit tests passed tests/test_prior_candidate_selector.py. Baseclassunchanged. '
        'tools/train_prior_candidate_selector_20261005.py trains bounded_prior_candidate_pilot_v1: exactSOURCE initial.pt/schedule.json, same800steps/AdamW1e-3cos1e-4/.001wd/clip1/CEsigma16/matchedsupporteligibility/equalimageloss asoldunconstrained control. Cachedtrain1489/VAL80; noextraTRAINduals. Onlyscoreparameterization changed. Final800only. Sourcehashes/protocol/checkpoints/histories/predictions/points saved. '
        'Results TRAINboundedunary/geometry real27.137/26.425,bottom29.504/29.765,syn21.772/21.248. VALmatched boundedunary real30.438,bottom41.197; geometry26.324,39.704. BothVALsynfallback19.225. IndependentunaryVALsyn18.063. Dualperqueryboundedunary25.657 stillworsebaseline23.299. CompareoldunboundedunaryVALreal29.511bottom36.122 vsoldgeometry29.588bottom61.161. Boundedgeometrybetterthanboundedunarybutbottomnotbestoldunary; donotchoosemodelperVALcamera. Last100CE unary.8718/geometry.8516 vsprior100.8872/.8661, slowingbutnotconvergenceproof. '
        'Visualboundedrow0 rescued formerlycatastrophicID17 hindtofront jump byretainingbaseline; row15 foot15and16collapseontosamevisiblepaw,15error64.41->131.71 (+67.30), so hardanatomicalfailurepersists. Row5savedgallerynotyetvisuallyreviewed. No promotion. '
        'NEXT bounded learnedgeometry viewstress withouttraining: rebuild features from cached_local_candidate_inputs_v1 validation24scenes sourcepaths (real8/syn16) with build_selector_inputs forfull/reverse/exactduplicates/V1/V2/threeviews/leaveoneout/missingall. Mustrecomputeevidence/eligibility forsubset, notjustslice fullfeatures; compareunarygeometry samegate andallpoints includingfallback. Sortdedup protectsordering; absentcamIDsnot embeddings. Use scene sourceviewindex to map packedrole80row indices; rawrow orderdiffersSOURCE112. Pure V1/V2mustexactfallback (>=2othersgate); independentunary isseparateexperiment. Defaultoffexact, missingall, permutation, duplicates exact. Reportcoverage aswellasmeans, noVALsubsetpicking. '
        'Then matched boundedprior trainingseed29/43replicates ifviewstresssane: frozenbaseline17/candidatesunchanged, onlynewheadinit+balancedimage schedule randomseed changes, clearly NOTfullencoder/posebaseline seeds. Botharms sameinit/schedule800. Needdonotclaimindependent3videoorfull3baseline-seedreplication. Onevariant(no threshold/auxloss sweep); ifevidenceinconsistent do notblindlyextend. CouldlaterexpandexistingdualTRAIN42 contextbeforemultiinstance training; no crosscameraqueryID assumption. '
        'Protectannotations/24excluded/135pending/R12oldmask/lockedGR2UR2UR3/backtailabdomen/dirtyworktree/productiondefaults. SharedGPUlock,harddeadline18:41:41UTC training18:11:41UTC, heartbeat15min unchanged. Do notrerunoldrecorders; updateplanstate coherently.')
    note='''

## 2026-10-05 dual-instance diagnostic and bounded probability-prior intervention

Frozen single-mouse-trained unary fails on 32 existing synthetic dual development images. All six queries are retained; GT counts select the offline evaluation cohort only, never enter forward. Each image has two confident queries, so the unique-query gate safely abstains everywhere. Enabling each predicted query worsens conditional nose/four-foot mean error 23.299 -> 26.063px, P95 56.86 -> 74.06, gated PCK32 .594 -> .553. 213/320 points change, with 20 improvements and 33 degradations exceeding 16px. Detector unchanged: 64GT,64 predictions,45 IoU>=.5 matches. Visual row13/view0 confirms a nose moving from the brown mouse to the black mouse and feet crossing instances. Other-instance-nearer count is only a diagnostic, not a reliable identity-swap label. Do not deploy this extension.

Next bounded experiment changes only the candidate-score parameterization, with identical original step800 initialization, image schedule, optimizer, eligibility and labels. New `PriorCandidateSelector` retains original mean as option0, adds native log-relative peak mass to other candidates and bounds learned relative correction to ±2 log-units (predeclared; no VAL search). A native relative mass below exp(-2) cannot overcome baseline. Seven unit tests pass: off/initial exactness, unavailable/NaN fallback, weak-candidate bound, head-only gradient, unary ignoring geometry, ordering and bound validation. No production factory modification.

| Bounded-prior experiment | TRAIN real | TRAIN bottom feet | TRAIN synthetic | VAL real | VAL bottom feet |
|---|---:|---:|---:|---:|---:|
| unary |27.137|29.504|21.772|30.438|41.197|
| geometry |26.425|29.765|21.248|26.324|39.704|

Unbounded controls: unary VAL29.511/36.122; geometry29.588/61.161. Bounded geometry now beats its matched unary control but does not outperform old unary on bottom feet. Do not choose a different model by VAL camera. Independent bounded unary improves synthetic VAL19.225 ->18.063, but dual per-query still worsens23.299 ->25.657. This constraint is not sufficient for multi-instance identity. Last100 sampled CE .8718/.8516 vs preceding100 .8872/.8661: slowing, not proof of convergence.

Visual inspection: bounded geometry preserves original point17 in row0, avoiding the previous ~267px added error; row15 still collapses points15/16 onto one visible paw, point15 worsening64.41 ->131.71px. Failures remain, no promotion. Next: recompute geometry under genuine view subsets/missing/duplicate/order stress, then controlled additional selector-training seeds29/43 if warranted. Frozen baseline remains seed17; these are not full-backbone seed repeats. Consider actual dual TRAIN contexts before any multi-instance training claim.

Artifacts and Windows reports: `dual_unary_diagnostic_v1`, `bounded_prior_candidate_pilot_v1` under the current cycle and `C:/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005`. Original annotations and locked sources unchanged.
'''
    doc=ROOT/'docs/singleview_gated_fusion_geometry_direction_20261005.md';text=doc.read_text(encoding='utf-8')
    if '## 2026-10-05 dual-instance diagnostic and bounded probability-prior intervention' not in text:doc.write_text(text+note,encoding='utf-8')
    shutil.copy2(doc,DEST/doc.name)
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='bounded_prior_viewstress_next',active_job=None,active_pid=None,updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),next=next_action)
    for key in ['dual_unary_visual_review','bounded_prior_7unit_tests','bounded_prior_visual_review']:
        if key not in state['completed']:state['completed'].append(key)
    write(CYCLE/'state.json',state);plan=json.loads((CYCLE/'plan.json').read_text());plan['next_command']=next_action;write(CYCLE/'plan.json',plan)
    index=DEST/'index.html';page=index.read_text(encoding='utf-8');link='<p><a href="dual_unary_diagnostic_v1/index.html">双鼠跨实例跳点</a> | <a href="bounded_prior_candidate_pilot_v1/index.html">最新：有界概率先验800步对照</a></p>'
    if link not in page:index.write_text(page+link,encoding='utf-8')
    print(state['stage'])


if __name__=='__main__':main()
