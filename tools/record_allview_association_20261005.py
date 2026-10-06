"""Record all-camera evidence, preserving the distinction from fusion success."""
import datetime as dt,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];CYCLE=ROOT/'output/singleview_fusion_cycle_20261005'
DEST=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005')


def main():
    summary=json.loads((CYCLE/'dual_allscene_association_audit_v1/summary.json').read_text())
    assert summary['images']==64 and summary['fusion_eligible']==0
    next_action='''Allview audits COMPLETE; no active jobs. composite_allview_association_audit_v1: first2dual scenes4views each +firstsingle WashFace8views,16 independentRGB encoder calls. dual_allscene_association_audit_v1: ALL16 existing dual development scenes4views=64 freshRGB calls, session46303/PID3770621 exited. No new training/renders/labels.
Scripts audit_composite_allviews_20261005.py(main requested_counts default{1:1,2:2}), audit_dual_allscene_association_20261005.py selects all16dual and overrides OUT. Review scripts similarly split; outputs include groups.pt with fresh predictions, projections, outputs, actual RGB, evaluation_data; results/protocol/status/sourcehashes. Fullbatch load_item preserves common camera scale; eachimageencodedindependently, no labels/count/matches into runtime or wrapper. All138queries. Synthetic projection max8.384e-5px, old32viewRGB output paritymax1.907e-6. V1/V2localfallback,permutation,duplicate,allinvalid exact. No need rerun these audits.
ALL64dualimages predict2 confidentqueries each. Every local/fusion gate abstains, all modes identical.128GT128pred97TP atIoU>=.5,precision/recall=.7578125,conditional640reliablepoints mean21.826px,P9554.775,PCK32gated.6328125. Not same cohort as old32image23.299px. Count correctness is not localization correctness. This does NOT exercise natural unique-query identity failure; no multi-mouse fusion success claim. Visualgroup07view1 feet on torso/tail and boxesmisplaced;group10view3 two overlapping boxes and clusteredpoints. Windows dual_allscene_association_audit_v1/index.html containsall64panels. Initial3scene report includes single8views counts[1,1,1,1,1,1,2,1],dupview6excluded; singlemean25.863->22.652 composite, but sameoldscene extra views, not independenttest. view6duplicateq3/q4 visual, geometry doesnotresolve.
NEXT bounded inference-only fault-injection audit on saved64dual predictions to expose unique-query association assumption. Explicitly synthetic failure injection, NOT natural dropout frequency/performance, not real image occlusion. Do not train/tune. Keep baseline underlying probability/maps/boxes/keypoints fixed and all6queries, modify a copied objectness_logits tensor to suppress all but one ORIGINAL confident query in eachview. Fixed modes chosen BEFORE results: (a) retain highest confidence query perview; (b) retain leftmost predicted box center; (c) alternate highest/second-highest confidence by physicalcamera ID parity. No GT instance identity/count used to choose query or modifyinputs. Inputs already have2 queries because cohort diagnostic; no GT inforward. Leave retainedlogit unchanged, others set stronglynegative e.g.-20 documented, do not alter original artifacts. Use frozen bounded wrapper all4views, local-onlyvscomposite; inspect count of fusioneligible, changes, and afterforward GTbox/objectness match or original staticboxassignment to quantify mixed-identity support. Report BOTH operational match with perturbedscores and fixed-originalbox assignment if metrics conflict; cannot silently change matching to improve claims. QueryIDs are not identities. Use stable per-scene instance_ids and IoU>=.5 reliability for posthoc identity audit; ambiguities nothardGT. NoGTtargettoqueryrouting.
Required checks: input original immutable/hash, defaultoffexact,1/2viewslocalfallback/order/dups/allinvalid, nonreliable18pointsunchanged. Inspect allquery RGB overlay where original and retainedquery visible, show lostdetections separately; raw actual images unchanged. If mixedidentitysupports passfusion, record concretecounterexample and never promote unique-query gate asmultimouseassociation. Do not patchthresholds or use knownGTcountguard. Following step can research prediction-only calibrated association with abstention, or explicit API scoping to single-subject until association tested; don't erase current gain via arbitraryVALtunedgate. Keep cached fault-injection and naturaltests separate.
Previous matched head/topology/heatmap failures documented; avoid repetitive training sweeps. Protect originals,24excluded135pending,R12oldmask,tail/abdomen,lockedGR2UR2UR3,dirtyworktree. No modelpromotion. Read plan for harddeadline18:41:41UTC/trainstop18:11:41UTC2026-10-05,sharedGPUlock,heartbeat15min. Original12h deadline NOTextended. CPU cached faulttest no newencoder needed; still report actualcompute.'''
    note='''

## 2026-10-05 all-camera audit: dual fallback is not association success

Expanded from two dual scenes to all16 existing synthetic dual development scenes, four physical cameras each,64 independent RGB encoder calls. Common camera gauge checked against synthetic projections (max0.000084px); existing-view cache parity max1.91e-6. All view-order/duplicate/invalid and1/2-view local fallback checks pass. No training, rendering, annotation changes or deployment.

All64 images predict two confident queries, causing the current restricted local/fusion module to abstain everywhere. All modes therefore equal baseline:128 predictions/128 targets,97 true positives at IoU>=0.5; conditional640 reliable points mean21.826px, P9554.775px, detection-gated PCK32=0.6328. Correct count does not establish correct boxes or points. This all64-view cohort differs from the prior32-view score23.299px. No natural single-query association failure occurred, so this cannot validate safety under missed detections.

Visual group07/view1 retains misplaced boxes and paw positions on torso/tail; group10/view3 has overlapping boxes and clustered points. On the separately fixed single synthetic scene, eight views include one duplicate-detection view6 that abstains; mean25.863->22.652px with supported fusion. Same scene plus more cameras is not external generalization.

Next: a separately labeled fault-injection audit will suppress copied detection logits using predeclared prediction-only selection, leaving actual RGB and all original artifacts unchanged. Its purpose is to test whether missed detections can create mixed-instance support, not estimate natural error rates. Do not use GT identity/count for query selection and do not claim dropout tests are real-image occlusion tests. No further training until this assumption is understood.

Reports: `C:/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005/dual_allscene_association_audit_v1/index.html` and `composite_allview_association_audit_v1/index.html` in the same parent.
'''
    doc=ROOT/'docs/singleview_gated_fusion_geometry_direction_20261005.md';content=doc.read_text(encoding='utf-8')
    if '## 2026-10-05 all-camera audit:' not in content:doc.write_text(content+note,encoding='utf-8')
    shutil.copy2(doc,DEST/doc.name)
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='association_fault_injection_next',active_job=None,active_pid=None,next=next_action,updated_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    for key in ['allview_three_scene_visual_review','all16dual_allview_rgb_audit','all16dual_visual_review']:
        if key not in state['completed']:state['completed'].append(key)
    (CYCLE/'state.json').write_text(json.dumps(state,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    plan=json.loads((CYCLE/'plan.json').read_text());plan['next_command']=next_action
    (CYCLE/'plan.json').write_text(json.dumps(plan,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    index=DEST/'index.html';page=index.read_text(encoding='utf-8');link='<p><a href="dual_allscene_association_audit_v1/index.html">全部16个双鼠场景/64视角检查：数量正确不等于定位正确</a></p>'
    if link not in page:index.write_text(page+link,encoding='utf-8')
    print(state['stage'])


if __name__=='__main__':main()
