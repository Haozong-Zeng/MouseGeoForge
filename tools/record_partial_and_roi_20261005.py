"""Negative partial matching/ROI evidence and next genuinely different probe."""
import datetime as dt,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';DEST=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005')


def main():
    note='''

## 2026-10-05 strict partial matching has zero coverage; pre-query ROI cosine also fails

`partial_camera_association.py` implements a separate default-off partial grouping utility (7tests): mutual nearest across physical cameras plus frozen absolute geometry cutoff, maximal complete groups of>=3cameras, at most one detection/camera, reject ties and overlapping ambiguous groups. No animal-count input, no pose edits. TRAIN-only1% negative-edge calibration froze2.36447px before development evaluation:1/164 negative and only2/162 positive TRAIN pairs pass. Natural TRAIN21scenes, development16scenes and all three logit-fault modes produce ZERO qualifying groups. This is zero coverage, not safe-fusion success. Cutoff was not relaxed on development results.

A distinct no-training probe extracted features before instance query embeddings: current frozen Omega image backbone patch tokens1024D versus projected image memory256D, each mean-pooled on5x5samples within fresh predicted boxes. No GT boxes/masks or new downloads.148independent RGB passes,24.59s total, exact original pose/box replay;5ROI tests pass. This is feature-map pooling after full-image encoding, NOT crop-image re-encoding.

| Descriptor | TRAIN identity AUC | Development identity AUC | Development known mutual correct | Unknown / all mutual |
|---|---:|---:|---:|---:|
|Pre-query image backbone ROI|0.496|0.469|40/84|46/130|
|Projected memory ROI|0.416|0.411|41/102|56/158|

Neither raw cosine is usable as re-ID. Visual inspection of first-scene predicted crops confirms that some side-view boxes include both mice; boxed-region purity and full-image context are plausible confounds, not experimentally isolated causes. Do not claim a different feature-extraction layer solved identity association.

Next bounded hypothesis: predicted-box RGB crop BEFORE a separate pass through the existing frozen image backbone, compared with current full-image ROI pooling. Preserve aspect ratio,224square auxiliary crop input, unchanged768-based main detector/pose input; record extra encoder cost. Use only predicted-confident queries, no GT crop/count. Predeclare crop CLS and mean-patch readouts, RGB and grayscale controls, no training or VAL threshold optimization. Grayscale removes chroma but not coat brightness, so it cannot establish same-color generalization. If frozen crop descriptors still fail, stop raw-cosine variants and consider a TRAIN-supervised association descriptor with independent-source/same-color validation rather than more pose-head steps.

Windows report: `C:/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005/partial_and_prequery_review_v1/index.html`.
'''
    doc=ROOT/'docs/singleview_gated_fusion_geometry_direction_20261005.md';text=doc.read_text(encoding='utf-8')
    if '## 2026-10-05 strict partial matching' not in text:doc.write_text(text+note,encoding='utf-8')
    shutil.copy2(doc,DEST/doc.name)
    next_action='''Two new probes COMPLETE, no activejobs. partial_association_control_v1: TRAIN-onlymean-epipolarcutoff2.3644697163px strict<, empiricalbudgetfloor(.01*164)=1wrongnegative, accepts1negativeand2of162positives. Calibration.json persisted11:05:11.782UTC/hashBEFOREVALread; noVALtuning. partial_camera_association.py defaultoff utility7tests usesmutualnearest/absolutecutoff/maximalcliques>=3physicalcams/atmostonepercam/tiesrejected/overlappinggroupsALLrejected. NaturalTRAIN21VAL16+3faultmodes allZEROgroups, TRAIN3pairEdgesVAL1. Zero coverage notsuccess; nofusion/posechanges. Keepnegative, doNOTloosenVALcutoff.
Then prequery_roi_descriptors_v1 completed session60584/PIDinstatus,148freshRGB=84TRAIN64VAL,24.586sec,totalposeboxlogitdelta0. tools/extract_prequery_roi_descriptors_20261005.py hooks runtime.model.aggregator.patch_embed outputdict x_norm_patchtokens (currentOmegaimagebackbone1024D BEFOREinstancequeries) andrawmaps256D. mousegeoforge/predicted_roi_descriptor.py5tests pools5x5bincenters within1xpredictedbox (notold2xROI), ignoresoutofimage samples, all6queries; outputvalidmask. NoGTforward. descriptors.pt list(role,group,scene,views,descriptors{image_backbone[V6,1024],projected_memory[V6,256],valid[V6],sampled_fraction[V6]}), results sourcehashes/runtimehashes/148timings. Originalfullimageencoder, NOTRGBcropencoder.
prequery_roi_affinity_diagnostic_v1 tools/diagnose_prequery_roi_20261005.py: separatecosine, nolearning/threshold. TRAIN/VALimagebackbone identityAUC.496/.469;VALmutual40correct84known+46unknown(130total). Projectedmemory.416/.411;VAL41of102known+56unknown(158total). No usablefrozenreIDcosine. Allcandidate andunknowncoverage explicit. Sourceevidence cross_query_affinity_diagnostic_v1/evidence.pt providesnodes/identitiesONLYposthoc. Reviewpartial_and_prequery_review_v1/index.html Windows; firstscene8predictedcropcontactsheet visuallyinspected: several sideviewboxes mixblack+brownmouse, notpureinstanceRGB; alreadyIoUqualifiedboxescancontainothermouse. MainRGBsizesunchanged768based.
NEXT one genuinelydifferent bounded appearance probe: encode predicted-box RGB crops BEFOREimagebackbone ratherthanpoolfull-imagecontext. UseexistingcurrentOmegaimagebackbone, noexternalmodel/download/learning. Read vggt-omega/vggt_omega/models/aggregator.py _build_patch_embed(patch_size=16,embed_dim=1024), DinoVisionTransformer24layers; aggregator.patch_embed forward returnsdict x_norm_clstoken andx_norm_patchtokens. Couldload ONLYaggregator.patch_embed.* fromsame vggt-omega/vggt_omega_1b_512.pt (strip prefix,strictload,verifySHA) insteadloadingwholeOmega; builderinitmatchingstate, onGPUevalno_grad bf16. Originalimage normalizingconstants fromaggregator mean[.485,.456,.406],std[.229,.224,.225]. Main768detector/poseNOTchanged; auxiliarycrop224x224multiple16, aspectpreservingresize+meanRGBpadding. Recordseparateextraencodercalls, nofairnessclaim.
Sourcegroups TRAINdual_train_allview_affinity_context_v1/groups.pt21scenes4views; VALdual_allscene_association_audit_v1/groups.pt16x4. Source savedRGBPNGforvisual hasquantization; forencoding use load_item(g.record).batch.images exactpreprocessing floats, sourcepredictedboxes/score fixedfromverifiedruntime (notGT). Selectqueries withoriginalobjectness>=.5 ONLY (naturally2/image) for296predictedcrops; this ispredictionfilter,notGTcount. DoNOTselectbyIoUorGTmatch, allunknownboxesretained. Invalid/outsidecrop abstains,donotpretendzerosvalid. Crop boundsclippedtoimage, preserveaspect; outputfixedquantizationpolicy andcropboxpath/hash. Predeclare2readouts cropCLS andmeanpatch, bothRGB andgrayscale inputs. Grayscale .299R+.587G+.114B replicated3channels removesCHROMAonly, leavesbrightnessfurcue; cannotclaimsamecolormicegeneralization. Colorandgraycanbatchbutuseforwardevaldeterminismtestpermutation/singlevsbatch, noqueryembedding. Saveinputcropcontactsheet anddescriptorartifact[C6D]withvalidqueriesmask; preservefull6-indexmapping, noGTtoforward.
Then threshold-freeidentity/samequeryAUC andmutualNNaccuracy withunknowncoverage onTRAIN/VAL, fixed5otherbaselinechannelsalreadyrecorded. UseGTboxIoU>=.5 identityONLYafterdescriptorforward asprior; checkfixedGTIDs acrosscamera. Do notfitweights/thresholdsonVALorchoosebestreadoutascertifiedsolution. Allsameoriginalblackbrownsourcevideos; noexternalsamecolortest. Ifrawcropcosinestillfails, stopfreefeaturevariantsandconsiderTRAINsupervisedcontrastiveassociation (notposehead) withsourcegroupedheldoutcalibration, orcollectindependentmulti data; don'tblindincrease800poseheadsteps. Do notintegrateunsafeuniquefusion orpromoteweights.
Protectannotations24excluded135pendingR12oldmasktailabdomenlockedGR2UR2UR3dirtytree.12hdeadline unchangedtrain18:11:41UTC/hard18:41:41UTC2026-10-05. Checktime/PID/GPUlock beforeencoder. Existingheartbeat15min, no activegoalneeded. Savehashes/visuals/docs/coherenthandoff,doNOTrerunoldrecorders.'''
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='predicted_crop_encoder_probe_next',active_job=None,active_pid=None,next=next_action,updated_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    state['completed']=list(dict.fromkeys(state['completed']+['partial_association_7tests','partial_association_zero_coverage_review','predicted_roi_5tests','prequery_roi_affinity_negative','partial_roi_visual_review']))
    (CYCLE/'state.json').write_text(json.dumps(state,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');plan=json.loads((CYCLE/'plan.json').read_text());plan['next_command']=next_action
    (CYCLE/'plan.json').write_text(json.dumps(plan,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    index=DEST/'index.html';page=index.read_text(encoding='utf-8');link='<p><a href="partial_and_prequery_review_v1/index.html">保守关联零覆盖与冻结ROI特征负结果</a></p>'
    if link not in page:index.write_text(page+link,encoding='utf-8')
    print(state['stage'])


if __name__=='__main__':main()
