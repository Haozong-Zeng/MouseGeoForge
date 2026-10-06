"""Record bounded crop probe and verified source-group correction."""
import datetime as dt,json,shutil,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';DEST=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005')


def main():
    posepath=ROOT/'dataset/Homecage/LightningPose-CRIM13-Diverse400.plan.json';pose=json.loads(posepath.read_text());frames={x['scene_frame']:x for x in pose['keyframes']}
    source_groups={};verified=[]
    for role,folder in [('train','dual_train_allview_affinity_context_v1'),('validation','dual_allscene_association_audit_v1')]:
        records=json.loads((CYCLE/folder/'protocol.json').read_text())['records'];source_groups[role]=sorted({r['source_group'] for r in records})
        for r in records:
            frame=r.get('frame',int(Path(r['scene']).name.split('_')[-1]))
            source=frames[frame]['source_image'];session=Path(source).parent.name
            assert r['source_group']=='crim13/'+session
            verified.append(dict(role=role,frame=frame,source_group=r['source_group'],source_image=source))
    assert set(source_groups['train']).isdisjoint(source_groups['validation']) and len(verified)==37
    correction=dict(source_groups=source_groups,verified_frames=verified,pose_plan_sha256=hashlib.sha256(posepath.read_bytes()).hexdigest(),
        correction='Earlier same-source-video/different-frame wording in some reports was inaccurate. All21TRAIN and16VAL source_group fields agree with actual pose-plan source_image sessions and are disjoint. They still share rendered arena/meshes/materials/cameras and repeatedly used development role; not independent real RGB evidence. Raw experiment outputs left immutable; this provenance correction supersedes those prose caveats, not metrics.')
    (CYCLE/'dual_source_provenance_correction.json').write_text(json.dumps(correction,indent=2),encoding='utf-8');shutil.copy2(CYCLE/'dual_source_provenance_correction.json',DEST/'dual_source_provenance_correction.json')
    errata='<p style="background:#fff3cd;padding:12px">来源说明更正（11:27 UTC）：双鼠 TRAIN 来自 CRIM13 021709_A25_Block22_BCfe1_t，开发集来自 021309_A29_Block16_BCma1_t；已逐帧核对姿态计划。此前“同源视频不同帧”的表述不准确。两组仍共用 Blender 场景/模型/毛色/相机，不等于独立真实视频验证。此说明更正文字，不改变指标。</p>'
    for root in [CYCLE,DEST]:
        for report in ['cross_query_affinity_diagnostic_v1','partial_and_prequery_review_v1','predicted_crop_affinity_diagnostic_v1']:
            path=root/report/'index.html'
            if path.exists():
                content=path.read_text(encoding='utf-8')
                if errata not in content:path.write_text(errata+content,encoding='utf-8')
    note='''

## 2026-10-05 crop re-encoding: reduced slot bias, insufficient raw identity separation

Encoded296 predicted-confident boxes from148images as RGB/gray pairs:592 main crop inputs plus5 repeat-check inputs,150 batch calls,11.42s total. Auxiliary224square aspect-preserving crops; frozen existing Omega image-backbone only, no new weights/download/training. Main768based detector/pose unchanged. MeanRGB padding is fixed across RGB/gray.5crop preprocessing tests pass. Reverse batch is exact; singleton versus batch bf16 maxdifference0.00805, minimum descriptor cosine0.999985, explicitly not claimed bitwise identical.

| Crop readout | TRAIN identity AUC | Development identity AUC | Development mutual known correct | Unknown / all mutual |
|---|---:|---:|---:|---:|
|RGB CLS|0.553|0.559|55/79|41/120|
|RGB patch mean|0.517|0.502|39/71|42/113|
|Gray CLS|0.519|0.524|42/74|35/109|
|Gray patch mean|0.504|0.491|34/73|39/112|

The crop encoder's same-query AUC is near0.5 rather than the former semantic0.945, but raw identity separation remains weak. RGB CLS improves the limited mutual-ranked subset, not enough to validate unknown-identity fusion. Actual input contact sheets still show mixed-animal crops. Gray removes chroma only, not coat brightness, and is not a same-color-animal test. Stop replacing raw cosine feature variants.

### Provenance correction

The previous wording suggesting TRAIN and development share the same source video was inaccurate. Checked all37frame records against `dataset/Homecage/LightningPose-CRIM13-Diverse400.plan.json` source_image paths: all21TRAIN scenes use `crim13/021709_A25_Block22_BCfe1_t`; all16development scenes use `crim13/021309_A29_Block16_BCma1_t`. The action sources are disjoint, while arena, meshes, materials and camera setup are shared. This is source-action-separated synthetic development, not independent real RGB or same-color validation. Historical numerical outputs remain unchanged; `dual_source_provenance_correction.json` supersedes inaccurate prose caveats.

Next bounded experiment: a small shared linear association projection trained only on TRAIN cropCLS features, with RGB/gray matched initialization/schedule and explicit unknown-label exclusion in the loss. Reserve5of21TRAIN scenes for association-stage internal calibration (same action source; do not call it source-independent), keeping the outer source split intact. Evaluate final fixed-budget models on the existing development source without choosing steps there. No pose-head training or fusion deployment; learned black/brown appearance remains a possible shortcut.

Windows report: `C:/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005/predicted_crop_affinity_diagnostic_v1/index.html`.
'''
    doc=ROOT/'docs/singleview_gated_fusion_geometry_direction_20261005.md';text=doc.read_text(encoding='utf-8')
    text=text.replace('A frame split here is not an independent-video test.','The dual action-video sources are disjoint (verified in the provenance correction below), but the render scene/assets are shared and this is not independent real RGB validation.')
    if '## 2026-10-05 crop re-encoding:' not in text:text+=note
    doc.write_text(text,encoding='utf-8');shutil.copy2(doc,DEST/doc.name)
    next_action='''Cropencoder probe COMPLETE noactivejobs session80539/PIDstatus exited. Tools extract_crop_encoder_affinity_20261005.py,diagnose_crop_encoder_20261005.py,review_crop_encoder_20261005.py. mousegeoforge/predicted_rgb_crops.py5tests. CurrentOmegaimagebackbone built _build_patch_embed16/1024, strictloadaggregator.patch_embed.* fromverifiedc02da418... existingcheckpoint; no newmodel. sourceexactRGBload_item, confidentpredictedboxes>=.5 (naturally2/image), clippedfloor/ceilbounds,aspect224resizebilinearantialias,meanRGBpadding, normalizedmean/std. NoGTcrops/count/IoUselection. RGBgrayactivecontent,graypaddingfixed;graydoesNOTremoveluminancecoatcue. Main768inputunchanged.592maincropinputs=296boxesx2 +5repeatinputs,150batchcalls,11.42sec. Reverseexact,singletonbatchmax.008048bf16,mincos.99998492,notbitwiseexact.
predicted_crop_encoder_v1/descriptors.pt list{role,group,scene,views,descriptors{rgb_cls,rgb_patch_mean,gray_cls,gray_patch_mean}[V6,1024],valid[V6],cropinfo}; invalid/unencodedqueriesNaNvalidfalse,full6mapping. crops/firstTRAINandVALall4viewsq3/q4RGBgray224PNG actualinputs saved. sourcegroups unchanged. predicted_crop_affinity_diagnostic_v1 evidence.pt/matrices,train/validationedges andrankings,json. Generaldiagnose_prequery_roi_20261005.py nowmain(channels optional) sharedmetric evaluator, nolearning. IdentityAUCTRAIN/VALrgbCLS.553/.559,rgbpatch.517/.502,grayCLS.519/.524,graypatch.504/.491. VALmutualrgbCLS55of79known+41unknown120total,grayCLS42of74+35unknown109total; rawcosweak. SamequeryAUCnear.5 reducesoldslotbias butnotidentitysafe. Actualcontactsheet inspected, mixedanimalcrops remain. Windows predicted_crop_affinity_diagnostic_v1/index.html.
IMPORTANTprovenanceCORRECTION: earlierhandoff/reports claimed sameoriginalvideos, thiswaswrong. VerifiedALL37frames actualposeplan source_image againstrecords.source_group:21TRAIN=crim13/021709_A25_Block22_BCfe1_t,16VAL=crim13/021309_A29_Block16_BCma1_t. See dual_source_provenance_correction.json withposeplanSHA/framepaths. OuteractionsourceheldoutALREADYenforced. SameBlenderarena/mesh/material/camera and reusedVAL, NOTindependentrealRGB/samecolor. OldrawmetricJSONimmutablecontainsincorrectprosecaveat;correctionandreportbannerssupersede. Donotpropagatesamevideoassertion. TRAINhasONLYoneactionsource sointernalcalibcannotbesourcedisjoint;makeexplicitstage-specificTRAINsceneholdout whileouterVALsourceuntouched.
NEXT stoprawcosinevariants; smallTRAIN-supervisedassociationmetric pilot, distinctfromfailedpose-heads. Sharedlinearprojection1024->64, noqueryID/cameraID embedding, L2normalizeinput/output; defaultoffpreservesoriginaldescriptor, enabledprojectionopt-in. 65536trainableparameters no bias; seeds17/29/43 orthogonalinit, RGBCLS vsgrayCLS arms SAMEinit/schedulewithinseed; rawcos andfrozeninitialprojectioncontrols. Keepmainencoder/detector/posecompletelyfrozen. Allconfidentquerydescriptors enterforward includingunknown; onlyloss masks useposthocTRAINboxassignmentIoU>=.5 knownID, unknownmustnotbecome negatives. Within-scene supervisedcontrastiveloss: positives sameanimal acrossdifferentphysicalcameras, negatives differentanimal including samecamera; noGTidentity/count toembeddingforward, noframesmixedglobalIDassumption. Anchorsmusthave>=1positiveand1negative. Temperature.1fixed, meananchorsperscene thenmeanbatch; lossinvariantnodeordering. Unit tests defaultoff,querypermutation,unknownmask,noGTforward,finitegradients.
Splitfixedbeforetrain: NumPydefault_rng2718 permutation of21TRAINsceneIDs in sourcegroupsfileorder; first5associationcalibration,16fit. ThiscalibsameactionsourceA andbasepose/detectorpreviouslysawTRAIN; onlynewassociationheadholdout, notglobalunseen. NeveraddVALsourceBtoloss/calibration. Recordexactsplit/hashesprotocolbeforejob. Initialboundedbudget800steps/arm,8fit-scenes sampledwithreplacement,AdamWlr1e-3 cosine1e-4 wd.01 clip1. Three seeds pairedRGBgray, initial/final/per100checkpoints, train+internalcalib loss/identityseparationevery100, noVALuntilfixedfinal; stepcountnotchosenonVAL. Ifcaliblackpositiveanchors,reportandadjustusingTRAINonlynotVAL. Budgetcanextendlateronlybasedonfit/calibcurves,notautomatically. No newRGB needed,usecachedcropCLS; GPUlockrequiredforexperiment; CPUalsofinebutrecordcompute.
Aftertraining reportfit/calibration/VAL pairAUC,mutualknownaccuracy/unknowncoverage/samequerybias,rawandinitialcontrols,seedspread. Modelinferencefeaturesonly. DoNOTimmediatelyfitVALthreshold oruseidentityheadforfusion. Geometrypartialcontrolpriorcutoff2.364pxgavezero3viewgroups (failure); cannotjustrelaxitafterlookingVAL. A latergatedcombinedaffinity wouldrequireTRAIN/internalcalibration,faultstress,unknown/missedqueries andsamecolor/realtests. CropRGBgraycomparisonmayrevealchromadependence butbrightnessremains; nofoundationgeneralizationclaim. Ifmetricfails, consolidate/reconsiderdatacoverage vslearningbackbone, notendlesssameposeheadvariants.
Preserveoriginals24excluded135pendingR12oldmasktailabdomenlockedGR2UR2UR3dirtytree; no checkpointpromotion. Deadline18:11:41UTCtrain/hard18:41:41UTC2026-10-05unchanged;heartbeat15min/sharedGPUlock. Readtime/status/PIDbeforejob; updateresults/doc/handoffnotoldrecorders.'''
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='supervised_association_metric_pilot_next',active_job=None,active_pid=None,next=next_action,updated_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    state['completed']=list(dict.fromkeys(state['completed']+['predicted_rgb_crops_5tests','crop_encoder_four_readout_diagnostic','crop_encoder_visual_review','dual_source_provenance_verified_corrected']))
    (CYCLE/'state.json').write_text(json.dumps(state,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');plan=json.loads((CYCLE/'plan.json').read_text());plan['next_command']=next_action
    (CYCLE/'plan.json').write_text(json.dumps(plan,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    index=DEST/'index.html';page=index.read_text(encoding='utf-8');link='<p><a href="predicted_crop_affinity_diagnostic_v1/index.html">预测框裁剪编码与RGB/灰度对照（含来源分组更正）</a></p>'
    if link not in page:index.write_text(page+link,encoding='utf-8')
    print(state['stage'])


if __name__=='__main__':main()
