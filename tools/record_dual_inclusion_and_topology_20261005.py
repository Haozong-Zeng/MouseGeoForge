"""Persist failed data-only fix and explicitly bounded next topology experiment."""
import datetime as dt,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';DEST=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005')


def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def main():
    next_action=(
        'Data inclusion control COMPLETE outputdual_train_inclusion_control_v1, session71828/PID3768967 exited, noactivejobs. '
        'Same PriorCandidateSelectorunarybound2/headinit17/800steps, single_only vsdual_mixed replacinglast2of16imagesusingRNG17042; first14same. Botharms NEWindependentper-queryscore>=.5 eligibility andsupervision recomputed afterboxmatching; oldmatchedsupport masksnotused. NoGTcount inforward. sourcehashes/protocol/recomputed_supervision.pt/schedules.json/checkpoints/history/predictions/points/results saved. '
        'Baseline/single_only/dual_mixed TRAINreal30.508/27.546/27.884,bottom32.920/28.930/28.479,synsingle26.359/21.135/21.431,syndual22.235/25.611/21.674. VALreal33.246/30.862/30.667,bottom48.461/39.823/41.778,synsingle19.225/17.922/19.883,syndual23.299/26.356/25.557. DataimprovesdualTRAINmuchmorethanVAL, dualstillworsebase andsinglevalidationregresses; doNOTpromoteorblindlyaddsteps. CEtraindualsingle1.74mixed1.56 vsval1.75/1.64; sampledlast100single1.181vsprior1001.171,mixed1.236vs1.222. No conclusive convergenceclaimbutnoevidenceforblindextension. '
        'DualVALotherinstance-nearerbase24/control29/mixed29,newcross6mixed. Boxescapeaudit review.json: all6newother-nearerinsideown1.25predbox,23/25harm>16insideownbox. GT21/320outsideexpandedbox, sohardboxclippinginsufficientandcouldharmcorrectlabels. Visualrow13nosebrown nowstaysnearbaseline butfoot17(q3)stillonblackmouseupperregion, andblackquery18onbrownmouse. Detectionboxesremainpoor/overlapping, matchesGTonlypostforward. Windowsdual_train_inclusion_control_v1/index.html. '
        'Research primary GroupPose https://github.com/Atten4Vis/GroupPose andmodels/grouppose/transformer_deformable.py inspected, within-instance then same-semantic acrossinstance attentions; ED-Pose officialrepo+paper explicitbox/keypointcontexts. Current DecoderSpatialPoseReadout/MultiInstanceMouseHead flattenall6x23unrestrictedselfattn; currentCore23SpatialReadout alreadysoftsuppresses outside1.25predbox,cappenalty-6 (notnew). Hypothesis topologylimits inappropriatecrossinstancecrosssemanticinteraction, NOTprovenrootcause. Noexternalweightsdownload/reproductionclaim. '
        'IMPLEMENTED PROTOTYPE mousegeoforge/models/topology_pose_adapter.py TopologyPoseAdapter(baseline,topology=full|structured_union), defaultenabledFalse. Deepcopiesimmutablebaseline (no grads evalalways), copieskeypoint_decoder asnewtrainablebranch, zero-initializedLinear256->256 tokenresidual. enabledTrue baseline.tokens+residual(newdecoder_context), recomputescoarse+frozen spatialreadout withgradthroughpose tokens; nativeV1memory/instance_tokens/predboxesonly. update_mask onlynosefourfeet[0,14,15,16,17], allothercoords exactfrozen (backtailabdomen). Branchall138querieskept. full usesunrestrictedselfattn; structured_unionmaskallowssameinstanceslot ORsamesemantic (oneunionmaskNOTGroupPose sequentialtwoattention reproduction). Offandinitialexact; branchdecoderadditionalforwardneedscomputecostmeasurement. Sameparamsbotharms; nofactoryintegration. 7unittests passed tests/test_topology_pose_adapter.py defaultoff/initexact,frozensafety,gradient,maskrules,Qpermutation,untargetedexact,parametercount. NOTTRAINEDYET. '
        'NEXT preparematchednative-memorytopologytraining, NOTanothercandidateMLP. UseOLD=output/core23_spatial_cycle_20261004/single_view_train_context_v1 andvalidation manifests1223/112nativeFP32[256,h,w],raw.instance_tokens[6,256]; verifySHAmetadata. SOURCE=NEW/singleview_fusion_inputs_seed17_v1/train.pt,validation.pt suppliesfrozendetboxes/objectness,GTseparate,pixels,globalimage_indices; sourceexpected_instances only offline TRAINbucket/eval, neverforward. Loadfull_seed17pose asfrozenbaseline, verifydefaultoffandzeroinit enabledpredictions vsSOURCE.xy at<2e-5 ontrain/VALwithoutchangingthreshold. CouldpreloadnativeGPU~fewGB; groupbatchesbynativeHWreal704x768vs syn768square, doNOTpadchanges. Useall6query23pointsforward. '
        'Bothfull/structured_unionbranches identicalcopiedbaselineweights+zeroresidual, freshoptimizer; onlytopologymaskdiffers. Trainingonlypoint IDs1,15-18, GTmatching afterfrozenboxprediction outsideforward usingmatch_detection. Use matched_image_point_loss (pixels/768 smoothL1beta.01 equalimage) with only5pointvalid mask; noGTcountcrops/inferenceidentities, noCEcandidatehead, noextraNLL/geometry/boxobjective. Fixedinitialbudget800steps, balanced4realbottom4otherreal6syntheticsingle2syntheticdual, seed17, samesequencebotharms. NativeTRAIN1223 hasreal1028single-syn153dual42; doNOTpretendfullsyn461context frompreviouscandidatecache containsnativeencoder memory (extra308notcachednative). Newarms matchedsamepool, referenceoldcandidate results onlycontextnotstrictcomputecontrol. AdamW suggesteddecoder1e-5->1e-6,zeroprojection1e-4->1e-5cosine,wd.001clip1; recordbeforetrain. Initialsmokegradient/finite andtime/memory beforefull800. Save100stepcheckpoints separately iftrajectoryneeded, noVALselectedcheckpoint; final800evaluatefull112 withreal/synsingle/dual/bottom andunmodifieddetectcounts. CouldmeasureTRAINconvergenceattheend; doNOTautoextendunlessnew evidence. Defaultoff aftertrainingmustremainoriginalbaseline. Allother23pointoutputs unchanged. '
        'Read docs/singleview_gated_fusion_geometry_direction_20261005.md latestsectionfornumbers/sources. 12hdeadline18:41:41UTC,trainstop18:11:41UTC,heartbeat15minACTIVE/sharedGPUlock. Protect24excluded135pendingR12oldmasklockedGR2UR2UR3dirtyworktree/production. No activeGPUjobs andno traininglaunchedforprototype. Do notrerunoldrecorders; updateplanstate.')
    note='''

## 2026-10-05 dual TRAIN inclusion result and decoder-topology hypothesis

Data-only control: identical unary prior-bound2 head, initialization,800steps and per-query predicted-score>=.5 gate. Mixed arm replaces2/16 synthetic-single images with existing dual TRAIN42; first14 samples remain identical. Supervision recomputed independently of old multi-view support gate.

| Dataset (mean px) | Baseline | Single-only | Dual-mixed |
|---|---:|---:|---:|
|TRAIN real reliable|30.508|27.546|27.884|
|TRAIN bottom feet|32.920|28.930|28.479|
|TRAIN synthetic single|26.359|21.135|21.431|
|TRAIN synthetic dual|22.235|25.611|21.674|
|VAL real reliable|33.246|30.862|30.667|
|VAL bottom feet|48.461|39.823|41.778|
|VAL synthetic single|19.225|17.922|19.883|
|VAL synthetic dual|23.299|26.356|25.557|

Dual data improves TRAIN more than VAL; dual remains worse than original and synthetic-single regresses. No promotion or blind increase in steps. Last100 sampled losses are slightly higher than preceding100; raw sampled losses are not smooth full-dataset convergence curves. Dual other-instance-nearer count24/29/29; mixed adds6 such points, all6 within their own expanded predicted box.23/25 points harmed by >16px also remain in that box;21/320 GT points lie outside it. Hard box clipping is neither sufficient nor uniformly safe. Visual row13 nose improves relative to earlier errors, but cross-instance feet remain. Current original spatial readout already applies soft predicted-box suppression.

Primary-source research: [GroupPose](https://github.com/Atten4Vis/GroupPose) and its [official decoder](https://raw.githubusercontent.com/Atten4Vis/GroupPose/main/models/grouppose/transformer_deformable.py) separate within-instance and same-semantic across-instance attention. [ED-Pose](https://arxiv.org/abs/2302.01593) and [official repository](https://github.com/IDEA-Research/ED-Pose) motivate explicit instance-box/keypoint context. These human-pose results do not establish mouse transfer. Current local decoder flattens all6×23 queries into unrestricted attention; constrained interaction is a testable hypothesis, not a proven root cause. No external weights downloaded or paper reproduction claimed.

Prototype `TopologyPoseAdapter` now exists, **not trained**. Both arms copy the same decoder and add an identically zero-initialized token-residual projection over a frozen baseline. `full` preserves unrestricted new-branch attention; `structured_union` permits same-instance OR same-semantic interactions via one static mask. This is not GroupPose's sequential two-attention implementation. No GT identity or count enters forward; all138 queries remain. Only nose/fourfoot outputs can change. Back/tail/abdomen remain frozen. Default-off and initial-enabled exactness, frozen baseline, nonzero adapter gradient, query-permutation equivariance, mask rule and equal parameter tests pass (7 tests).

Next bounded trial: frozen genuine V1 native-memory pool1223TRAIN/112VAL (not previous extra308 synthetic candidate-only views), balanced4real-bottom+4other-real+6single-synthetic+2dual-synthetic, matched800steps. Coordinate-only `matched_image_point_loss`,5 reliable points, frozen box Hungarian only in loss/eval. Suggested predeclared AdamW decoder1e-5→1e-6, residual1e-4→1e-5 cosine,wd.001,clip1. Measure additional decoder forward compute/memory, check native golden replay and default-off before/after training. No checkpoint chosen on reused development VAL.

Windows report: `C:/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005/dual_train_inclusion_control_v1/index.html`. This prototype is isolated; production and labels unchanged.
'''
    doc=ROOT/'docs/singleview_gated_fusion_geometry_direction_20261005.md';text=doc.read_text(encoding='utf-8')
    if '## 2026-10-05 dual TRAIN inclusion result and decoder-topology hypothesis' not in text:doc.write_text(text+note,encoding='utf-8')
    shutil.copy2(doc,DEST/doc.name)
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='native_topology_matched_trial_next',active_job=None,active_pid=None,updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),next=next_action)
    for key in ['dual_inclusion_visual_box_escape_audit','primary_group_pose_edpose_research','zero_residual_topology_prototype_7tests']:
        if key not in state['completed']:state['completed'].append(key)
    write(CYCLE/'state.json',state);plan=json.loads((CYCLE/'plan.json').read_text());plan['next_command']=next_action;write(CYCLE/'plan.json',plan)
    index=DEST/'index.html';page=index.read_text(encoding='utf-8');link='<p><a href="dual_train_inclusion_control_v1/index.html">最新：加入双鼠训练的对照、框内跨鼠错误与下一步结构假设</a></p>'
    if link not in page:index.write_text(page+link,encoding='utf-8')
    print(state['stage'])


if __name__=='__main__':main()
