"""Record negative topology result and stop repetitive head/loss sweeps."""
import datetime as dt,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';DEST=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005')


def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def main():
    r=json.loads((CYCLE/'native_topology_control_v2/results.json').read_text())
    next_action=(
        'Native topology control COMPLETE native_topology_control_v2,session20400/PID3769593 exited,noactiveGPUjobs. v1 FAILEDpretraining bucketassert because21single syntheticframeskindcurve notliteral synthetic; v1failure preserved. v2usesnonreal-single bucket132synthetic+21curve=153; no curve labels used. tools/train_native_topology_control_20261005.py nowpointsOUTv2. '
        'Full vsstructured_union800steps sameinit/schedule1223TRAIN112VALnativeV1,4realbottom4otherreal6synsingle2syndual,coordinateonlylossIDs1,15-18,all138queryforward. frozenoriginalbaseline+copiedtrainabledecoder+zeroLinear256residual,1119744trainableparamsbotharms. Module mousegeoforge/models/topology_pose_adapter.py,7tests. Noencodertrain/newforward; nativecacheGPUtraining~24secs/arm,peakallocated~3.8GB/cacheincluded,notRGBinferencecost. Defaultoff/sourcegolden<2e-5,initialresidualexact0,aftertrainingoffunchangedand18untargetedcoords EXACT0. '
        'Results full/structured TRAINreal28.575/28.764,bottom27.782/27.977,synsingle15.136/15.201,syndual21.073/21.116. VALreal33.267/33.374,bottom46.296/46.177,synsingle18.376/18.616,syndual22.420/22.398. OriginalVAL33.246/48.461/19.225/23.299. No cleartopologyadvantage; no promotion/extension ofthisvariant. FullTRAINsynsingleoriginalpool153 baseline15.83, NOT26.359whichuses461fullviewsfromcandidateexperiments; cannotcrosscompareasmatchedtrial. ShapevariesnativeHWincludingreal704x768andsyndual768x432, no padding. Checkpointsstep0100..0800 separately botharms, initial.pt,evalinputs,results/gates/compute. '
        'Visualv2row34realbottomduplicateq3/q4stillshowfeetinsideabdomenandhandsclustered; twoarmsnearlysame. dualrow61brownfeetclustered, noresolveanatomy. Windowsnative_topology_control_v2/index.html. Sampledlast100lossfull.01352vsprior100.01297,structured.01360vs.01309; notclearevidenceextend. '
        'IMPORTANTpriornegativefound: OLD/review_reliable_heatmap_v1/decision.json andreadout_post4800_reliable_heatmap_v1/protocol.json. Matchedcontinuation4800->6400+.002GaussianKLsigma12px forIDs1,15-18,all3spatialseeds REAL43.565/45.231/44.038->49.922/50.330/48.741 andDUAL18.807/18.080/18.740->20.770/19.858/20.818. DoNOTrerunsigma/weightheatmaptuning asnovel. Read docs/pymousestudio_core23_spatial_cycle_results_20261005.md foroldnegative/risk. RepeatedreusedFR2/AZR2L2VALadaptationrisk explicit. '
        'NEXT stopnewtrainingfornow; engineeringauditactualRGB integration ofpromising bounded candidate selectors. Fixedhistoricalseed17, NOTbestVALseed. Use bounded_prior_candidate_pilot_v1/unary_latest.pt ANDgeometry_latest.pt (PriorCandidateSelector bound2), original frozenposefull17/detmixed17/encoder. tools/singleview_candidate_runtime.py FrozenSingleViewCandidateRuntime exposes from_images([1,1,3,H,W]) -> xy[6,23,2],boxes[6,4],objectness_logits[6],tokens[6,5,256],probability[6,5,48,48],maps[1,256,48,48]. local_candidate_inputs forms6candidatepositions/masses/features; build_selector_inputs formslocal/scalarsandphysicalcameraPgeometry. No labels inforward. '
        'Proposedisolatedwrapper twoopt-ins(local=False,fusion=False) defaultbothOFFreturnsoriginalXYexact. local enabled usesboundedunary independently ineachimageonlyuniqueconfquery>=.5, noothercamreq. fusion enabled usesboundedgeometry onlywhere >=2OTHERdistinctuniqueconfphysicalcamera supports (existingbuildereligibility); otherwise FALLBACKTOUNARYwhenlocalon, ororiginaliflocaloff. Geometryon overrideseligible reliable5points only. Nooptimisticperquerydualextension (failed), everydetectorqueryretained andboxes/objectnessunchanged. Existingbothheads zero-init baselineexact; learnedheadsnotclaimednewzero. NoGTcount/match/cameraIDidentity. Associationassumessingle-instancepilot; uniqueconfquerydoesNOTprovephysicalsameidentity ifmultimouseunder-detected. Explicitlimitation, no productionmulti-instancefusionclaim. Dualwith2confqueriesfallsbackoriginal; diagnosticall6predictionskeep. '
        'First wrapperunit/cacheparity: defaultoffexact,1/2viewswithlocal equalsindependentunary,geometryunsupportedfallbackconsistent,fullsameeligiblegeometryasold,order/duplicates/missingall no NaNs,untargetedexact,gradtestsifneededfrozenheadsnottrainmode. CompositecacheVAL80comparebaseline/unary/composite, newbaseline fallbackmustbeexplicit notreusematchedgate scores. Then actualRGBsubset re-extraction using load_item/load_scene existing exactpreprocess (realalready704x768derived; donotapplyoriginal1080K); noGTpixels/boxesintoRuntime. CameraPfromauditedcamera_geometry_audit_v2 train/validationcameras orscene fromcached_local_candidate_inputs_v1; preservecommonscene.scale/projections/nativepixelresize. OldprobabilityRGBreplaytools canservepattern. Needtest1/2/allavailable observedviews fixedphysIDs,reverse,duplicate. CanstartfirstfewrealVALIDscenes+synthetic2view+dualregression, predeterminedbeforeviewresults. No cache slicing masquerades asRGB; eachV1 encodedindependently; Nencodercalls notclaimedcomputefairwithsinglemultiviewencoder. '
        'SaveWindowsrawallconfqueryoverlay/cacheparity/gates/timing/hashes, noGTcropping/countfilter. Inspectsemanticfailuresstillpossibleevenifnumericparitypasses. Donotmodifyproductionfactory ordatasets,do notpromotecheckpoint. No newheads/training necessaryunlessauditnewconcreteissue. '
        '12hdeadline18:41:41UTC/trainstop18:11:41UTC,heartbeat15minACTIVE/sharedGPUlock. Protect24excluded135pendingR12oldmasklockedGR2UR2UR3dirtyworktree. Noactivejobs. Updateplanstatecoherently,donotrerunoldrecorders.')
    note='''

## 2026-10-05 native topology control: no clear advantage

`native_topology_control_v1` stopped before training because21 synthetic frames have kind=`curve`, so literal kind=`synthetic` missed them. Failure preserved; v2 correctly includes132 synthetic+21 curve-kind synthetic single images, still supervising only nose/fourfeet. Native pool1223TRAIN/112VAL, frozen original baseline17, same800step schedule and coordinate loss,1119744 trainable adapter parameters in both arms. No encoder training. `full` versus `structured_union` changes only new-branch self-attention topology.

| Metric px | Baseline VAL | Full VAL | Structured VAL |
|---|---:|---:|---:|
|Real reliable|33.246|33.267|33.374|
|Bottom feet|48.461|46.296|46.177|
|Synthetic single|19.225|18.376|18.616|
|Synthetic dual|23.299|22.420|22.398|

TRAIN full/structured real28.575/28.764,bottom27.782/27.977,synsingle15.136/15.201,syndual21.073/21.116. The single synthetic TRAIN pool is153images here, not461 from prior candidate experiments; cross-experiment numbers are not controlled comparisons. Both architecture arms nearly identical; no extension/promotion. This does not disprove complete GroupPose architectures.

All frozen-path tests pass: initial residual exact0, default-off and native-source golden<2e-5, after-training untargeted18points exactunchanged. GPU cached training about24seconds/arm with~3.8GB peak allocated including caches; this excludes original1B RGB encoding. All100-step checkpoints retained. Visual row34 still has duplicate query feet on abdomen; dualrow61 feet remain clustered. Sampled last100 loss slightly above previous100, not evidence for blind step extension.

Reviewed prior heatmap failure before considering new losses: `output/core23_spatial_cycle_20261004/review_reliable_heatmap_v1/decision.json` rejected matched1600-step continuation with.002GaussianKL/sigma12px on reliable points in all3seeds (both real and dual degraded). Do not repackage the same sigma/weight sweep as a new hypothesis.

Next: pause new model training and verify actual-RGB integration of already-promising bounded candidate heads. Fixed seed17 unary gives independent local fallback; geometry overrides only eligible supported points, otherwise retains unary when local enabled. Both options default off, allqueries/boxes/back/tail unchanged. Establish cache parity, 1/2/allviews, ordering/duplicates/missing support and RGB runtime cost. Unique confident query is only a restricted single-subject association heuristic, not proven multi-mouse identity. No GTcount in forward, no automatic deployment. Reused development data cannot prove external generalization.

Windows report: `C:/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005/native_topology_control_v2/index.html`.
'''
    doc=ROOT/'docs/singleview_gated_fusion_geometry_direction_20261005.md';text=doc.read_text(encoding='utf-8')
    if '## 2026-10-05 native topology control: no clear advantage' not in text:doc.write_text(text+note,encoding='utf-8')
    shutil.copy2(doc,DEST/doc.name)
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='bounded_candidate_rgb_integration_next',active_job=None,active_pid=None,updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),next=next_action)
    for key in ['native_topology_visual_review','prior_heatmap_negative_rechecked']:
        if key not in state['completed']:state['completed'].append(key)
    write(CYCLE/'state.json',state);plan=json.loads((CYCLE/'plan.json').read_text());plan['next_command']=next_action;write(CYCLE/'plan.json',plan)
    index=DEST/'index.html';page=index.read_text(encoding='utf-8');link='<p><a href="native_topology_control_v2/index.html">最新：原生注意力拓扑负对照、默认关闭及非目标点保护</a></p>'
    if link not in page:index.write_text(page+link,encoding='utf-8')
    print(state['stage'])


if __name__=='__main__':main()
