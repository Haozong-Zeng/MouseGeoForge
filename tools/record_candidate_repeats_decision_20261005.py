"""Persist completed robustness work and next bounded TRAIN-only context task."""
import datetime as dt,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';DEST=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005')


def write(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def main():
    summary=json.loads((CYCLE/'candidate_repeats_review_v1/summary.json').read_text())
    next_action=(
        'Candidate stress/RGB/seed repeats COMPLETE. RGBsession16032/PID3765804 exited; repeatssession2061/PID3765928 exited; no activeGPU/CPUjob. '
        'NEW/epipolar_candidate_stress_v1 fullvsreversevsduplicate exact0; V1and2view exactbase. Fixed target+first2physicalsupport threeviews real27.47px/full30.56; bottomthree42.59/full35.80; deletingC/Dsupportbottom47.64/46.02. DoNOTselectbestcameraonVAL. '
        'NEW/epipolar_candidate_rgb_audit_v1 replayedfirstVALrealgroup5qualifiedviews+firstsyn2=7images, memory/instance exact0, finalcandidatemax5.3644e-7norm. Same independentV1 encoder+originalMLP+ROI mixed17+posefull17; cacheonlycomparedafterinference, nofeaturesfedfromcache. 15.34sectotalinclmodelload. '
        'NEW/epipolar_candidates_seed_repeats_v1 freshlydecodedseed29/43 all138tokens fromOLDsingleviewnativecaches, samefixedrule5peaks/local3x3/sigma16/geometryweight1 vsunary0; sameTRAIN1181/VAL80cohort, immutableheadcheckpoints. Baselinebox/posegoldenpass. VALseed17/29/43 reliable33.25->30.56,33.31->29.75,34.19->29.98; bottom48.46->35.80,51.29->47.99,47.68->40.97. TRAINbottomallworse32.92->34.19,34.80->36.35,35.79->37.07. No promotion. Repeatpredictionsoutsidecohortretain17values, MUSTmaskcohort; notcreditdual. Saved seed29/43VALprobabilitymaps only; TRAINmapsnotstored, canreplayifneeded. Reference17probmapsinNEW/singleview_probability_diagnostic_v1 all6queries5points. '
        'Windows+NEW/candidate_repeats_review_v1 has3seedrawcamA/C/Fgallery; inspectedcamA,18recoversfootall3butforefoot16stillambiguous29. summary.json; inventorysynthetic_train_view_inventory.json confirmedload_sceneactualcamera IDs, notduplicatefilecount:69synTRAINscenes22x4views44x8views3x7. First66currently2views;308existingadditionalTRAINimages, allcamera_present. '
        'NEXT prepare FULLexisting syntheticTRAINcontexts fromthatinventory, no renderneeded, noVALsourceadded. For69scenes retainexactsourceframe/actiongroup/role, use available_views from load_scene; originalSOURCE metadatarecord fromNEW/singleview_fusion_inputs_seed17_v1/train.pt and commongeometry fromNEW/camera_geometry_audit_v2/train_cameras.pt. Existing2views oldnormalizationscale fromg.scale; use record normalization_scale=g.scale forwholefullscene tokeepunitgauge. load_itemfullscene forK/E/targets andoldviewGTxy/validparity. No labelchanges, confirminstancecount/availablefullannotation independently; doNOTmarkmissinglabelsnegative. Preserve originalimage_hw768basedresolution. '
        'Reuse existing frozenV1 rows foroldviews orreplaywithgolden; encodeonly308newviewimages withimmutableencoder+initialheads. Script validate_epipolar_candidate_rgb_20261005.py implementscorrectRGBhooksandnumericroute; DO NOTinvokeitsmainagainorolddeadlines. NEWfreshscript/output, acquire output/teacher_temporal_cycle_20261001/gpu_job.lock andnewcycle cutoff. Encoder vggt-omega/vggt_omega_1b_512.pt; initial output/surface_curve_cycle_20261003/training_v1/pilot/initial_heads.pt; baseline17 originaldetector OLD/detection_features_v1/initial_detector.pt, ROI OLD/detection_view_context_control_v1/mixed_seed17_latest.pt, pose OLD/full_context_pose_control_v1/full_seed17_latest.pt. Encoderbf16TF32off/headFP32TF32on; nativeFP32->half->48map->predROIhalf. Captureall138posequeries thencompact5maps; noGTcrop/queryselection. Savepredboxes/logits/XY/maps/candidatefeatures with separatelystoredGT andcamera metadata/sourcehashes; avoidlargeRGBorencoderfeaturedup. '
        'Audit syntheticfullview reprojection beforetraining. Existingtwo-viewVALsynfallback isNOTfusiongeneralization; noVALcameraorparamsearch. Then designboundedmatched candidate-ranking/soft reliability learning using acceptedTRAINlabels only, baselinecoordinate asfallbackoption, localappearance/heatmapcandidate quality plusgeometry vs sameparamno-geometry; originalback/tail unchanged. LossGTonly, noGTvisibilityforward, unknownmultiinstanceabstain. Initialseed17pilotbefore3seeds. Do not endlesslyretunehardcoordLOO/failedsemantic-only. Readupdateddocs/singleview_gated_fusion_geometry_direction_20261005.md. Deadlinetraining18:11:41UTC/hard18:41:41UTC; heartbeat15minACTIVE; protectedannotations24exclusions135pendingR12oldmasklockedGR2UR2UR3/production unchanged. Updateplanstatecoherently, neverrerunolderrecordscripts.')
    result=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),decision='Three-seed development improvements reproduced but TRAIN bottom degrades; do not promote. Expand existing synthetic TRAIN contexts before a new learning control.',summary=summary,next=next_action,production_changed=False)
    write(CYCLE/'candidate_repeats_decision_v1.json',result);shutil.copy2(CYCLE/'candidate_repeats_decision_v1.json',DEST/'candidate_repeats_decision_v1.json');shutil.copy2(ROOT/'docs/singleview_gated_fusion_geometry_direction_20261005.md',DEST/'singleview_gated_fusion_geometry_direction_20261005.md')
    pagepath=DEST/'index.html';page=pagepath.read_text();link='<p><a href="candidate_repeats_review_v1/index.html">最新：三种子、视角压力测试和原图重放</a></p>'
    if link not in page:pagepath.write_text(page+link,encoding='utf-8')
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='synthetic_train_context_expansion_next',active_job=None,active_pid=None,updated_utc=result['utc'],next=next_action)
    for item in ['candidate_cached_view_stress','candidate_three_seed_visual_review','synthetic_existing_train_view_inventory']:
        if item not in state['completed']:state['completed'].append(item)
    write(CYCLE/'state.json',state);plan=json.loads((CYCLE/'plan.json').read_text());plan['next_command']=next_action;write(CYCLE/'plan.json',plan)
    print(json.dumps(dict(stage=state['stage'],report=str(DEST/'candidate_repeats_review_v1/index.html'))))


if __name__=='__main__':main()
