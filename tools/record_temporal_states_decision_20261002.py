"""Record paired temporal results and newly diagnosed input-encoding mismatch."""
import datetime as dt,json
from run_teacher_cycle_dlc_20261001 import OUT,write_state

def main():
    result=json.loads((OUT/'temporal_joint_states_v1/summary.json').read_text())
    imageaudit=json.loads((OUT/'temporal_joint_states_v1/image_parity_audit/summary.json').read_text())
    manifest=json.loads((OUT/'temporal_paw_peaks_v1/manifest.json').read_text())
    report=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),result=result,imageaudit=imageaudit,
        cached_seed_clips=len(manifest),cached_model_parity_max_px=max(r['parity_max_px'] for r in manifest),
        conclusions=[
            'Actual LP3seed top3candidatecache finished for4clips,12seedclips,24camera-streams perseed. Exactnormalization and same-video modelparityguards passed. No teacher training.',
            'Temporal finite-state max-sum and independent zero-weight control use identical candidate states, with unknown states, nointerpolation, fixedcameras, heldcameraexcluded. Synthetic tests cover missingframes,fastsinglehypothesis and exhaustivepathverification.',
            'Pilot temporal adds4acceptedmanualobservations;stress adds2. On commonaccepted41pilot/59stresspoints, predictions/errors unchanged; noidentitycorrection>1px. Severe>100pxerrors remain3pilot/6stress. AggregateP90change reflects changedcohort, not improvedcommonpoints.',
            'ReviewedFR2f779 andAZR2L2f1169 heldA/Fpanels. Existingpawdiscrepancies persist,includingcontralateralhindpaw case. Full52fpssource/26fpsplaybackLOOvideos exported. No dense3DGT, therefore trueactionlag notvalidated.',
            'SparsePNGjoint baseline and videojoint baseline NOT interchangeable:29acceptance disagreements,maxcommonerrorchange11.70px. Audit84matchedimages:0byte-identical,medianabsolutechannelerror2.714/255;all84sameindexbestamong+/-2neighbors.',
            'Provenance explains input-encoding mismatch: tools/export_pymousestudio_rectified.py exportsRGBvideos H264CRF16yuv420p but annotationPNGs directlyfromrawcv2remap. FFV1derivedclips preserve the decodedH264, not originalPNGpixels. No evidenceofframeoffset in this local5framecomparison; not proof of all-camera synchronization.',
            'Within-current-video pairedtemporalcomparison remains valid; do not attribute earlierstatic-vs-video differences to temporalarchitecture. Need lossless raw-remapclip control before moremotioncost experiments.'
        ],
        next_actions=[
            'Create NEW output-only lossless clips for existing windows from original PyMouseStudioRaw videos. Read sequence provenance and original source_calibration; keep identicalK,D,cv2remapINTER_LINEAR/BORDER_CONSTANT. Apply lensmapping ONCE to unrectifiedrawsource, never to alreadyrectifiedsessionvideos. Originalinputsandcamerasunchanged.',
            'Use same704x768letterbox(PILLANCZOS691x768,left6/right7). At all84manualsampleframes assert rawremap exactlyequalsoriginalreturned1080x1200PNG and letterbox exactlyequalsderivedteacherPNG. Verify FFV1pixelroundtrip. Abort ratherthanrelax exactpixelgate if mismatch.',
            'If exactparity passes, cache new3seed heatmappeaks in separatefolder and rerun SAME frozengeometry/temporalsettings, no tuning. Compare staticvsdenseindependent parity again, then temporalon/off.',
            'Do not expanddensepseudo-labels or addteachersteps based on this mixedencodingcomparison. Keep24pendinggroups plusSEM01outofwindows,holdoutclosed,stopnewexperiments07:21:52UTC.'
        ],
        visual_review='C:/Users/55395/Downloads/MouseGeoForge-TeacherTemporal-20261001/temporal_joint_states_v1',
        locked_holdout_opened=False,original_labels_or_calibration_changed=False,pseudo_gt_released=False)
    p=OUT/'heartbeat_20261002_0310_decision.json';p.write_text(json.dumps(report,indent=2))
    write_state(stage='temporal_states_evaluated_encoding_mismatch_diagnosed',pid=None,active_run=None,mode='idle_between_heartbeats',job_deadline_utc=None,
        latest_decision=str(p),latest_result=str(OUT/'temporal_joint_states_v1/summary.json'),next_action=report['next_actions'][0],pseudo_gt_released=False)
    print(p)

if __name__=='__main__':main()
