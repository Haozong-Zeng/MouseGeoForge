"""Record read-only geometry audit and rejected coordinate-level fusion."""
import datetime as dt
import json
import shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005'
DEST=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005')


def write(path,data):path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def main():
    audit=json.loads((CYCLE/'camera_geometry_audit_v2/results.json').read_text())
    trial=json.loads((CYCLE/'prediction_geometry_pilot_v1/results.json').read_text())
    assert audit['scenes']==249 and audit['max_target_xy_normalized_delta']==0
    next_action=(
        'Camera audit and prediction-only LOO COMPLETE; no active CPU/GPU jobs. '
        'Read docs/singleview_gated_fusion_geometry_direction_20261005.md updated07:42UTC. '
        'NEW/camera_geometry_audit_v2 has train_cameras.pt225scenes, validation_cameras.pt24scenes, role-local row_indices into SOURCE singleview_fusion_inputs_seed17_v1 train1223/val112. All249 scenes single-instance cohort; 1181train/80valimages. Audit K/E/GTxy parity exactly0; load_item canonicalize_camera=False. v1failed3x4vs4x4shape only, preserved; v2correct. Synthetic 2D projects from instance_keypoint_world_xyz within~0.00005px, not alternate surface_world_xyz_instances(~2.3px); field name does not prove anatomical internaljoint semantics. Realacceptedreliableepipolarmean TRAIN7.56/VAL6.63px, back12.90/11.33. camA-F trainfeetmedian17.75px; cause not isolated, doNOTeditcalibration/labels or dropviewsbasedonVAL. '
        'NEW/prediction_geometry_pilot_v1 completed fixedno-training weightedLOO,robustLOO,boundedblend with rawgallery all7cams. Functions mousegeoforge/prediction_only_triangulation.py support-only targetexcluded, uniqueconfidentquery, >=3robustinliers,16px,2degree,50percentblendonlyshift<=48px. 7tests+camera9passed. GTonlypostforwardeval. Baseline/weighted/robust/bounded real33.246/45.753/36.985/33.169px; camAfeet48.461/72.454/55.781/47.440. Boundedchangesonly4of32camApoints and66of175reliable; tiny result insufficient, no adoption orVALthresholdsearch. TRAINreal30.508->30.203tiny. VisualcamA shows correcthindfootpulledontorso byrobust,camFfootclusterhead persists. Dualsnotincluded; synVAL2viewsfallbackcannotclaimfusionbenefit. Back/tailnotchanged. '
        'NEXTspatialprobabilitydiagnostic: reextract fixed baseline seed17 heatmaps from OLD/single_view_train_context_v1 and single_view_validation_context_v1 nativeFP32 caches, orSOURCEtokens/coarse plus native memory, using OLD/full_context_pose_control_v1/full_seed17_latest.pt and SOURCEpredictedboxes. Exactsame48x48map probability reproduction as tools/prepare_singleview_fusion_inputs_20261005.py; verify probability expectedXY parity toSOURCE. No encoderretraining. AcquireGPUlock output/teacher_temporal_cycle_20261001/gpu_job.lock ifGPU; boundeddeadline. Savecompact5reliablepoints maps only andpredictedqueries with metadata plus goldenparity, nothugeallimage/tokensduplicates. Examine whether wrongmean liesbetweenseparatepeaks orallmapmasswrong. GTmass/rankdiagnostic isPOSTFORWARDonly, neverusedtochooseinferencequery orcrop. Then geometryguided probabilitymap/candidate reweighting insteadofhardcoordinateaveraging; readprimaryimplementationifborrowing, do notclaimpaperreproduction. Fixedsingleinstancecohort, unknownidentitymustabstain, off/V1/missing/duplicate/permutation tests. Chooseanyhyperparameters onTRAIN only, nooptimizingreusedVAL, nofake3DGT. Current12hheartbeat15minACTIVE; cutofftraining18:11:41UTC/hard18:41:41UTC. Originalannotations, excluded24groups/pending135/R12oldmask/lockedGR2UR2UR3 unchanged; no newmodelpromotion. Updatefreshstate andplan after anynext job; do not rerun old record scripts that revertnext.')
    decision=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),decision='Coordinate-level geometry not adopted; inspect spatial probability evidence next.',
        camera_audit=audit,prediction_geometry=trial,visual_checked=['camA','camF'],production_changed=False,next=next_action)
    write(CYCLE/'geometry_pilot_decision_v1.json',decision)
    shutil.copy2(CYCLE/'geometry_pilot_decision_v1.json',DEST/'geometry_pilot_decision_v1.json')
    shutil.copy2(ROOT/'docs/singleview_gated_fusion_geometry_direction_20261005.md',DEST/'singleview_gated_fusion_geometry_direction_20261005.md')
    index=DEST/'index.html';page=index.read_text();link='<p><a href="prediction_geometry_pilot_v1/index.html">最新：相机审计后的预测驱动几何实验与各视角对比</a> | <a href="geometry_pilot_decision_v1.json">完整审计和决策</a></p>'
    if link not in page:index.write_text(page+link,encoding='utf-8')
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='coordinate_fusion_rejected_probability_diagnostic_next',active_job=None,active_pid=None,updated_utc=decision['utc'],next=next_action)
    if 'prediction_only_geometry_pilot_and_visual' not in state['completed']:state['completed'].append('prediction_only_geometry_pilot_and_visual')
    write(CYCLE/'state.json',state)
    plan=json.loads((CYCLE/'plan.json').read_text());plan['next_command']=next_action;write(CYCLE/'plan.json',plan)
    print(json.dumps(dict(stage=state['stage'],report=str(DEST/'prediction_geometry_pilot_v1/index.html'))))


if __name__=='__main__':main()
