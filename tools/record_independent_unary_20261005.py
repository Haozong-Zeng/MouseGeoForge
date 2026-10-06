"""Persist frozen unary audit without altering prior experiment artifacts."""
import datetime as dt
import json
import shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005'
DEST=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005')


def write(path,obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def main():
    audit=json.loads((CYCLE/'independent_unary_eligibility_audit_v1/results.json').read_text())
    assert all(v==0 for k,v in audit['tests'].items() if 'difference' in k)
    next_action=(
        'Independent unary eligibility audit COMPLETE, no new training, no active GPU jobs. '
        'Read independent_unary_eligibility_audit_v1/results.json, predictions.npz, role_arm_points.json and Windows same-name report. '
        'Frozen local_candidate_selector_pilot_v1/unary_latest.pt step800. Independent gate requires exactly one predicted query score>=.5 in this image; no othercamera condition, no GTcount. Geometry explicitlyzero. '
        'TRAIN baseline->independent real30.508->26.588, bottom32.920->30.045, syn26.359->19.175. VALreal33.246->29.511, bottom48.461->36.122, syn19.225->17.451 (160points117changed,14improve>16,9harm>16,maxharm41.50). Real sameas matchedgate. '
        'Single-image vsbatch/permutation/duplicates/NaNgeometry ignored all EXACTzero; oldGPU output replayexact. No multi-instance claim. Views0 synthetic notbottom. Visually inspected row060view2 rearfoot17 shifted upward onto torso (22.06->63.57px); row063view0 frontpaws improved frombody/lowerlimb toface/washing posture. '
        'NEXT frozen dual-instance diagnostic, notmoretrainingyet. SOURCE=singleview_fusion_inputs_seed17_v1/validation.pt has112 images; 32dual missingfromcandidatecache. Source metadata expected_instances>1 identifies evaluationcohort only; NEVERfeedGTcount/matchedquery beforeforward. Use OLD=output/core23_spatial_cycle_20261004 single_view_validation_context_v1/manifest.json nativeFP32memory/instance_tokens and full_seed17 pose+mixed_seed17 detector or existingSOURCE tensors. DecodeALL138queries, notGTselected, probability48grid forKP=[0,14,15,16,17], local_candidate_inputs usesfullmaps. Can instantiate only DecoderSpatialPoseReadout and SpatialDetectionReadout; no need1Bencoder ifcached, SOURCEdet outputs alreadygolden. Pose spatialreadout canreuse SOURCEtokens/coarse with native memory.half().float() resized48, predictedboxes togenerateprobability; assert expectedxy==SOURCE.xy, samplelocalfeatures. Candidatebuilder build_selector_inputs with V1 returnsineligible thenoverrideeligibility. '
        'Compare unchangedbaseline vs unique-query gate (abstainswhen2predqueries) vs independentper-query score>=.5 gate as explicitOODdiagnostic ondualVAL32. All6queries kept, noGTmatch preforward, noGTcounts, no crossviewqueryID assumption. Oldsource tensors objectness_logits vs packed logits names differ. Use postforward GTbox assignment only; reportallvalidnose/footpoints, gatedPCK32, changedharm, detectioncounts. Renderallconfqueries, checkcross-instancejumps. Preserveback/tail. No train/threshold/checkpointselection onVAL. '
        'Then decide bounded nativeprobability-prior residual ranking (notimplementedyet) insteadofunconstrainedgeometry: baselineoption0, priorlogpeakmass, boundedlearnedresidual limitpredeclaredorTRAINonly; matched800sameinit/schedule control. DoNOTblindlyextendfailedgeometry with61.16bottomerror. Unarynotpromoted. '
        'Packed TRAIN1489(real1028+fullsyn461) andVAL80 in local_candidate_selector_pilot_v1; oldtrain_valid includesmatchedsupportgate so recompute supervision iftrainingindependentgate. Fullcachedcontexts ready. Originalannotations/24excluded/135pending/R12oldmask/lockedGR2UR2UR3/dirtyworktree unchanged. '
        'Heartbeat15min ACTIVE; deadline18:41:41UTC andtrainingcutoff18:11:41UTC unchanged. SharedGPUlock output/teacher_temporal_cycle_20261001/gpu_job.lock forGPUwork. Updatecoherentstate; do notrerunoldrecordscripts.')
    note='''

## 2026-10-05 independent unary eligibility audit

Frozen 800-step unary weights, no extra training: remove the requirement for other cameras and enable only the unique predicted query with score >= 0.5. No GT count, correspondence or visibility in forward. Geometry is explicitly zero. All labels are consulted only after prediction.

| Input-pixel mean error | Baseline | Old matched-support gate | Independent gate |
|---|---:|---:|---:|
| TRAIN real reliable |30.508|26.588|26.588|
| TRAIN real bottom feet |32.920|30.045|30.045|
| TRAIN synthetic reliable |26.359|19.154|19.175|
| VAL real reliable |33.246|29.511|29.511|
| VAL real bottom feet |48.461|36.122|36.122|
| VAL synthetic reliable |19.225|19.225|17.451|

The previously fallback-only synthetic validation is now activated. 117/160 points changed: 14 improve by >16px, 9 worsen by >16px, worst added error 41.50px. Real bottom 11 improve and 2 worsen by >16px. Full CPU versus previous GPU output replay, singleton/batch, permutation, duplicates and ignored NaN geometry all have exactly zero coordinate differences. Default-off remains exact.

Visual inspection: row060 view2 foot17 moves incorrectly from the hindfoot vicinity up onto the torso (22.06 -> 63.57px); row063 view0 front feet move from the lower body into the face-level washing posture, near the existing annotations. These are diagnostic extrema, not randomly representative cases. Synthetic view0 is not a physical bottom-view claim.

Decision: independent V1 unary remains promising, but not promoted. Next evaluate existing 32 dual-instance development images with frozen weights and prediction-only gates before new training. This reused development set is not an external generalization test. No GT-count/query matching in forward; report conditional pose metrics and unchanged detector counts separately. Avoid extending failed unconstrained geometry solely because CE is descending. Consider a TRAIN-predeclared bounded native-probability prior afterwards. Original labels, tail/back, locked sessions and production defaults unchanged.

Artifacts: `output/singleview_fusion_cycle_20261005/independent_unary_eligibility_audit_v1`; Windows report in same-name subfolder of `C:/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005`.
'''
    doc=ROOT/'docs/singleview_gated_fusion_geometry_direction_20261005.md'
    text=doc.read_text(encoding='utf-8')
    if '## 2026-10-05 independent unary eligibility audit' not in text:doc.write_text(text+note,encoding='utf-8')
    shutil.copy2(doc,DEST/doc.name)
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='dual_instance_unary_diagnostic_next',active_job=None,active_pid=None,updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),next=next_action)
    if 'independent_unary_gate_audit_and_visual' not in state['completed']:state['completed'].append('independent_unary_gate_audit_and_visual')
    write(CYCLE/'state.json',state)
    plan=json.loads((CYCLE/'plan.json').read_text());plan['next_command']=next_action;write(CYCLE/'plan.json',plan)
    index=DEST/'index.html';page=index.read_text(encoding='utf-8');link='<p><a href="independent_unary_eligibility_audit_v1/index.html">最新：真正单视角启用检查及合成预测对比</a></p>'
    if link not in page:index.write_text(page+link,encoding='utf-8')
    print(state['stage'])


if __name__=='__main__':main()
