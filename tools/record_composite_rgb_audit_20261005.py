"""Persist the inspected RGB integration result and the next bounded audit."""
import datetime as dt
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CYCLE = ROOT / 'output/singleview_fusion_cycle_20261005'
DEST = Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005')


def main():
    review = json.loads((CYCLE / 'composite_candidate_rgb_audit_v1/review.json').read_text())
    assert review['max_composite_cache_difference'] < 2e-5
    next_action = '''Composite candidate integration is COMPLETE and visually inspected. No active jobs. Do not rerun old recorder scripts.
Files: mousegeoforge/models/bounded_candidate_fusion.py (7 tests), tools/audit_composite_candidate_cache_20261005.py (load_wrapper helper), tools/validate_composite_candidate_rgb_20261005.py, tools/review_composite_rgb_20261005.py. Output composite_candidate_cache_audit_v1 and composite_candidate_rgb_audit_v1. Fixed bounded_prior_candidate_pilot_v1 unary/geometry seed17 heads, frozen original pose/detector/encoder. Both flags default OFF exact original; local requires unique confident query; geometry requires >=2 OTHER distinct unique-query cameras and otherwise falls back to local. Geometry candidate0 deliberately overrides local with original. Only reliable five points updated; all boxes/other18points unchanged; no labels/count in forward.
Cache80 validation: baseline/local/composite real33.246/30.438/26.324px, bottom48.461/41.197/39.704, single synthetic19.225/18.063/18.063. Reused development, conditional GT-box matching after forward, not independent-test/AP. Composite real P95 102.61 vs baseline105.91; bottom P95 worsens107.45->121.14; real11 points worsen>16px, worst86.05. No promotion.
Fresh audit14 uniqueRGB/15 independentencoder calls: first2real scenes AZR2L2 frames0,519 (5views each), firstsingle WashFace frame1 (views2,0), firstdual frame352 (views2,0). Fresh pose/composite vs cache max5.364e-7, logits2.861e-6; repeatedRGB, reverse/duplicates and V1/V2 local fallback pass. Subsets combine fresh independent outputs, not separately encoded joint contexts. Sum14forward1.616s,totalwall13.56s includes loading; not fair comparison to jointencoder. Dual2views both predict2 and abstain exactly, not general identity proof. Windows composite_candidate_rgb_audit_v1/index.html; actualRGB exports and fresh_groups.pt retained.
Visual: group00 bottom local/composite recover two foot positions from torso toward palms; group01 bottom still collapses frontfoot15/16 near one paw; sideview5 detection box covers equipment and misses much of mouse, feet still cluster near head; dualtop unchanged with inaccurate paw positions. Numeric parity is engineering success, not semantic correctness.
NEXT no new training yet. Predetermine first TWO existing dual validation scenes in singleview_fusion_inputs_seed17_v1/validation.pt metadata and first single synthetic scene, then inspect all available physical views, not just old pair. Use load_scene inventory and exact load_item preprocessing. Encode each RGB independently with FrozenSingleViewCandidateRuntime; pass only images, all6queries. Obtain calibrated cameraP from full-view load_item batch with COMMON scene normalization; verify K/E projection against synthetic GT only as posthoc engineering check, no fitted calibration or GT inputs. Do not assume camera0 is bottom or all cameras perspective. Need inspect camera audit implementation for exact batch fields/gauge. Reused pair should retain RGB/pose parity; additional views are stress-test coverage, not TRAIN expansion.
Audit local/fusion/composite gates and errors for V1,V2,all,reverse,duplicate; overlay every query>=.5 with raw RGB. Crucial failure mode: different cameras may miss different mice and each predict one, so unique-query gate does not establish same-instance association. Never equate query index across cameras or provide GT identities/count to inference. Report any unsafe fusion; consider conservative abstention only after documenting concrete evidence. No production defaults or checkpoint promotion. Save hashes, actualRGB, allview detection counts, images, metrics and Windows report. Preserve locked GR2/UR2/UR3, excluded24/pending135, R12 oldmask, tail/abdomen/original labels and user dirty worktree. Avoid repeated heatmap/head sweeps already negative.
Deadline unchanged: training cutoff18:11:41UTC, hardstop18:41:41UTC on2026-10-05; 15min heartbeat pymousestudio, shared GPUlock output/teacher_temporal_cycle_20261001/gpu_job.lock. Check fresh time/lock/processes before newjob. Stop own jobs and pause automation at deadline, no extension.'''
    note = '''

## 2026-10-05 composite readout: actual RGB parity, remaining semantic failures

Implemented isolated default-off `BoundedCandidateFusion`: local bounded unary selection works independently; supported geometry overrides it, including choosing the original candidate. Unsupported geometry retains local output. Only nose/fourfeet can change. Seven unit tests cover exact fallback, ordering/duplicates, invalid support, initial equality and frozen inference. No production integration or deployment.

On the reused 80-image development cohort, baseline/local/composite mean errors are real33.246/30.438/26.324px, bottom48.461/41.197/39.704px, synthetic-single19.225/18.063/18.063px. These are model-input pixels and conditional pose metrics, not end-to-end AP or external-test evidence. Mean gains do not eliminate outliers: composite bottom P95 is121.14 versus107.45 baseline;11 real points worsen by>16px, worst86.05px.

Fresh RGB replay used14 predetermined unique images and15 independent encoder calls. Maximum pose/composite cache difference5.364e-7 normalized; logits2.861e-6. View-order/duplicate and insufficient-view fallback checks pass. Subset tests reuse these fresh independent outputs, not joint-context re-encoding. Summed14 forward timings1.616s; total13.56s includes initialization/loading. No fairness claim against a joint multiview encoder. The two dual views each predict two queries and abstain; this does not establish robust multi-mouse association.

Visual inspection of four panels: first real bottom frame recovers feet from torso toward palms; second bottom frame still puts frontfoot15/16 near one paw; sideview5 retains a badly placed box and clustered feet; dual top is unchanged and imperfect. Cache/RGB parity is engineering evidence, not anatomical correctness.

Next bounded audit: all existing physical views of the first two dual validation scenes and first synthetic-single scene, fixed before seeing predictions. Verify camera gauge and RGB parity; examine whether missed mice create misleading unique-query cross-camera fusion. Do not infer instance identity from query index or unique detection. No new training or production promotion yet.

Report: `C:/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005/composite_candidate_rgb_audit_v1/index.html`.
'''
    doc = ROOT / 'docs/singleview_gated_fusion_geometry_direction_20261005.md'
    content = doc.read_text(encoding='utf-8')
    if '## 2026-10-05 composite readout:' not in content:
        doc.write_text(content + note, encoding='utf-8')
    shutil.copy2(doc, DEST / doc.name)
    for filename in ['state.json', 'plan.json']:
        path = CYCLE / filename
        data = json.loads(path.read_text())
        if filename == 'state.json':
            data.update(stage='allview_multi_instance_association_audit_next', active_job=None,
                        active_pid=None, next=next_action,
                        updated_utc=dt.datetime.now(dt.timezone.utc).isoformat())
            for item in ['composite_candidate_wrapper_7tests', 'composite_candidate_cache_audit',
                         'composite_candidate_rgb_visual_review']:
                if item not in data['completed']:
                    data['completed'].append(item)
        else:
            data['next_command'] = next_action
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    index = DEST / 'index.html'
    content = index.read_text(encoding='utf-8')
    link = '<p><a href="composite_candidate_rgb_audit_v1/index.html">组合读出：真实 RGB 复跑与剩余足部错误</a></p>'
    if link not in content:
        index.write_text(content + link, encoding='utf-8')
    print('Composite audit recorded; next: all-view association stress, no active jobs.')


if __name__ == '__main__':
    main()
