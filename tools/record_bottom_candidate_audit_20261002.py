"""Preserve frozen diagnostic results, flag one review group, and record next work."""
import datetime as dt
import json
import shutil
from pathlib import Path
from bottom_anchor_common_20261002 import ROOT,OUT,budget,write_state,exclusions


def main():
    plan,_,check=budget(120);check()
    src=OUT/'anchor_candidate_audit_v1';summary=json.loads((src/'summary.json').read_text())
    rays=json.loads((src/'manual_ray_audit.json').read_text())
    review=dict(id='BA01',session='FR2',frame=1039,camera='B',ids=[16,18],
        status='needs_human_review_not_automatically_corrected',
        evidence=[r for r in rays if r['session']=='FR2' and r['frame']==1039 and r['camera']=='B'],
        visual='CamB human16 lies near the bottom15 epipolar line, while human18 lies near bottom16 line. Other views and nearby frame1104 make an identity conflict plausible, not certain. Calibration/localization/synchronization cannot be excluded solely by this test.',
        exclusion='Whole FR2 frame1039 source group and any temporal context containing it excluded from FUTURE experiments; original files and previous metrics retained unchanged.',
        proposed_correction=None,approved_for_training=False)
    (src/'BA01_pending_review.json').write_text(json.dumps(review,indent=2))
    assert ('FR2',1039) in exclusions(plan) and ('AZR2L2',584) in exclusions(plan)
    assert summary['side']['severe_raw_over100']==sum(summary['side'][k] for k in [
        'severe_recoverable_within15_samepair','severe_recoverable_within15_only_otherpair','severe_no_candidate_within15'])
    decision=dict(time_utc=dt.datetime.now(dt.timezone.utc).isoformat(),experiment=src.name,
        quantitative=dict(bottom_points=191,complete_bottom_pairs=95,clear_bilateral_swap_pairs=2,
            clear_swap_cases=[['AZR2L2',129,'15/16'],['FR2',1104,'15/16']],
            bottom_over100=11,bottom_over100_without_any_candidate_under15=7,
            side_over100=36,side_recoverable_samepair_under15=7,side_recoverable_otherpair_only_under15=4,
            side_no_candidate_under15=25,manual_ray_sampson_median_px=summary['manual_same_id_sampson']['median'],
            manual_ray_sampson_over15=163,manual_ray_n=575),
        visually_inspected=[
            dict(file='bottom_ranked_identity_audit.jpg',finding='AZR2L2 frame129 and FR2 frame1104 clearly exchange forepaw15/16 against bottom human labels. FR2 frame584 is NOT a clean swap: both predictions collapse near one paw. Other listed bottom poses retain identity despite localization errors.'),
            dict(file='FR2_001039_allview_manual_rays.jpg',finding='New manual label identity concern BA01 at camB. The large manual-anchor LOO error cannot be confidently attributed to model correction alone; retain old metric but quarantine entire source group for future tests.'),
            dict(file='FR2_001104_allview_manual_rays.jpg',finding='Bottom teacher forepaw swap contrasted with side camB human15/16 points that lie near their corresponding bottom manual rays; confirms bottom prediction can corrupt otherwise sensible side identities.'),
            dict(file='side_candidate_ceiling_examples.jpg',finding='Distinct categories visible: same-pair candidate on correct paw; candidate located on correct paw but under front/hind wrong ID; and missing candidate where all four predictions miss the labeled target. Reassignment alone cannot solve the last category.')],
        decision='Do not train longer or propagate correction labels yet. Prioritize frozen native temporal-teacher heatmap candidate recall audit to determine whether missed candidates exist as secondary peaks; identity logic only makes sense when a correct candidate exists. Keep human-bottom and automatic-bottom routes separate.',
        next_action=[
            'Read current plan and common exclusions(plan); enforce all24 original D/C groups, SEM01 and NEW BA01 FR2 frame1039, including any intersecting context window. Do not regenerate old metric files.',
            'Use the frozen native LP MHCRNN temporalON checkpoint and exactly the existing PNG-normalization/context pipeline. Extract a fixed top3 local heatmap peaks per paw from both branches if required by the official selected-output logic; document selection and decoder parity with cached uv_selected. No retraining or score/threshold sweep.',
            'Evaluate groundtruth-based candidate recall ONLY as an optimistic diagnostic: top1 baseline vs same-ID top3 vs same-pair pooled top3. Report all eligible points, source-session breakdown, missing candidates and semantic swaps; GT must never guide actual matching/inference.',
            'Run one bounded GPU job using shared flock and local/global deadlines if inference is needed; reuse caches where parity/provenance holds. Never use holdout images or modify prior STOP.',
            'Visually inspect actual heatmap secondary peaks on a few baseline failures and any newly recovered candidate. If secondary peaks do not contain plausible correct locations, stop this geometry-only route and summarize a targeted teacher-data/identity improvement rather than repeating filters.',
            'BA01 is a review flag, not an authorized replacement label. Separate any label-consistency audit from corrected-prediction evaluation and never promote oracle minima as performance.'
        ],no_original_edits=True,pseudo_gt_released=False,
        holdout_images_opened=False,holdout_training_or_evaluation=False,
        incidental_metadata_access='A read-only broad rg search for calibration provenance matched units_evidence lines in locked-holdout calibration JSONs. No holdout images, labels, predictions, metrics, or fitted parameters were used. Restrict subsequent provenance searches explicitly to allowed sessions.',
        limitations='Reused development videos/single seed; optimistic pointwise oracle allows duplicate candidate use; original audit includes newly flagged BA01 and is kept frozen rather than silently improving metrics by post-hoc deletion.')
    path=OUT/'heartbeat_20261002_0835_decision.json';path.write_text(json.dumps(decision,indent=2))
    export=Path('/mnt/c/Users/55395/Downloads/MouseGeoForge-BottomAnchor-20261002/anchor_candidate_audit_v1')
    shutil.copy2(src/'BA01_pending_review.json',export/'BA01_pending_review.json')
    page=export/'index.html';html=page.read_text();html=html.replace('</html>','<h2>新增待复核 BA01</h2><p>FR2 第1039帧 camB 的16/18存在跨视角身份疑点；未修改标签。该整帧组及相交上下文从后续实验排除。此前统计保持原样，不以事后剔除冒充算法改善。</p><a href="BA01_pending_review.json">复核证据</a></html>');page.write_text(html)
    write_state(stage='anchor_candidate_audit_visually_reviewed',pid=None,active_run=None,mode='idle_between_heartbeats',job_deadline_utc=None,
        latest_result=str(src/'summary.json'),latest_decision=str(path),next_action=decision['next_action'],pseudo_gt_released=False)
    print(json.dumps(dict(decision=str(path),new_review_id='BA01',exclusion_count=len(exclusions(plan)),next='frozen temporal heatmap candidate recall audit'),indent=2))


if __name__=='__main__':main()
