"""Summarize fixed failure probes, keeping diagnostic component boxes separate."""
import json
from collections import Counter
import numpy as np
from run_depth_pose_ablation import ROOT,write_json,fingerprint

OUT=ROOT/'output/mask_box_failures_v1_actual_train'


def main():
    assert all(json.loads((OUT/n).read_text())['passed'] for n in ('verification.json','selection_verification.json'))
    inputs={n:fingerprint(OUT/n) for n in ('query_statistics.json','comparisons.json','protocol.json','verification.json','selection_verification.json')}
    rows=json.loads((OUT/'query_statistics.json').read_text());comparisons=json.loads((OUT/'comparisons.json').read_text())
    summary=[];failures=[]
    for seed in ('seed17','seed29'):
        for subset in ('train','selected_dev_failure'):
            local=[r for r in rows if r['seed']==seed and r['subset']==subset and r['matched_instance'] is not None]
            failures.extend(r for r in local if r['score']>=.5 and r['mask_box_gt_iou']<.5)
            old=np.array([r['regressed_gt_iou'] for r in local]);new=np.array([r['mask_box_gt_iou'] for r in local]);large=np.array([r['diagnostic_largest_box_gt_iou'] for r in local]);score=np.array([r['score'] for r in local])
            lost=(score>=.5)&(old>=.5)&(new<.5)
            cs=[r for r in comparisons if r['seed']==seed and r['subset']==subset]
            report=dict(seed=seed,subset=subset,matched_instance_views=len(local),old_mean_iou=float(old.mean()),full_mask_mean_iou=float(new.mean()),diagnostic_largest_mean_iou=float(large.mean()),
                fixed_old_detected=int(((score>=.5)&(old>=.5)).sum()),fixed_mask_detected=int(((score>=.5)&(new>=.5)).sum()),
                lost=int(lost.sum()),lost_diagnostic_largest_recovers=int((lost&(large>=.5)).sum()),
                lost_largest_mass_fractions=[r['largest_component_fraction'] for r,l in zip(local,lost) if l],
                lost_main_full_box_area_ratios=[r['largest_to_full_box_area'] for r,l in zip(local,lost) if l],
                metrics={arm:{k:sum(c[arm][k] for c in cs) for k in ('detection_tp','detection_fp','detection_fn')} for arm in ('regressed','mask')})
            summary.append(report)
    compact=[]
    for r in failures:
        if r['seed']!='seed29':continue
        compact.append({k:r[k] for k in ('subset','tag','scene','view','role','query','matched_instance','score','regressed_gt_iou','mask_box_gt_iou','diagnostic_largest_box_gt_iou','components','largest_component_fraction','largest_to_full_box_area','touches_boundary','gt_mask_ious')})
    report=dict(summary=summary,seed29_failures=compact,source_sha256=inputs,analysis_code_sha256=fingerprint(ROOT/'tools/analyze_mask_box_failures.py'),
        limitations=['Selected development failures are not an unbiased evaluation set.',
            'Largest-component box numbers are diagnostic, not a deployed policy, and can delete genuine disconnected anatomy.',
            'Actual training scenes only; no new data or source-video generalization.',
            'Masks selected at fixed >0 threshold; GT overlap is diagnostic and never used to choose inference outputs.'],optimizer_steps=0,production_promoted=False)
    assert all(fingerprint(OUT/n)==h for n,h in inputs.items())
    write_json(OUT/'analysis.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
