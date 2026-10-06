"""All64 dual views plus aggregate conditional/detection metrics."""
import json
import numpy as np
import review_composite_allviews_20261005 as review
from run_depth_pose_ablation import fingerprint
review.OUT=review.ROOT/'output/singleview_fusion_cycle_20261005/dual_allscene_association_audit_v1'
review.DEST=review.DEST.parent/review.OUT.name
if __name__=='__main__':
    review.main(dual_only=True)
    r=json.loads((review.OUT/'results.json').read_text());points=[]
    for i in range(len(r['groups'])):points+=json.loads((review.OUT/f'group{i:02d}_baseline_points.json').read_text())
    det=[d for g in r['groups'] for d in g['detections']]
    tp=sum(p['iou'] is not None and p['iou']>=.5 for d in det for p in d['predictions'])
    errors=np.array([p['error'] for p in points])
    summary=dict(images=len(det),scenes=len(r['groups']),points=len(points),predictions=sum(d['count'] for d in det),gt_instances=2*len(det),true_positives=tp,
        mean_error=float(errors.mean()),p95_error=float(np.percentile(errors,95)),gated_pck32=float(np.mean([p['hit'] and p['error']<=32 for p in points])),
        fusion_eligible=sum(d['fusion_eligible'] for d in det),max_projection_px=max(g['max_projection_px'] for g in r['groups']),
        max_old_rgb_delta=max(max(p['delta'].values()) for g in r['groups'] for p in g['old_view_parity']),
        all_modes_unchanged=all(g['metrics']['baseline']==g['metrics']['composite'] for g in r['groups']),
        driver_sha256=fingerprint(review.ROOT/'tools/audit_dual_allscene_association_20261005.py'),
        conclusion='All64 images predict2 queries, but count correctness is not localization correctness. Fusion abstains throughout; natural single-query identity failure was NOT tested. No multi-mouse fusion success claim.')
    for folder in [review.OUT,review.DEST]:(folder/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
