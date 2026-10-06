"""Manual-label diagnostic only: nearest-other-paw confusion, never relabels."""
import json,datetime as dt
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from audit_teacher_dlc_predictions_20261001 import stats
from run_teacher_cycle_dlc_20261001 import ROOT,OUT

def main():
    dest=OUT/'paw_identity_audit_v1';dest.mkdir(exist_ok=False)
    settings=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),nearest_other_max_px=15,
        improvement_margin_px=30,prediction_collision_px=15,high_confidence_min=.6,low_spread_max_px=10,
        no_relabeling=True,train_allowed=False,
        proposed_followup='Frozen15px predicted-paw collision rejection, both identities unknown in that view. No GT used at inference. No threshold sweep.')
    (dest/'settings.json').write_text(json.dumps(settings,indent=2))
    m=json.loads((OUT/'dlc_dataset_manifest.json').read_text());plan=json.loads((OUT/'plan.json').read_text())
    ids=m['ids_1based'];paws=[15,16,17,18];js=[ids.index(k) for k in paws]
    with np.load(OUT/'lp_ensemble_17_31_73/predictions_source_pixels.npz') as z:
        uv=z['uv_median'][:,js];gt=z['gt'][:,js];conf=z['confidence'][:,js];spread=z['spread_px'][:,js]
    excluded={(r['session'],r['frame']) for r in json.loads((ROOT/plan['exclusion_manifest']).read_text())['cases']}
    rows=[];oracle=[]
    for i in m['validation_indices']:
        e=m['entries'][i];assert e['session'] in plan['development_validation_sessions'] and (e['session'],e['frame']) not in excluded
        valid=np.isfinite(gt[i]).all(-1);dist=np.linalg.norm(uv[i,:,None]-gt[i,None],axis=-1)
        pred_dist=np.linalg.norm(uv[i,:,None]-uv[i,None],axis=-1);np.fill_diagonal(pred_dist,np.inf)
        if valid.all():
            a,b=linear_sum_assignment(dist)
            oracle.append(dict(session=e['session'],frame=e['frame'],camera=e['camera'],
                labeled_identity_mean_px=float(np.diag(dist).mean()),oracle_permuted_mean_px=float(dist[a,b].mean()),
                permutation=[paws[j] for j in b],warning='Uses GT for diagnostic only; NOT an inference algorithm or correction.'))
        for j,k in enumerate(paws):
            if not valid[j]:continue
            others=[t for t in range(4) if t!=j and valid[t]]
            closest=min(others,key=lambda t:dist[j,t]) if others else None
            flagged=closest is not None and dist[j,closest]<=15 and dist[j,j]-dist[j,closest]>=30
            collision=bool((pred_dist[j]<=15).any())
            kind=None
            if flagged:
                kind='left_right' if (k in [15,16])==(paws[closest] in [15,16]) else ('fore_hind_same_side' if k%2==paws[closest]%2 else 'fore_hind_opposite_side')
            rows.append(dict(session=e['session'],frame=e['frame'],camera=e['camera'],id=k,
                target_error_px=float(dist[j,j]),nearest_other_id=paws[closest] if closest is not None else None,
                nearest_other_error_px=float(dist[j,closest]) if closest is not None else None,
                suspected_identity_confusion=bool(flagged),confusion_type=kind,
                annotated_paws=int(valid.sum()),confidence=float(conf[i,j]),spread_px=float(spread[i,j]),
                predicted_collision=collision,collides_with=[paws[t] for t in range(4) if pred_dist[j,t]<=15]))
    def summarize(rs):
        flagged=[r for r in rs if r['suspected_identity_confusion']];collided=[r for r in rs if r['predicted_collision']]
        return dict(n=len(rs),error=stats([r['target_error_px'] for r in rs]),
            suspected_confusions=len(flagged),confusion_types={k:sum(r['confusion_type']==k for r in flagged) for k in ['left_right','fore_hind_same_side','fore_hind_opposite_side']},
            high_conf_low_spread_confusions=sum(r['confidence']>=.6 and r['spread_px']<=10 for r in flagged),
            prediction_collisions=len(collided),confusions_with_collision=sum(r['predicted_collision'] for r in flagged),
            collisions_on_error_under15=sum(r['target_error_px']<15 for r in collided))
    matrix=[[sum(r['id']==a and r['nearest_other_id']==b and r['suspected_identity_confusion'] for r in rows) for b in paws] for a in paws]
    report=dict(all=summarize(rows),by_camera={c:summarize([r for r in rows if r['camera']==c]) for c in sorted({r['camera'] for r in rows})},
        by_keypoint={str(k):summarize([r for r in rows if r['id']==k]) for k in paws},
        confusion_counts=matrix,matrix_ids=paws,complete_four_paw_images=len(oracle),
        oracle_assignment=dict(labeled=stats([r['labeled_identity_mean_px'] for r in oracle]),permuted=stats([r['oracle_permuted_mean_px'] for r in oracle])),
        train_allowed=False,warning='Nearest annotated OTHER paw suggests confusion, not proof. Missing paws are unknown; confusion counts underestimate unannotated alternatives. Oracle assignment uses manual GT and is never used for pseudo-labels or reported as deployable accuracy.')
    (dest/'summary.json').write_text(json.dumps(report,indent=2));(dest/'observations.json').write_text(json.dumps(rows,indent=2));(dest/'oracle_diagnostic.json').write_text(json.dumps(oracle,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
