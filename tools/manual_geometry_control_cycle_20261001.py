"""Manual-only geometric compatibility control on the SAME sparse pilot frames.
This diagnoses label/calibration compatibility; it is not a 3D accuracy floor.
"""
import json
from pathlib import Path
import numpy as np
from audit_pymousestudio_anipose_20261001 import group_from_cal,fit,safe_project
from audit_pymousestudio_manual_multiview import stats
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/teacher_temporal_cycle_20261001';BASE=ROOT/'dataset/PyMouseStudio/annotation_returns/20260930_full_v2'

def main():
    plan=json.loads((OUT/'plan.json').read_text());ids=np.array(plan['approved_body_ids_1based'])
    clips=json.loads((OUT/'continuous_lp_seed17_kl/manifest.json').read_text())
    manifest=json.loads((BASE/'ingestion_manifest.json').read_text());rows=[]
    lp=json.loads((OUT/'temporal_anipose_lp_seed17_kl/manual_observation_errors.json').read_text())
    key=lambda r:(r['session'],r['frame'],r['camera'],r['id'])
    lpmap={key(r):r for r in lp}
    for clip in clips:
        session=clip['session'];assert session in plan['development_validation_sessions']
        groups=[g for g in manifest['frame_groups'] if g['session']==session and clip['source_start']<=g['frame']<=clip['source_end_inclusive']]
        cams=clip['cameras'];cal=json.loads((BASE/'calibration'/session/'rectified_calibration.json').read_text());cg=group_from_cal(cal,cams)
        uv=np.full((len(cams),len(groups),7,2),np.nan)
        for fi,g in enumerate(groups):
            with np.load(BASE/g['normalized_file']) as z:gt=z['uv'][:,0,ids-1]
            for ci,c in enumerate(g['cameras']):
                if c in cams:uv[cams.index(c),fi]=gt[ci]
        flat=uv.reshape(len(cams),-1,2);_,xyz=fit(cg,flat);allp=safe_project(cg,xyz).reshape(uv.shape)
        loo=np.full_like(uv,np.nan)
        for vi,c in enumerate(cams):
            others=[i for i in range(len(cams)) if i!=vi]
            _,xx=fit(cg.subset_cameras(others),flat[others]);loo[vi]=safe_project(cg.subset_cameras([vi]),xx)[0].reshape(len(groups),7,2)
        for vi,c in enumerate(cams):
            for fi,g in enumerate(groups):
                for j,k in enumerate(ids):
                    if not np.isfinite(uv[vi,fi,j]).all():continue
                    record=dict(session=session,frame=g['frame'],camera=c,id=int(k))
                    def err(v):return float(np.linalg.norm(v-uv[vi,fi,j])) if np.isfinite(v).all() else None
                    record.update(manual_allview_error_px=err(allp[vi,fi,j]),manual_loo_error_px=err(loo[vi,fi,j]))
                    record.update({k:v for k,v in lpmap[key(record)].items() if 'error_px' in k})
                    rows.append(record)
    keys=['manual_allview_error_px','manual_loo_error_px','raw_teacher_error_px','scale0_loo_error_px','scale1_loo_error_px','scale4_loo_error_px']
    common=[r for r in rows if all(r[k] is not None for k in keys)]
    result=dict(observations=len(rows),paired_n=len(common),paired={k:stats([r[k] for r in common]) for k in keys},
        manual_loo_by_keypoint={str(k):stats([r['manual_loo_error_px'] for r in common if r['id']==k]) for k in ids},
        train_allowed=False,calibration_changed=False,manual_labels_changed=False,
        warning='Manual-only diagnostic uses other-view manual labels, not an independent teacher. Not a 3D truth metric or lower bound. Residuals mix annotation correspondence, calibration, synchronization and model projection assumptions; do not attribute cause from this alone.')
    dest=OUT/'manual_geometry_control';dest.mkdir(exist_ok=False)
    (dest/'summary.json').write_text(json.dumps(result,indent=2));(dest/'observations.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
