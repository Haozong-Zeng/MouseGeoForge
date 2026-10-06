"""Fixed-threshold LP/DLC agreement audit; diversity does not certify labels."""
import datetime as dt,json
import numpy as np
from run_teacher_cycle_dlc_20261001 import ROOT,OUT
from audit_teacher_dlc_predictions_20261001 import stats

def main():
    dest=OUT/'cross_framework_agreement_v1';dest.mkdir(exist_ok=False)
    (dest/'settings.json').write_text(json.dumps(dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        lp='three-seed KL median',dlc='existing best70',agreement_max_px=15,lp_spread_max_px=10,
        matched_coverage=[.5,.75],candidate_scoring_unchanged=True,train_allowed=False),indent=2))
    m=json.loads((OUT/'dlc_dataset_manifest.json').read_text());plan=json.loads((OUT/'plan.json').read_text())
    with np.load(OUT/'lp_ensemble_17_31_73/predictions_source_pixels.npz') as z:lp=dict(z)
    with np.load(OUT/'dlc_extension_review/predictions_source_pixels.npz') as z:dlc=dict(z)
    assert np.allclose(lp['gt'],dlc['gt'],equal_nan=True)
    excluded={(r['session'],r['frame']) for r in json.loads((ROOT/plan['exclusion_manifest']).read_text())['cases']};excluded.add(('AZR2L2',584))
    rows=[]
    for i in m['validation_indices']:
        e=m['entries'][i];assert e['session'] in plan['development_validation_sessions']
        if (e['session'],e['frame']) in excluded:continue
        for j,k in enumerate(m['ids_1based']):
            if not np.isfinite(lp['gt'][i,j]).all():continue
            rows.append(dict(session=e['session'],frame=e['frame'],camera=e['camera'],id=k,
                lp_error=float(np.linalg.norm(lp['uv_median'][i,j]-lp['gt'][i,j])),
                dlc_error=float(np.linalg.norm(dlc['uv'][i,j]-lp['gt'][i,j])),
                disagreement=float(np.linalg.norm(lp['uv_median'][i,j]-dlc['uv'][i,j])),
                lp_spread=float(lp['spread_px'][i,j])))
    def metrics(rs,total):return dict(coverage=len(rs)/total,lp_errors=stats([r['lp_error'] for r in rs]),
        dlc_errors=stats([r['dlc_error'] for r in rs]),lp_over50=sum(r['lp_error']>50 for r in rs),lp_over100=sum(r['lp_error']>100 for r in rs))
    report={}
    for name,rs in [('all_body',rows),('paws',[r for r in rows if r['id']>=15])]:
        n=len(rs);cross=[r for r in rs if r['disagreement']<=15];low=[r for r in rs if r['lp_spread']<=10];both=[r for r in low if r['disagreement']<=15]
        report[name]=dict(n=n,all=metrics(rs,n),cross_framework=metrics(cross,n),low_lp_spread=metrics(low,n),both=metrics(both,n),matched_coverage={})
        for fraction in [.5,.75]:
            count=int(n*fraction)
            report[name]['matched_coverage'][str(fraction)]={key:metrics(sorted(rs,key=lambda r:r[key])[:count],n) for key in ['disagreement','lp_spread']}
    report.update(train_allowed=False,warning='Same labeledtrainingcohort and ResNet50backbone across frameworks; agreement is not independence/calibratedconfidence. Only2DLP accuracy diagnosed; no3D improvement claimed.')
    (dest/'summary.json').write_text(json.dumps(report,indent=2));(dest/'observations.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps(report['paws'],indent=2))

if __name__=='__main__':main()
