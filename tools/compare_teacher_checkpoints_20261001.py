"""Paired DLC checkpoint audit. Swap tests are diagnostics, never label corrections."""
import datetime as dt
import json
import shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/teacher_temporal_cycle_20261001'

def main():
    m=json.loads((OUT/'dlc_dataset_manifest.json').read_text())
    report={}
    for name in ['dlc_baseline_review','dlc_extension_review']:
        with np.load(OUT/name/'predictions_source_pixels.npz') as z:p=dict(z)
        summaries=json.loads((OUT/name/'summary.json').read_text())
        pairs={}
        for a,b in [(15,16),(17,18)]:
            ia,ib=[m['ids_1based'].index(k) for k in [a,b]]
            rows=[]
            for i,e in enumerate(m['entries']):
                gt=p['gt'][i,[ia,ib]];pred=p['uv'][i,[ia,ib]]
                if not np.isfinite(gt).all() or not np.isfinite(pred).all():continue
                direct=float(np.linalg.norm(pred-gt,axis=-1).mean())
                swapped=float(np.linalg.norm(pred-gt[::-1],axis=-1).mean())
                rows.append(dict(session=e['session'],frame=e['frame'],camera=e['camera'],role=e['role'],
                    direct_px=direct,swapped_px=swapped,
                    strong_swap_diagnostic=bool(direct>50 and swapped<.5*direct)))
            pairs[f'{a}_{b}']={}
            for role in ['train','validation']:
                sub=[r for r in rows if r['role']==role]
                pairs[f'{a}_{b}'][role]=dict(eligible_images=len(sub),strong_swap_count=sum(r['strong_swap_diagnostic'] for r in sub),
                    by_camera={c:dict(n=sum(r['camera']==c for r in sub),swaps=sum(r['camera']==c and r['strong_swap_diagnostic'] for r in sub)) for c in sorted({r['camera'] for r in sub})})
            (OUT/name/f'swap_diagnostic_{a}_{b}.json').write_text(json.dumps(rows,indent=2))
        report[name]=dict(validation=summaries['validation']['all'],pairs=pairs)
    report['warning']='Only a relative-to-accepted-manual semantic diagnostic; does not prove anatomical correctness. No labels/predictions swapped.'
    report['utc']=dt.datetime.now(dt.timezone.utc).isoformat()
    report['decision']='DLC improves but severe paw outliers persist. No extra DLC epochs now; actual Lightning Pose fixed-cohort supervised comparison launched.'
    report['next']='Compare LP original-pixel EPE/P90 and swaps; inspect failures before running ensemble or temporal expansion.'
    (OUT/'heartbeat_20261001_2234_decision.json').write_text(json.dumps(report,indent=2))
    dest=Path('/mnt/c/Users/55395/Downloads/MouseGeoForge-TeacherTemporal-20261001')
    shutil.copytree(OUT/'dlc_extension_review',dest/'dlc_extension_review',dirs_exist_ok=True)
    shutil.copy2(OUT/'heartbeat_20261001_2234_decision.json',dest/'dlc_checkpoint_comparison.json')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
