"""Scientific review chart generated from measured development results."""
import json,shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/teacher_temporal_cycle_20261001'
r=json.loads((OUT/'lp_ensemble_17_31_73/summary.json').read_text())
fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
labels=['DLC best70','LP seed17','LP seed31','LP seed73','3-seed mean','3-seed median']
values=[json.loads((OUT/'dlc_extension_review/summary.json').read_text())['validation']['all']]+list(r['single_models'].values())+[r['mean_ensemble'],r['median_ensemble']]
x=list(range(len(labels)));axs[0].bar(x,[a['p90'] for a in values],color='#5078a0',label='P90')
axs[0].scatter(x,[a['median'] for a in values],color='#e39a32',label='Median',zorder=3)
axs[0].set_xticks(x,labels,rotation=25,ha='right');axs[0].set_ylabel('Original-image error (pixels)');axs[0].legend();axs[0].set_title('Same 1,382 labeled development points')
fractions=[.5,.75,.9]
for key,label,color in [('low_spread','Lowest teacher disagreement','#268b69'),('high_confidence','Highest average confidence','#b56878')]:
    axs[1].plot([f*100 for f in fractions],[r['matched_coverage'][str(f)][key]['p90'] for f in fractions],'o-',label=label,color=color)
axs[1].set_xlabel('Retained labeled points (%)');axs[1].set_ylabel('P90 error of 3-seed mean (pixels)');axs[1].legend();axs[1].set_title('Quality vs coverage; development diagnostic')
fig.suptitle('Agreement helps reject errors, but shared mistakes remain',fontsize=14)
dest=OUT/'lp_ensemble_17_31_73/review';dest.mkdir(exist_ok=True);fig.savefig(dest/'ensemble_quality_coverage.png',dpi=150)
export=Path('/mnt/c/Users/55395/Downloads/MouseGeoForge-TeacherTemporal-20261001/lp_ensemble_17_31_73_review')
export.mkdir(exist_ok=True)
for p in [dest/'ensemble_quality_coverage.png',OUT/'lp_ensemble_17_31_73/summary.json',OUT/'lp_ensemble_17_31_73/shared_failure_cases.json']:shutil.copy2(p,export/p.name)
print(export)
