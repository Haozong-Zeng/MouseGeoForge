"""Plot fixed synthetic-development checkpoints; not best-model selection."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'output/spatial_anchor_pilot_v1'
NEW=ROOT/'output/spatial_anchor_resume800_v1'


def main():
    assert json.loads((NEW/'verification.json').read_text())['passed']
    baseline=json.loads((OLD/'results.json').read_text())['baseline']
    groups=['bottom_only','old_cylinder_side_pair','old_dual_pair','sides_bottom']
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    points={}
    for arm,color in [('mask_prior_off','#bb4a35'),('mask_prior_on','#2777aa')]:
        history=json.loads((NEW/f'{arm}_history.json').read_text())
        loss=np.array([r['loss'] for r in history]);smooth=np.convolve(loss,np.ones(16)/16,'valid')
        axes[0,0].plot(np.arange(16,len(loss)+1),smooth,color=color,label=arm)
        points[arm]={}
        for step in (100,200,400,800):
            folder=OLD if step<=200 else NEW
            records=json.loads((folder/f'{arm}_eval_{step}.json').read_text())
            points[arm][step]={g:{k:float(np.mean([r['metrics'][k] for r in records if r['group']==g])) for k in ['body_keypoint_nme','tail_nme','detection_aware_pck_0.1']} for g in groups}
        for ax,g in zip(axes.flat[1:5],groups):
            ax.plot([100,200,400,800],[points[arm][s][g]['body_keypoint_nme'] for s in (100,200,400,800)],'o-',color=color,label=arm)
        axes[1,2].plot([100,200,400,800],[points[arm][s]['old_cylinder_side_pair']['tail_nme'] for s in (100,200,400,800)],'o-',color=color,label=arm)
    axes[0,0].set_title('Train loss: rolling 16 updates')
    axes[0,0].set_ylabel('SmoothL1 refined + 0.5 coarse')
    for ax,g in zip(axes.flat[1:5],groups):
        value=np.mean([r['metrics']['body_keypoint_nme'] for r in baseline if r['group']==g])
        ax.axhline(value,color='#666666',ls='--',label='frozen baseline')
        ax.set_title(g+' / body NME');ax.set_ylabel('Lower is better')
    axes[1,2].axhline(np.mean([r['metrics']['tail_nme'] for r in baseline if r['group']=='old_cylinder_side_pair']),color='#666666',ls='--')
    axes[1,2].set_title('old_cylinder_side_pair / tail NME')
    for ax in axes.flat:ax.grid(alpha=.2);ax.set_xlabel('Cumulative optimizer steps')
    axes[0,0].legend(fontsize=8);axes[0,1].legend(fontsize=8)
    fig.suptitle('Same 16 synthetic train records; fixed 16-record source-group-disjoint development\n100/200 original run; 400/800 exact checkpoint+AdamW resume. No real GT.',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.92));fig.savefig(NEW/'convergence.png',dpi=150);plt.close(fig)
    (NEW/'convergence_points.json').write_text(json.dumps(points,indent=2)+'\n')
    print(json.dumps(points))


if __name__=='__main__':main()
