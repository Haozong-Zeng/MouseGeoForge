"""Descriptive diagnostics only: percentiles are not independent-sample CIs."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def main():
    root=Path(__file__).resolve().parents[1];folder=root/'output/gate_objective_diagnostic_v1';out=folder/'diagnostic.png'
    if out.exists():raise FileExistsError(out)
    s=json.loads((folder/'summary.json').read_text());grad=json.loads((folder/'gradients.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(13,4.7),layout='constrained')
    groups=['mouse_climb_wall_91','moseq_sample10','crim13/021709_A25_Block22_BCfe1_t'];names=['ClimbWall','MoSeq','Two mice']
    colors=['#2277aa','#db6d27']
    for ai,arm in enumerate(['static','dynamic']):
        for i,g in enumerate(groups):
            values=[r['gradient_cosine'] for r in grad if r['arm']==arm and r['split']=='train' and r['group']==g and r['gradient_cosine'] is not None]
            x=i+(-.16 if ai==0 else .16)
            axes[0].scatter(np.full(len(values),x)+np.linspace(-.07,.07,len(values)),values,s=20,alpha=.55,color=colors[ai],label=arm if i==0 else None)
            axes[0].plot([x-.1,x+.1],[np.mean(values)]*2,color=colors[ai],lw=3)
    axes[0].axhline(0,color='black',ls='--',lw=1);axes[0].set(xticks=range(3),xticklabels=names,ylim=(-1.05,1.05),ylabel='Gradient cosine: refined vs coarse',title='Some conflicts, not globally opposed');axes[0].legend()
    for ri,role in enumerate(['bottom','side']):
        xs=[];means=[];lows=[];highs=[]
        for i,split in enumerate(['train','dev','real']):
            r=s['dynamic'][split][role];xs.append(i+(-.12 if ri==0 else .12));means.append(r['gate']);lows.append(r['gate']-r['gate_p10']);highs.append(r['gate_p90']-r['gate'])
        axes[1].errorbar(xs,means,yerr=[lows,highs],fmt='o-',capsize=4,label=role,color=colors[ri])
    axes[1].set(xticks=range(3),xticklabels=['Train','Development','Real development'],ylim=(0,1),ylabel='Weight on spatial evidence',title='Dynamic body gates: mean and 10-90th percentiles');axes[1].legend()
    fig.suptitle('Frozen gate diagnostic | no optimizer updates | real has no manual GT')
    fig.text(.5,-.02,'Different poses, view combinations and sizes across populations; these are descriptive distributions, not accuracy or confidence intervals.',ha='center',fontsize=9)
    fig.savefig(out,dpi=160,bbox_inches='tight');plt.close(fig)


if __name__=='__main__':main()
