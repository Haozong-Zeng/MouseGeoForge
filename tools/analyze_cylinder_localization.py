"""Scientific figures and paired summaries of frozen local-head diagnostics."""
from collections import defaultdict
import json
import math
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from audit_cylinder_localization import ROOT,OUTPUT
from run_depth_pose_ablation import torch,write_json,fingerprint
from mousegeoforge.localization_audit import summarize


def main():
    assert json.loads((OUTPUT/'verification.json').read_text())['passed']
    rows=json.loads((OUTPUT/'points.json').read_text())
    summary=json.loads((OUTPUT/'summary.json').read_text())
    def get(tag,group,subset='body',role='bottom'):
        return next(r for r in summary if r['checkpoint']==tag and r['group']==group and r['role']==role and r['subset']==subset)
    groups=['bottom_only','side_bottom','sides_bottom','sides_bottom_top']
    names=['original','old_sampling','new_sampling'];colors=['#708090','#b27649','#218d89']
    fig,axes=plt.subplots(1,3,figsize=(15,4.7),layout='constrained')
    for i,(name,color) in enumerate(zip(names,colors)):
        values=[get(name,g) for g in groups]
        for ax,key in zip(axes,('inside_window','refinement_delta_px','entropy_normalized')):
            ax.bar(np.arange(4)+(i-1)*.25,[r[key] for r in values],.25,label=name,color=color)
    for ax,title in zip(axes,('GT inside local window (fraction)','Local correction magnitude (pixels)','Heatmap entropy / log(64)')):
        ax.set_title(title,fontsize=11);ax.set_xticks(range(4),groups,rotation=25,ha='right')
    axes[0].legend(fontsize=8);axes[0].set_ylim(0,1);axes[2].set_ylim(0,1.02)
    fig.suptitle('Frozen checkpoints | valid body points | synthetic Cylinder, network input 768 px')
    fig.savefig(OUTPUT/'localization_summary.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,part in zip(axes,('head','limbs')):
        for i,(tag,color) in enumerate(zip(names,colors)):
            values=[get(tag,g,part) for g in groups]
            ax.plot(range(4),[r['coarse_nme'] for r in values],ls='--',marker='o',color=color,label=tag+' coarse')
            ax.plot(range(4),[r['refined_nme'] for r in values],ls='-',marker='x',color=color,label=tag+' refined')
        ax.set_title(part+' NME: coarse vs refined');ax.set_xticks(range(4),groups,rotation=25,ha='right');ax.legend(fontsize=7)
    fig.suptitle('Box-assisted matching held fixed; lower is better')
    fig.savefig(OUTPUT/'coarse_vs_refined.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,group in zip(axes,('side_bottom','old_dual_pair')):
        for tag,color in zip(names,colors):
            selected=[r for r in rows if r['checkpoint']==tag and r['group']==group and r['part']!='tail'
                and (r['role']=='bottom' if group=='side_bottom' else True)]
            s=summarize(selected)
            ax.plot([1,4,8,16],[s[f'scale_{x}_nme'] for x in (1,4,8,16)],marker='o',label=tag,color=color)
        ax.set(title=group,xlabel='Posthoc cosine-logit multiplier (not trained)',ylabel='Body NME');ax.legend(fontsize=8)
    fig.suptitle('Counterfactual only: fixed coarse prediction, fixed assignment, no weight updates')
    fig.savefig(OUTPUT/'temperature_counterfactual.png',dpi=150);plt.close(fig)
    # Same hard frame selected before any candidate was trained, fixed anatomical IDs.
    selected=next(r for r in rows if r['checkpoint']=='new_sampling' and r['group']=='side_bottom' and r['scene'].endswith('000055'))
    saved=torch.load(OUTPUT/'new_sampling'/f'prediction_{selected["record"]:03d}.pt',weights_only=True,map_location='cpu')
    r=saved['record'];v=r['physical_roles'].index('bottom');q=saved['oracle_assignment'][v][0][0]
    p,t=saved['predictions'],saved['targets'];h,w=r['image_hw'];pixel=np.array([w,h])
    image=Image.open(ROOT/r['scene']/'blended_images'/f'{r["views"][v]:08d}.jpg').resize((w,h))
    fig,axes=plt.subplots(3,4,figsize=(13,10),layout='constrained')
    for row,k in enumerate((0,14,18)):
        coarse=p['instance_keypoint_coarse_xy'][0,v,q,k].numpy()*pixel
        final=p['instance_keypoint_xy'][0,v,q,k].numpy()*pixel
        gt=t['keypoint_xy'][0,v,0,k].numpy()*pixel
        logits=p['instance_keypoint_local_heatmap_logits'][0,v,q,k].numpy()
        radius=.1*pixel;lo=coarse-radius;hi=coarse+radius
        ax=axes[row,0];ax.imshow(image)
        ax.add_patch(Rectangle(lo,*(2*radius),fill=False,ec='orange',lw=1.5))
        ax.scatter(*gt,c='lime',s=35,label='GT');ax.scatter(*coarse,c='cyan',s=40,marker='+',label='coarse')
        ax.scatter(*final,c='magenta',s=35,marker='x',label='refined')
        bounds=np.stack((lo,hi,gt));xmin,ymin=bounds.min(0)-25;xmax,ymax=bounds.max(0)+25
        ax.set_xlim(max(xmin,0),min(xmax,w));ax.set_ylim(min(ymax,h),max(ymin,0));ax.axis('off')
        ax.set_title(saved['names'][k]);
        if row==0:ax.legend(fontsize=7)
        for col,scale in enumerate((1,8,16),1):
            z=logits*scale;prob=np.exp(z-z.max());prob/=prob.sum()
            ax=axes[row,col];im=ax.imshow(prob,extent=(lo[0],hi[0],hi[1],lo[1]),cmap='viridis',vmin=0,vmax=1)
            if (gt>=lo).all() and (gt<=hi).all():ax.scatter(*gt,c='lime',s=25,marker='o')
            ax.scatter(*coarse,c='cyan',s=30,marker='+')
            entropy=-(prob*np.log(prob.clip(1e-30))).sum()/math.log(64)
            ax.set_title(f'scale {scale}: max P={prob.max():.3f}\nH/log64={entropy:.3f}; GT '+('inside' if (gt>=lo).all() and (gt<=hi).all() else 'OUTSIDE'),fontsize=9)
            ax.set_xlabel('image x');ax.set_ylabel('image y')
    fig.colorbar(im,ax=axes[:,1:].ravel().tolist(),shrink=.6,label='Probability per grid sample (same scale)')
    fig.suptitle('Fixed frame 55, side+bottom, new sampling checkpoint\norange: local search window; temperature panels are posthoc only')
    fig.savefig(OUTPUT/'hard_frame_local_windows.png',dpi=150);plt.close(fig)
    # Aggregate all old dual views, but keep bottom-view groups separate.
    compact={}
    for tag in names:
        compact[tag]={g:{s:get(tag,g,s) for s in ('body','head','limbs','inside','outside') if any(
            r['checkpoint']==tag and r['group']==g and r['role']=='bottom' and r['subset']==s for r in summary)} for g in groups}
        compact[tag]['old_dual_all_views']=summarize([r for r in rows if r['checkpoint']==tag and r['group']=='old_dual_pair' and r['part']!='tail'])
    report=dict(summary=compact,cosine_logit_max_probability_bound=float(math.exp(2)/(63+math.exp(2))),
        uniform_probability=1/64,points_sha256=fingerprint(OUTPUT/'points.json'),
        interpretation='Cosine logits lie in [-1,1]; the probability upper bound is theoretical, not measured. Posthoc scaling is not a trained improvement.')
    write_json(OUTPUT/'report.json',report)
    for tag in names:
        for group in groups:
            s=get(tag,group)
            print(tag,group,json.dumps({k:round(s[k],5) for k in ('coarse_nme','refined_nme','inside_window','refinement_delta_px','entropy_normalized','scale_8_nme')}))


if __name__=='__main__':main()
