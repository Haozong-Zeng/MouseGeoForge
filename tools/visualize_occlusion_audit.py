"""Static scientific figures for the frozen-checkpoint occlusion audit."""
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vggt-omega'))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image


def main():
    folder=ROOT/'output/occlusion_pose_v1'
    report=json.loads((folder/'geometry_report.json').read_text())
    calibration=json.loads((folder/'uncertainty_calibration.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
    rows=[r for r in report['metrics'] if r['dataset'].startswith('Homecage-BlackBrown')
          and r['visibility']=='all' and r['method'] in ('linear_oracle_association','robust_oracle_association',
             'pred_visibility_oracle_association','robust_predicted_association')]
    labels=['Linear\nGT-assisted ID','Robust\nGT-assisted ID','Visibility gate\nGT-assisted ID','Robust\nPredicted ID']
    for ax,key,title in zip(axes[:2],('coverage','mean_error_native'),('Supported 3D points (%)','Error on supported points (mm)')):
        vals=[r[key]*(100 if key=='coverage' else 1) for r in rows]
        ax.bar(range(len(rows)),vals,color=['#74899b','#227da8','#de924c','#bb5366'])
        ax.set_xticks(range(len(rows)),labels,fontsize=8)
        for i,v in enumerate(vals): ax.text(i,v,f'{v:.1f}',ha='center',va='bottom',fontsize=10)
        ax.set_ylim(0,max(vals)*1.2); ax.set_title(title); ax.spines[['top','right']].set_visible(False)
    vals=[calibration[s][m]['coverage_95']*100 for s in ('train','validation') for m in ('initial','fitted')]
    axes[2].bar(range(4),vals,color=['#74899b','#227da8','#74899b','#bb5366'])
    axes[2].axhline(95,color='black',linestyle='--',label='Nominal 95%')
    axes[2].set_xticks(range(4),['Train\nInitial','Train\nFitted','Validation\nInitial','Validation\nFitted'],fontsize=8)
    axes[2].set_ylim(0,110); axes[2].set_title('Uncertainty coverage: calibration FAIL')
    for i,v in enumerate(vals): axes[2].text(i,v,f'{v:.1f}%',ha='center',va='bottom')
    axes[2].legend(fontsize=8); axes[2].spines[['top','right']].set_visible(False)
    fig.suptitle('768 frozen model | Dual-mouse held-out views | Rejection changes coverage, not just error',fontsize=13)
    fig.savefig(folder/'occlusion_summary.png',dpi=160); plt.close(fig)
    cached=[torch.load(p,weights_only=True) for p in sorted((folder/'predictions').glob('*.pt'))]
    worst=sorted([x for x in cached if x['record']['split']=='val' and x['record']['expected_instances']==2],
                 key=lambda x:x['baseline_2d']['box_matched_nme'],reverse=True)[:2]
    fig,axes=plt.subplots(2,2,figsize=(16,9.8),layout='constrained')
    for row,item in enumerate(worst):
        r=item['record']; h,w=r['image_hw']; scale=np.array([w,h])
        views=[int(v) for v in item['actual_view_ids'][0].split(',')]
        for v,vid in enumerate(views):
            ax=axes[row,v]
            rgb=Image.open(ROOT/r['scene']/'blended_images'/f'{vid:08d}.jpg').convert('RGB').resize((w,h))
            ax.imshow(rgb)
            qs,ms=item['oracle_assignment'][v]
            for q,m in zip(qs,ms):
                gt=item['targets']['keypoint_xy'][0,v,m].numpy()*scale
                pred=item['predictions']['instance_keypoint_xy'][0,v,q].numpy()*scale
                valid=item['targets']['keypoint_valid'][0,v,m].numpy()
                vis=item['targets']['keypoint_visible'][0,v,m].numpy()
                for k in np.flatnonzero(valid):
                    ax.plot([gt[k,0],pred[k,0]],[gt[k,1],pred[k,1]],color='white',alpha=.35,linewidth=.6)
                ax.scatter(gt[valid & vis,0],gt[valid & vis,1],s=22,facecolors='none',edgecolors='#45ff6f',linewidths=1)
                ax.scatter(gt[valid & ~vis,0],gt[valid & ~vis,1],s=25,facecolors='none',edgecolors='#ffbf35',linewidths=1.3)
                ax.scatter(pred[valid,0],pred[valid,1],s=18,c='#ff4364',marker='x',linewidths=.9)
            ax.set_title(f'Frame {r["frame"]} / camera {vid:03d} | NME {item["baseline_2d"]["box_matched_nme"]:.3f}',fontsize=11)
            ax.set_xlim(0,w); ax.set_ylim(h,0); ax.axis('off')
    fig.suptitle('Green circles: visible GT | Amber circles: occluded GT | Red crosses: predictions\nGT box-assisted correspondence for diagnosis only; original model unchanged',fontsize=13)
    fig.savefig(folder/'hard_occlusion_views.png',dpi=140); plt.close(fig)


if __name__=='__main__': main()
