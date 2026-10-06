"""Fixed-pose visual check and training-exposure stratification of pose gap."""
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_cylinder_pose_gap import ROOT,OUT,PARENT,AUX
from run_depth_pose_ablation import torch,write_json
from mousegeoforge.localization_audit import summarize


def main():
    spec=json.loads((OUT/'protocol.json').read_text());rows=json.loads((OUT/'points.json').read_text())
    cohort=spec['cohort'];scenes=sorted({e['record']['scene'] for e in cohort if e['group']=='side_bottom'})
    scene=scenes[len(scenes)//2]
    idx=next(i for i,e in enumerate(cohort) if e['group']=='side_bottom' and e['record']['scene']==scene)
    parent=json.loads((PARENT/'protocol.json').read_text())
    dev_idx=next(i for i,e in enumerate(parent['evaluation']) if e['group']=='side_bottom' and parent['records'][e['key']]['scene'].endswith('000055'))
    fig,axes=plt.subplots(2,3,figsize=(12,8),layout='constrained')
    for col,(tag,devpath) in enumerate((('original',PARENT/'initial'),('adapted',PARENT/'combination_coverage'),('auxiliary',AUX/'coarse_body_aux'))):
        for row,path in enumerate((OUT/tag/f'prediction_{idx:03d}.pt',devpath/f'prediction_{dev_idx:03d}.pt')):
            saved=torch.load(path,weights_only=True,map_location='cpu');r=saved['record'];p,t=saved['predictions'],saved['targets']
            v=r['physical_roles'].index('bottom');qs,ms=saved['oracle_assignment'][v];q=qs[ms.index(0)]
            h,w=r['image_hw'];scale=np.array([w,h]);xy=p['instance_keypoint_xy'][0,v,q].numpy()*scale;gt=t['keypoint_xy'][0,v,0].numpy()*scale
            valid=t['keypoint_valid'][0,v,0].numpy().copy();valid[7:14]=False
            ax=axes[row,col];ax.imshow(Image.open(ROOT/r['scene']/'blended_images'/f'{r["views"][v]:08d}.jpg').resize((w,h)))
            ax.scatter(gt[valid,0],gt[valid,1],c='lime',s=15);ax.scatter(xy[valid,0],xy[valid,1],c='magenta',s=18,marker='x')
            for a,b in zip(gt[valid],xy[valid]):ax.plot([a[0],b[0]],[a[1],b[1]],c='magenta',lw=.5,alpha=.5)
            lo=np.maximum(gt[valid].min(0)-90,0);hi=np.minimum(gt[valid].max(0)+90,scale)
            ax.set_xlim(lo[0],hi[0]);ax.set_ylim(hi[1],lo[1]);ax.axis('off')
            diag=np.linalg.norm(t['boxes_cxcywh'][0,v,0,2:].numpy()*scale)
            nme=np.linalg.norm(xy[valid]-gt[valid],axis=-1).mean()/max(diag,1)
            ax.set_title(f'{tag} / '+('train pose' if row==0 else 'development')+f'\nbody NME={nme:.3f}')
    fig.suptitle(f'Same [2,6] input, different actions | train {scene.rsplit("/",1)[-1]}, dev frame55\nGT green / prediction magenta; fixed selection, off-crop points counted')
    fig.savefig(OUT/'fixed_train_dev_poses.png',dpi=150);plt.close(fig)
    exposure=[]
    for tag in ('original','adapted','auxiliary'):
        for group in ('bottom_only','side_bottom','sides_bottom','sides_bottom_top'):
            for subset in ('exact_seen','subset_seen','subset_unseen'):
                selected=[r for r in rows if r['checkpoint']==tag and r['split']=='train_pose' and r['group']==group and r['role']=='bottom'
                    and (r['exact_order_visits']>0 if subset=='exact_seen' else r['unordered_subset_visits']>0 if subset=='subset_seen' else r['unordered_subset_visits']==0)]
                exposure.append(dict(checkpoint=tag,group=group,subset=subset,scenes=len({r['scene'] for r in selected}),**summarize(selected)))
    write_json(OUT/'exposure_summary.json',exposure)


if __name__=='__main__':main()
