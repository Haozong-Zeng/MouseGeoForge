"""Named-landmark scientific overlay from saved predictions; no inference."""
import json
import numpy as np
import torch
from run_depth_pose_ablation import ROOT,write_json,fingerprint


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    source=ROOT/'output/origin_coordinate_residual_audit_v1/real_readouts.pt'
    data=torch.load(source,map_location='cpu',weights_only=True)
    out=ROOT/'output/real_landmark_visual_review_v1';out.mkdir(exist_ok=False)
    names=['AZL3_000780_bottom','CR2_000780_bottom','FR2_000780_bottom','FR2_001559_bottom','FR2_000780_side']
    marks=[(0,'Nose','#ffdf00'),(14,'LF','#00ccff'),(15,'RF','#ff55cc'),(16,'LH','#00ee66'),(17,'RH','#ff8800'),(22,'Abd','#aa88ff')]
    hashes={str(source.relative_to(ROOT)):fingerprint(source)}
    for name in names:
        camera='A' if name.endswith('bottom') else 'D'
        path=ROOT/f'output/real_cylinder_bottom_frozen_v1/review/{name.rsplit("_",1)[0]}_{camera}_input.png'
        hashes[str(path.relative_to(ROOT))]=fingerprint(path);rgb=np.array(Image.open(path));h,w=rgb.shape[:2]
        fig,axes=plt.subplots(1,4,figsize=(16,5.5),layout='constrained')
        for ax,(arm,pred) in zip(axes,data[name].items()):
            score=pred['instance_objectness_logits'][0,0].sigmoid();q=int(score.argmax());xy=pred['instance_keypoint_xy'][0,0,q].numpy()*[w,h]
            ax.imshow(rgb);ax.plot(xy[7:14,0],xy[7:14,1],'-o',color='#dddddd',linewidth=1,markersize=2,label='Predicted tail7-13')
            for k,label,color in marks:
                ax.scatter(*xy[k],s=30,c=color,edgecolors='black',linewidths=.4)
                ax.annotate(label,xy[k],xytext=(5,5),textcoords='offset points',color=color,fontsize=8,
                    bbox=dict(facecolor='black',alpha=.65,pad=.5,edgecolor='none'))
            ax.set_xlim(0,w);ax.set_ylim(h,0);ax.axis('off');ax.set_title(arm+f'\nq{q}, positive slots={int((score>=.5).sum())}',fontsize=9)
        fig.suptitle(name+' | NO MANUAL GT | L/R are animal anatomy, not screen side | Highest-score slot only',fontsize=11)
        fig.savefig(out/f'{name}.png',dpi=150);plt.close(fig)
    write_json(out/'manifest.json',dict(selected=names,training_steps=0,encoder_calls=0,source_hashes=hashes,
        marks=marks,warning='Visual diagnostic only. Highest-score slot, not complete instance recall evaluation; tail may include invisible points.'))
    print('COMPLETED')


if __name__=='__main__':main()
