"""Show fixed source replacement poses without selecting by model accuracy."""
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train_cylinder_source_mix import ROOT,OUTPUT,PREFLIGHT
from run_depth_pose_ablation import load_batch


def main():
    pre=json.loads((PREFLIGHT/'protocol.json').read_text())
    report=json.loads((PREFLIGHT/'report.json').read_text())
    fig,axes=plt.subplots(2,4,figsize=(13,7),layout='constrained')
    for ax,row in zip(axes.flat,report['diagnostics']):
        source=next(r for r in pre['training_sequence'] if r['scene']==row['scene'])
        rec=dict(source,views=[0],diagnostic_view_order=[0],physical_roles=['bottom'])
        batch=load_batch(rec,num_views=1)
        xy=batch['instance_keypoint_xy'][0,0,0].numpy()*768
        valid=batch['instance_keypoint_valid'][0,0,0].numpy().copy();valid[7:14]=False
        ax.imshow(Image.open(ROOT/rec['scene']/'blended_images/00000000.jpg').resize((768,768)))
        ax.scatter(xy[valid,0],xy[valid,1],s=8,c='lime')
        lo=np.maximum(xy[valid].min(0)-65,0);hi=np.minimum(xy[valid].max(0)+65,768)
        ax.set_xlim(lo[0],hi[0]);ax.set_ylim(hi[1],lo[1]);ax.axis('off')
        ax.set_title(f'Sample10 frame {row["new_frame"]} | bottom id=0')
    fig.suptitle('Eight preselected TRAIN poses | body16 GT green | cropped for inspection')
    fig.savefig(OUTPUT/'source_poses.png',dpi=140);plt.close(fig)


if __name__=='__main__':main()
