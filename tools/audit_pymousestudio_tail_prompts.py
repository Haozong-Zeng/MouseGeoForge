"""Audit source31 tail points in rectified pixels before using any as prompts."""
from pathlib import Path
import sys,json
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from reconstruct_pymousestudio_pilot import triangulate
from export_pymousestudio_rectified import write


def main():
    s=ROOT/'dataset/PyMouseStudio/sessions/AZL3';out=s/'tail_dlc_audit_v1';out.mkdir(exist_ok=False)
    cal=json.loads((s/'calibration.json').read_text());Ks={n:np.array(c['K']) for n,c in cal.items()};Es={n:np.array(c['world_to_camera_mm']) for n,c in cal.items()}
    data={n:dict(np.load(s/f'keypoints_source31/cam{n}.npz',allow_pickle=False)) for n in 'BCDEFG'}
    records=[]
    for f in range(780,812):
        for j in range(24,31):
            obs={n:a['uv'][f,j] for n,a in data.items() if a['likelihood'][f,j]>=.8 and np.isfinite(a['uv'][f,j]).all()}
            X,e=triangulate(obs,Ks,Es)
            records.append(dict(frame=f,id=j,name=str(data['B']['source_names'][j]),confident_views=list(obs),
                                xyz_mm=X.tolist() if X is not None else None,median_inlier_reprojection_px=e))
    write(out/'report.json',dict(threshold=.8,minimum_geometry_views=3,records=records,
       reconstructed_frames_per_part={str(data['B']['source_names'][j]):sum(r['id']==j and r['xyz_mm'] is not None for r in records) for j in range(24,31)}))
    for f in [780,796,811]:
        fig,axs=plt.subplots(1,6,figsize=(21,6),layout='constrained')
        for ax,n in zip(axs,'BCDEFG'):
            img=cv2.cvtColor(cv2.imread(str(s/f'pilot_rgb/{f:06d}/cam{n}.png')),cv2.COLOR_BGR2RGB)
            mask=cv2.imread(str(s/f'pilot_input_masks/{f:06d}/cam{n}.png'),0)
            ax.imshow(img);ax.contour(mask>0,levels=[.5],colors=['cyan'],linewidths=.5)
            for j in range(24,31):
                x,y=data[n]['uv'][f,j];c=float(data[n]['likelihood'][f,j]);color='lime' if c>=.8 else 'orange'
                ax.scatter(x,y,c=color,s=18);ax.annotate(f'{j-24}:{c:.2f}',(x,y),xytext=(3,5),textcoords='offset points',fontsize=7,color=color)
            ax.set_title('cam'+n);ax.axis('off')
        fig.suptitle(f'Frame {f} | DLC tail root=0, tip=6 | green confidence>=0.8, orange below | cyan supplied mask; NOT GT')
        fig.savefig(out/f'tail_points_{f:06d}.png',dpi=110);plt.close(fig)
    print((out/'report.json').read_text()[-1200:])


if __name__=='__main__':main()
