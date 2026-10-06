"""Static per-joint error-change visualization, synthetic development only."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/spatial_anchor_joint_audit_v1'
NAMES=['nose','head','L ear','R ear','neck','spine','lumbar','tail base','tail q1','tail q2','tail q3','tail q4','tail q5','tail tip','L forepaw','R forepaw','L hindpaw','R hindpaw','L elbow','R elbow','L knee','R knee','abdomen']


def main():
    s=json.loads((OUT/'joint_summary.json').read_text());groups=list(s['baseline'])
    error=np.array([[s['mask_prior_on'][g]['per_joint'][str(k)]['nme']-s['baseline'][g]['per_joint'][str(k)]['nme'] for k in range(23)] for g in groups])
    failure=np.array([[s['baseline'][g]['per_joint'][str(k)]['detection_pck']-s['mask_prior_on'][g]['per_joint'][str(k)]['detection_pck'] for k in range(23)] for g in groups])
    fig,axes=plt.subplots(2,1,figsize=(15,10),layout='constrained')
    for ax,data,label in zip(axes,[error,failure],['NME change: ON800 minus frozen baseline','Detection PCK loss: frozen baseline minus ON800']):
        limit=max(float(np.abs(data).max()),.01)
        im=ax.imshow(data,cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto')
        ax.set_yticks(range(len(groups)),groups);ax.set_xticks(range(23),NAMES,rotation=45,ha='right')
        ax.set_title(label);ax.axvline(6.5,color='black',ls='--',lw=.8);ax.axvline(13.5,color='black',ls='--',lw=.8)
        fig.colorbar(im,ax=ax,shrink=.85)
    fig.suptitle('Per-joint changes on reused synthetic development / red = worse, blue = better\nDashed lines delimit tail channels. Point-weighted diagnostics; no real manual GT.',fontsize=12)
    fig.savefig(OUT/'joint_changes.png',dpi=140);plt.close(fig)


if __name__=='__main__':main()
