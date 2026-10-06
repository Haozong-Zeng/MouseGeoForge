"""Scientific plots of the fixed CPU geometry diagnostic; no image editing."""
import json
import numpy as np
from run_depth_pose_ablation import ROOT,write_json,fingerprint


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    source=ROOT/'output/origin_calibrated_geometry_v1';out=ROOT/'output/origin_geometry_review_v1';out.mkdir(exist_ok=False)
    assert json.loads((source/'verification.json').read_text())['passed']
    summary=json.loads((source/'summary.json').read_text());rows=json.loads((source/'records.json').read_text())
    groups=['old_dual_pair','old_cylinder_side_pair','sides_bottom'];names=['Dual mice: side + top','Single mouse: two sides','Single mouse: sides + bottom']
    modes=['independent','joint'];arms=['frozen','base_origin','anchor_origin'];colors=['#555555','#3377bb','#d47718']
    fig,axs=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for column,(group,title) in enumerate(zip(groups,names)):
        for ai,arm in enumerate(arms):
            xs=np.arange(2)+(ai-1)*.23
            values=[summary[group][m+'/'+arm] for m in modes]
            axs[0,column].bar(xs,[r['coverage'] for r in values],width=.22,color=colors[ai],label=arm)
            axs[1,column].bar(xs,[r['common_support']['mean_normalized'] for r in values],width=.22,color=colors[ai])
        axs[0,column].set_title(title);axs[0,column].set_ylim(0,1.05)
        axs[1,column].set_title('Common points across all six arms: '+str(values[0]['common_support']['n']))
        for ax in axs[:,column]:ax.set_xticks(range(2),modes);ax.grid(axis='y',alpha=.2)
    axs[0,0].set_ylabel('Supported fraction (higher is better)');axs[1,0].set_ylabel('Body3D diagonal-normalized error (lower)')
    axs[0,0].legend(fontsize=8);fig.suptitle('Known calibration, GT-assisted instance association; synthetic only')
    fig.savefig(out/'coverage_and_common_error.png',dpi=140);plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(13,8),layout='constrained');selected=[]
    for column,(group,title) in enumerate(zip(groups,names)):
        index=min(r['index'] for r in rows if r['group']==group);selected.append(dict(group=group,index=index))
        rr=[r for r in rows if r['index']==index and r['mode']=='joint'];truth={(r['instance'],r['keypoint']):r['gt_xyz_native'] for r in rr}
        for ri,(xaxis,yaxis,plane) in enumerate([(0,1,'XY'),(0,2,'XZ')]):
            ax=axs[ri,column];gt=np.array(list(truth.values()));ax.scatter(gt[:,xaxis],gt[:,yaxis],s=24,facecolors='none',edgecolors='black',label='GT')
            for ai,arm in enumerate(arms):
                supported=[r for r in rr if r['arm']==arm and r['supported']];xyz=np.array([r['xyz_native'] for r in supported])
                if len(xyz):ax.scatter(xyz[:,xaxis],xyz[:,yaxis],s=13,color=colors[ai],alpha=.8,label=arm+f' ({len(supported)}/{len(truth)})')
            ax.set_title(title+' / '+plane);ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.2)
            ax.set_xlabel(plane[0]+' / '+rr[0]['units']);ax.set_ylabel(plane[1]);ax.legend(fontsize=6)
    fig.suptitle('Fixed first scene per group, joint encoding; unsupported points omitted and counted')
    fig.savefig(out/'fixed_scene_3d_projections.png',dpi=140);plt.close(fig)
    write_json(out/'verification.json',dict(passed=True,selected_first_scenes=selected,training_steps=0,new_image_reads=0,
        source_sha256={p.name:fingerprint(p) for p in (source/'records.json',source/'summary.json')},code_sha256=fingerprint(ROOT/'tools/visualize_origin_geometry.py')))
    print('COMPLETED')


if __name__=='__main__':main()
