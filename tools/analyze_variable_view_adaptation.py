"""Fixed-gate analysis and visual review artifacts after the bounded pilot ends."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from train_variable_view_adaptation import ROOT, OUTPUT, torch, write_json, fingerprint


def assess(result):
    limits=result['config']['acceptance']; arms=result['arms']
    fixed=arms['fixed_two']['summary']; variable=arms['variable_one_to_four']['summary']
    checks={}
    for group in ('V1','V3','V4'):
        checks[f'{group}_nme_improves']=variable[group]['box_matched_nme'] <= fixed[group]['box_matched_nme']*limits['variable_vs_fixed_nme_ratio_max_for_v1_v3_v4']
    checks['fixed_pair_nme_retained']=variable['fixed_pair']['box_matched_nme']<=fixed['fixed_pair']['box_matched_nme']*limits['fixed_pair_nme_ratio_max']
    checks['fixed_pair_pck_retained']=variable['fixed_pair']['detection_aware_pck_0.1']>=fixed['fixed_pair']['detection_aware_pck_0.1']-limits['fixed_pair_detection_aware_pck_drop_max']
    for group in ('V1','V2','V3','V4','fixed_pair'):
        checks[f'{group}_body_depth_retained']=variable[group]['body_depth_absrel']<=fixed[group]['body_depth_absrel']*limits['body_depth_absrel_ratio_max']
    return dict(checks=checks,development_gate_passed=all(checks.values()),production_promoted=False,
        warning='Small reused development set and one seed. Passing is not independent generalization or deployment approval.')


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args(); out=args.output
    if json.loads((out/'status.json').read_text())['state']!='completed': raise RuntimeError('Training has not completed')
    result=json.loads((out/'results.json').read_text()); spec=json.loads((out/'protocol.json').read_text())
    if result['protocol_sha256']!=fingerprint(out/'protocol.json'): raise ValueError('Protocol/report mismatch')
    audit=assess(result); audit['training_budget']={}
    hashes=set()
    for arm,values in result['arms'].items():
        checkpoint=torch.load(out/arm/'final_heads.pt',weights_only=True,map_location='cpu')
        if checkpoint['protocol_sha256']!=result['protocol_sha256'] or checkpoint['steps']!=spec['config']['steps']:
            raise ValueError('Checkpoint provenance/step mismatch')
        hashes.add(checkpoint['initial_hash'])
        history=json.loads((out/arm/'history.json').read_text())
        if len(history)!=spec['config']['steps']: raise ValueError('Incomplete history')
        audit['training_budget'][arm]={key:values[key] for key in ('images_seen','patch_tokens_seen','training_seconds','peak_gpu_gib')}
        audit['training_budget'][arm]['view_count_steps']={str(n):sum(len(r['views'])==n for r in history) for n in range(1,5)}
        del checkpoint
    if len(hashes)!=1: raise AssertionError('Different arm initializations')
    audit['initializations_equal']=True
    write_json(out/'acceptance.json',audit)
    groups=['V1','V2','V3','V4','fixed_pair']; colors=['#718a9a','#ce9453','#25897f']
    fig,axes=plt.subplots(1,3,figsize=(15,4.7),layout='constrained')
    for ax,key,title in zip(axes,('box_matched_nme','detection_aware_pck_0.1','body_depth_absrel'),
                            ('Box-matched keypoint NME','Detection-aware PCK@0.1','Body depth AbsRel')):
        ax.plot(groups,[result['baseline']['summary'][g][key] for g in groups],
                ls='--',marker='.',color='#333333',label='initial checkpoint')
        for color,(arm,values) in zip(colors,result['arms'].items()):
            ax.plot(groups,[values['summary'][g][key] for g in groups],marker='o',color=color,label=arm)
        ax.set_title(title); ax.grid(alpha=.2); ax.tick_params(axis='x',labelsize=8)
    axes[0].legend(fontsize=8)
    fig.suptitle('768 adaptation | 32 train + 8 reused development poses | 400 updates per arm')
    fig.supxlabel('Equal updates and scene schedule, not equal image/token budget. No production promotion.',fontsize=10)
    fig.savefig(out/'comparison.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for ax,arm,color in zip(axes,result['arms'],colors):
        h=json.loads((out/arm/'history.json').read_text())
        for n in range(1,5):
            points=[x for x in h if len(x['views'])==n]
            if points: ax.scatter([p['step'] for p in points],[p['pose_loss'] for p in points],label=f'V={n}',alpha=.6,s=7)
        ax.set_title(arm); ax.set_xlabel('Update'); ax.set_ylabel('Pose loss'); ax.legend(fontsize=8)
    fig.suptitle('Training pose loss stratified by view count (unsmoothed)')
    fig.savefig(out/'training_curves.png',dpi=150); plt.close(fig)
    # Inspect both best and worst V4 change against fixed control, not just successes.
    fixed_paths=sorted((out/'fixed_two').glob('prediction_*.pt'))
    candidates=[]
    for p in fixed_paths:
        a=torch.load(p,weights_only=True); b=torch.load(out/'variable_one_to_four'/p.name,weights_only=True)
        if a['group']=='V4': candidates.append((b['metrics']['box_matched_nme']-a['metrics']['box_matched_nme'],p.name))
    for label,(_,name) in zip(('best','worst'),(min(candidates),max(candidates))):
        fig,axes=plt.subplots(2,4,figsize=(17,6),layout='constrained')
        frame=None
        for row,arm in enumerate(('fixed_two','variable_one_to_four')):
            item=torch.load(out/arm/name,weights_only=True); r=item['record']; frame=r['frame']
            h,w=r['image_hw']; scale=np.array([w,h])
            for v,camera in enumerate(r['views']):
                ax=axes[row,v]
                rgb=Image.open(ROOT/r['scene']/'blended_images'/f'{camera:08d}.jpg').convert('RGB').resize((w,h),Image.Resampling.BICUBIC)
                ax.imshow(rgb)
                qs,ms=item['oracle_assignment'][v]
                for q,m in zip(qs,ms):
                    gt=item['targets']['keypoint_xy'][0,v,m].numpy()*scale
                    pr=item['predictions']['instance_keypoint_xy'][0,v,q].numpy()*scale
                    valid=item['targets']['keypoint_valid'][0,v,m].numpy()
                    visible=item['targets']['keypoint_visible'][0,v,m].numpy()
                    ax.scatter(gt[valid&visible,0],gt[valid&visible,1],s=12,facecolors='none',edgecolors='#40ff75')
                    ax.scatter(gt[valid&~visible,0],gt[valid&~visible,1],s=12,facecolors='none',edgecolors='#ffb730')
                    ax.scatter(pr[valid,0],pr[valid,1],s=10,c='#ff3b60',marker='x')
                ax.set_title(f'{arm} | camera {camera}',fontsize=9); ax.set_xlim(0,w); ax.set_ylim(h,0); ax.axis('off')
        fig.suptitle(f'{label.capitalize()} V4 NME change: frame {frame} | green/amber GT, red prediction | GT-assisted match')
        fig.savefig(out/f'{label}_v4_comparison.png',dpi=150); plt.close(fig)
    print(json.dumps(audit,indent=2))


if __name__=='__main__': main()
