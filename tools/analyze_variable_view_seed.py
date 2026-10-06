"""Original-checkpoint retention gates and balanced visual diagnostics for replay V2."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from train_variable_view_seed_control import ROOT, OUTPUT, torch, write_json, fingerprint


def assess(result):
    limits=result['config']['acceptance']; initial=result['baseline']['summary']
    audits={}
    for arm,data in result['arms'].items():
        summary=data['summary']; checks={}
        for group in ('V1','V3','V4'):
            checks[group+'_nme_improves']=summary[group]['box_matched_nme']<=initial[group]['box_matched_nme']*limits['new_view_nme_ratio_max']
        for group in ('fixed_pair','single_original_pair'):
            checks[group+'_nme_retained']=summary[group]['box_matched_nme']<=initial[group]['box_matched_nme']*limits['retention_nme_ratio_max']
            checks[group+'_pck_retained']=summary[group]['detection_aware_pck_0.1']>=initial[group]['detection_aware_pck_0.1']-limits['retention_detection_aware_pck_drop_max']
        for group in initial:
            checks[group+'_depth_retained']=summary[group]['body_depth_absrel']<=initial[group]['body_depth_absrel']*limits['body_depth_absrel_ratio_max']
        audits[arm]=dict(checks=checks,development_gate_passed=all(checks.values()))
    return dict(arms=audits,reference='original checkpoint',production_promoted=False,
                warning='Two fresh seeds plus seed17 reference, reused development set, GT-assisted box matching. Not independent test or deployment.')


def visual_cases(out,result,target_arm):
    for group in ('V4','single_original_pair','fixed_pair'):
        candidates=[]
        for p in sorted((out/'initial').glob('prediction_*.pt')):
            a=torch.load(p,weights_only=True)
            if a['group']!=group: continue
            b=torch.load(out/target_arm/p.name,weights_only=True)
            candidates.append((b['metrics']['box_matched_nme']-a['metrics']['box_matched_nme'],p.name))
        for label,(_,name) in zip(('best','worst'),(min(candidates),max(candidates))):
            count=4 if group=='V4' else 2
            fig,axes=plt.subplots(3,count,figsize=(4.2*count,9),layout='constrained',squeeze=False)
            for row,arm in enumerate(('initial',*result['arms'])):
                item=torch.load(out/arm/name,weights_only=True); r=item['record']; h,w=r['image_hw']
                for v,camera in enumerate(r['views']):
                    ax=axes[row,v]
                    rgb=Image.open(ROOT/r['scene']/'blended_images'/f'{camera:08d}.jpg').convert('RGB').resize((w,h),Image.Resampling.BICUBIC)
                    ax.imshow(rgb)
                    qs,ms=item['oracle_assignment'][v]
                    for q,m in zip(qs,ms):
                        gt=item['targets']['keypoint_xy'][0,v,m].numpy()*np.array([w,h])
                        pr=item['predictions']['instance_keypoint_xy'][0,v,q].numpy()*np.array([w,h])
                        valid=item['targets']['keypoint_valid'][0,v,m].numpy()
                        visible=item['targets']['keypoint_visible'][0,v,m].numpy()
                        for mask,color in ((valid&visible,'#40ff75'),(valid&~visible,'#ffb730')):
                            ax.scatter(gt[mask,0],gt[mask,1],s=12,facecolors='none',edgecolors=color)
                        ax.scatter(pr[valid,0],pr[valid,1],s=10,c='#ff3b60',marker='x')
                    ax.set_title(f'{arm} | camera {camera}',fontsize=9); ax.set_xlim(0,w); ax.set_ylim(h,0); ax.axis('off')
            fig.suptitle(f'{label} {target_arm} minus ORIGINAL NME | {group} frame {r["frame"]}\nGreen/amber GT, red prediction; GT-assisted match')
            fig.savefig(out/f'{label}_{group}.png',dpi=130); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,default=OUTPUT)
    out=parser.parse_args().output
    if json.loads((out/'status.json').read_text())['state']!='completed': raise RuntimeError('Run incomplete')
    result=json.loads((out/'results.json').read_text()); spec=json.loads((out/'protocol.json').read_text())
    if result['protocol_sha256']!=fingerprint(out/'protocol.json'): raise ValueError('Protocol mismatch')
    audit=assess(result); audit['budgets']={}; hashes=set()
    parent=ROOT/result['config']['parent_run']
    assert fingerprint(parent/'results.json')==spec['parent_results_sha256']
    reference=json.loads((parent/'results.json').read_text())['arms']['replay_lr3e6']
    audit['external_seed17_reference']=reference['summary']
    target_arm=min(result['arms'],key=lambda arm:(not audit['arms'][arm]['development_gate_passed'],
        result['arms'][arm]['summary']['fixed_pair']['box_matched_nme']))
    audit['visual_target_arm']=target_arm
    for arm,data in result['arms'].items():
        checkpoint=torch.load(out/arm/'final_heads.pt',weights_only=True,map_location='cpu')
        assert checkpoint['protocol_sha256']==result['protocol_sha256'] and checkpoint['steps']==spec['config']['steps']
        hashes.add(checkpoint['initial_hash'])
        history=json.loads((out/arm/'history.json').read_text())
        assert len(history)==spec['config']['steps']
        tokens=sum(len(r['views'])*(r['image_hw'][0]//16)*(r['image_hw'][1]//16) for r in history)
        assert tokens==data['patch_tokens_seen']==history[-1]['patch_tokens_seen']
        assert all(math.isfinite(r[k]) for r in history for k in ('loss','pose_loss','depth_loss','gradient_norm'))
        audit['budgets'][arm]={k:data[k] for k in ('images_seen','patch_tokens_seen','training_seconds','peak_gpu_gib')}
        audit['budgets'][arm]['view_count_steps']=dict(Counter(str(len(r['views'])) for r in history))
        audit['budgets'][arm]['slot_kinds']=dict(Counter(r['slot_kind'] for r in history))
        audit['budgets'][arm]['instance_count_steps']=dict(Counter(str(spec['records'][spec['arms'][arm][i][0]]['expected_instances']) for i in spec['schedule']))
    assert len(hashes)==1
    audit['fresh_seeds_individually_passed']=all(v['development_gate_passed'] for v in audit['arms'].values())
    audit['replication_passed']=audit['fresh_seeds_individually_passed']
    write_json(out/'acceptance.json',audit)
    groups=list(result['baseline']['summary']); fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained')
    for ax,key in zip(axes,('box_matched_nme','detection_aware_pck_0.1','body_depth_absrel')):
        for label,values in [('original',result['baseline']),('V3 seed17 lr3e6',reference),*result['arms'].items()]:
            ax.plot(groups,[values['summary'][g][key] for g in groups],marker='o',label=label)
        ax.set_title(key); ax.grid(alpha=.2); ax.tick_params(axis='x',rotation=20,labelsize=8)
    axes[0].legend(fontsize=8); fig.suptitle('768 seed replication | 800 updates each | same source pools, new view draws | development only')
    fig.savefig(out/'comparison.png',dpi=150); plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for col,arm in enumerate(result['arms']):
        history=json.loads((out/arm/'history.json').read_text())
        for n in range(1,5):
            rows=[r for r in history if len(r['views'])==n]
            axes[0,col].scatter([r['step'] for r in rows],[r['pose_loss'] for r in rows],s=5,alpha=.5,label=f'V{n}')
        for kind in ('shared','dual_anchor','single_anchor'):
            rows=[r for r in history if r['slot_kind']==kind]
            axes[1,col].scatter([r['step'] for r in rows],[r['depth_loss'] for r in rows],s=5,alpha=.5,label=kind)
        for row in range(2): axes[row,col].set_title(arm); axes[row,col].legend(fontsize=8); axes[row,col].set_xlabel('Update')
        axes[0,col].set_ylabel('Pose loss'); axes[1,col].set_ylabel('Depth loss')
    fig.savefig(out/'training_curves.png',dpi=140); plt.close(fig)
    visual_cases(out,result,target_arm)
    print(json.dumps(audit,indent=2))


if __name__=='__main__': main()



