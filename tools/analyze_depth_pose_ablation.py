"""Summarize fixed-budget paired runs without selecting seeds/checkpoints."""
import argparse
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'output/depth_pose_ablation_pilot_v1')
    a=p.parse_args()
    results=json.loads((a.output/'results.json').read_text())
    assert results['passed']
    runs=results['runs']; cfg=results['config']
    metrics=['box_matched_nme','detection_aware_pck_0.1','body_depth_absrel',
             'visible_nme','occluded_nme','body_keypoint_nme','tail_nme',
             'matched_mask_iou','detection_precision','detection_recall','all_depth_absrel']
    summary={}
    groups=['source_macro']+list(runs[0]['validation']['per_source'])
    for group in groups:
        summary[group]={}
        for metric in metrics:
            def value(run):
                m=run['validation']['source_macro'] if group=='source_macro' else run['validation']['per_source'][group]
                return m[metric]
            values={mode:[value(next(r for r in runs if r['seed']==s and r['mode']==mode)) for s in cfg['seeds']]
                    for mode in cfg['modes']}
            if any(v is None for vs in values.values() for v in vs): continue
            row={mode:dict(mean=float(np.mean(vs)),seed_sd=float(np.std(vs,ddof=1)),values=vs)
                 for mode,vs in values.items()}
            delta=np.array(values['local_depth_features'])-np.array(values['off'])
            row['paired_on_minus_off']=dict(mean=float(delta.mean()),seed_sd=float(delta.std(ddof=1)),values=delta.tolist())
            summary[group][metric]=row
    for s in cfg['seeds']:
        pair=[r for r in runs if r['seed']==s]
        assert len(pair)==2
        assert pair[0]['initial_state_sha256']==pair[1]['initial_state_sha256']
        assert pair[0]['train_order_sha256']==pair[1]['train_order_sha256']
        assert pair[0]['initial_validation']==pair[1]['initial_validation']
        assert all(r['steps']==cfg['steps'] for r in pair)
    (a.output/'comparison.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    lines=['# Depth–pose explicit coupling pilot','',
           'Fixed final checkpoints; mean ± sample SD across paired initialization seeds, not independent animal confidence intervals.',
           '',f"Seeds: {cfg['seeds']}; {cfg['steps']} steps per mode/seed; T=off, tail-curve=off.",'']
    for group,rows in summary.items():
        lines += [f'## {group}','','| Metric | J off | J on | Paired on−off |','|---|---:|---:|---:|']
        for metric,row in rows.items():
            cols=[f"{row[k]['mean']:.4f} ± {row[k]['seed_sd']:.4f}" for k in ('off','local_depth_features','paired_on_minus_off')]
            lines.append(f"| {metric} | {' | '.join(cols)} |")
        lines.append('')
    lines += ['NME: pixel error / GT box diagonal, assignment uses boxes/objectness only.',
              'Detection-aware PCK: NME ≤ 0.1 AND score ≥ 0.5 AND box IoU ≥ 0.5; missed targets fail.',
              'Depth AbsRel uses valid rendered surface depth; mouse masks exclude fur curves.',
              'Groups are averaged equally. Same synthetic assets/rigs remain shared; no real-domain or unseen-animal claim.']
    (a.output/'comparison.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,3.6),layout='constrained')
    for ax,metric in zip(axes,metrics):
        row=summary['source_macro'][metric]
        for i,seed in enumerate(cfg['seeds']):
            ax.plot([0,1],[row['off']['values'][i],row['local_depth_features']['values'][i]],'o-',label=str(seed))
        ax.set_xticks([0,1],['J off','J on']); ax.set_title(metric); ax.grid(alpha=.2)
    axes[0].legend(title='Seed')
    fig.savefig(a.output/'paired_metrics.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for r in runs:
        history=json.loads((a.output/f"seed{r['seed']}_{r['mode']}"/'history.json').read_text())
        for ax,key in zip(axes,('pose_loss','depth_loss')):
            y=np.array([s[key] for s in history]); y=np.convolve(y,np.ones(25)/25,mode='valid')
            ax.plot(np.arange(25,len(history)+1),y,alpha=.7,label=f"{r['seed']} {r['mode']}")
            ax.set_title(f'{key}, 25-step mean'); ax.set_xlabel('Updates'); ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.savefig(a.output/'learning_curves.png',dpi=160); plt.close(fig)
    print(json.dumps(summary['source_macro'],indent=2))


if __name__=='__main__': main()
