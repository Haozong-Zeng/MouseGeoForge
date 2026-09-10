"""Archive the fixed-budget corrected Joff baseline without selecting checkpoints."""
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_depth_pose_ablation import ROOT,write_json


def main():
    folder=ROOT/'output/depth_pose_baseline_5000_v2'
    result=json.loads((folder/'results.json').read_text())
    check=json.loads((folder/'checkpoint_verification.json').read_text())
    assert check['passed'] and len(result['runs'])==1
    run=result['runs'][0]
    assert run['seed']==17 and run['mode']=='off' and run['steps']==5000
    assert result['config']['local_heatmap_weight']==0
    assert all(r['passed'] and r['validated_scenes']==32 for r in check['runs'])
    curve=json.loads((folder/'seed17_off/training_curve.json').read_text())
    manifest=json.loads((folder/'manifest.json').read_text())
    assert run['training']['scenes']==64 and run['validation']['scenes']==32
    assert [r['step'] for r in curve]==list(range(500,5001,500))
    report=dict(**result,checkpoint_verification=check,training_curve=curve,
                split_manifest=manifest['records'],
                execution_verified=True,production_ready=False,paired_ablation=False,
                findings=[
                    'Eight-scene pose-only memorization passed; this separate 64-scene multitask baseline is not equally fitted across sources.',
                    'Single-mouse training is well fitted; dual-mouse training remains weaker and both held-out sources show a generalization gap.',
                    'Do not attribute differences from old 1000-step runs to one factor: loss weight and update budget both changed.',
                    'One seed and shared synthetic assets; no evidence for real-animal generalization or coupling benefit.',
                    'No additional training was launched during acceptance.'
                ],
                visual_review=dict(files=sorted(str(p.relative_to(ROOT)) for p in (folder/'seed17_off/visuals').glob('*.png')),
                    observations=[
                        'Dual frame235: body localization improves, but the brown-mouse top-view tail is shortened/misdirected.',
                        'Dual frame352: predicted points cluster and the brown-mouse tail is largely missing; matching colors do not establish identity tracking.',
                        'WashFace frames1/7: trunk is roughly located, but head landmarks extend beyond the mouse and tail extent is shortened.',
                        'Depth preserves coarse cage/body structure but is noisy and blurred at fine boundaries; top-view mouse contrast is weak.'
                    ]))
    write_json(ROOT/'docs/experiments/pose_baseline_5000_v2_results.json',report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    sources=list(curve[0]['per_source'])
    for key,label in [('source_macro','Train source-macro')]+[(s,s) for s in sources]:
        metrics=[r['source_macro'] if key=='source_macro' else r['per_source'][key] for r in curve]
        for ax,metric in zip(axes,('box_matched_nme','detection_aware_pck_0.1','body_depth_absrel')):
            ax.plot([r['step'] for r in curve],[m[metric] for m in metrics],label=label)
            ax.set(xlabel='Updates',ylabel=metric); ax.grid(alpha=.2)
    for ax,metric in zip(axes,('box_matched_nme','detection_aware_pck_0.1','body_depth_absrel')):
        ax.scatter([5000],[run['validation']['source_macro'][metric]],marker='x',s=65,c='black',label='Held-out final only')
    axes[0].legend(fontsize=7)
    fig.suptitle('Corrected Joff / seed 17 / 64 train + 32 held-out scenes / fixed final checkpoint')
    fig.savefig(folder/'training_curve.png',dpi=150); plt.close(fig)
    print(json.dumps(dict(training=run['training']['source_macro'],validation=run['validation']['source_macro'],
                          checkpoint_verification=check),indent=2))


if __name__=='__main__': main()
