"""Export compact results and a scientific learning-curve figure for the pose gate."""
import hashlib
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_depth_pose_ablation import ROOT, write_json, build_manifest


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def main():
    paths=[ROOT/'output'/f'pose_overfit_v1_{v}' for v in
           ('legacy','no_local_heatmap','detached_grid')]
    paths += [ROOT/'output'/f'pose_overfit_3000_v1_{v}' for v in
              ('no_local_heatmap','detached_grid')]
    runs=[]
    for path in paths:
        result=json.loads((path/'result.json').read_text())
        config=result['config']
        records=json.loads((path/'manifest.json').read_text())['records']
        history=json.loads((path/'history.json').read_text())
        audits=[r for r in history if 'coordinate_heatmap_gradient_cosine' in r]
        curves=json.loads((path/'evaluations.json').read_text())
        runs.append(dict(path=str(path.relative_to(ROOT)),variant=config['variant'],steps=config['steps'],
                         initial_state_sha256=config['initial_state_sha256'],
                         manifest_sha256=digest(records),schedule_first_2000_sha256=digest(config['schedule'][:2000]),
                         overfit_passed=result['overfit_passed'],reload_exact=result['reload_exact'],
                         metrics=result['final']['summary'],
                         gradient_audit=dict(samples=len(audits),coordinate_heatmap_cosine_median=statistics.median(
                             r['coordinate_heatmap_gradient_cosine'] for r in audits),
                             heatmap_to_coordinate_norm_median=statistics.median(
                             r['local_heatmap_gradient_norm']/max(r['coordinate_gradient_norm'],1e-12) for r in audits)),
                         curve=[dict(step=r['step'],**r['summary']['source_macro']) for r in curves]))
    for key in ('initial_state_sha256','manifest_sha256','schedule_first_2000_sha256'):
        assert len({r[key] for r in runs})==1,key
    assert all(r['reload_exact'] for r in runs)
    # Confirm the standalone split reconstruction matches the inputs used in these runs.
    registered=build_manifest(json.loads((ROOT/'configs/experiments/depth_pose_pilot_v1.json').read_text()))
    import numpy as np
    groups={}
    for r in registered['records']:
        if r['split']=='train': groups.setdefault(r['source_group'],[]).append(r)
    recreated=[rs[i] for rs in groups.values() for i in np.linspace(0,len(rs)-1,4).round().astype(int)]
    assert digest(recreated)==runs[0]['manifest_sha256']
    report=dict(training_only=True,seed=17,scene_count=8,views_per_scene=2,
                gate='Final step: NME < .05, PCK@.1 >= .9, detection recall >= .9',
                comparable_initialization_and_schedule=True,
                note='CUDA grid_sample backward is not guaranteed bitwise deterministic; fresh 3000-step runs are not exact extensions.',
                runs=runs)
    write_json(ROOT/'docs/experiments/pose_overfit_results.json',report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for run in runs:
        label=f"{run['variant']} / {run['steps']}"
        for ax,key in zip(axes,('box_matched_nme','box_matched_pck_0.1','window_coverage')):
            ax.plot([r['step'] for r in run['curve']],[r[key] for r in run['curve']],label=label)
            ax.set(xlabel='Training updates',ylabel=key); ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.suptitle('8 training scenes only / seed 17 / frozen encoder / no depth supervision')
    output=ROOT/'output/pose_overfit_summary'; output.mkdir(exist_ok=True)
    fig.savefig(output/'learning_curves.png',dpi=150); plt.close(fig)
    print(json.dumps([dict(variant=r['variant'],steps=r['steps'],passed=r['overfit_passed'],
                           nme=r['metrics']['source_macro']['box_matched_nme'],
                           pck=r['metrics']['source_macro']['box_matched_pck_0.1'],
                           gradient=r['gradient_audit']) for r in runs],indent=2))


if __name__=='__main__': main()
