"""Non-mutating training monitor; derived curves/decision only, no worker launch."""
import datetime as dt
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bottom_anchor_common_20261002 import ROOT,OUT,PRIOR,budget,write_state


def main(run_name=None):
    plan,_,check=budget(120);check()
    state=json.loads((OUT/'state.json').read_text())
    run_name=run_name or state.get('active_run')
    assert run_name in ('superanimal_quadruped_hrnet_v1','superanimal_quadruped_hrnet_seed29_v1')
    run=OUT/run_name
    assert state.get('active_run')==run_name,'Monitor must correspond to current active run'
    now=dt.datetime.now(dt.timezone.utc)
    dest=OUT/f'quadruped_monitor_{now:%Y%m%d_%H%M%S}';dest.mkdir()
    records=[]
    for line in (run/'epochs.jsonl').read_text().splitlines():
        try:records.append(json.loads(line))
        except json.JSONDecodeError:pass
    train=[r for r in records if r['mode']=='train'];valid=[r for r in records if r['mode']=='eval']
    with (ROOT/plan['gpu_lock']).open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);held=False
        except BlockingIOError:held=True
    alive=bool(state.get('pid') and Path(f'/proc/{state["pid"]}').exists())
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    x=[r['epoch'] for r in valid]
    axes[0,0].plot(x,[r['metrics']['metrics/test.rmse'] for r in valid],'o-',label='Quadruped validation, unfiltered')
    axes[0,0].set_title('INPUT pixel EPE; raw CPU audit confirms units')
    axes[0,1].plot(x,[r['metrics']['metrics/test.mAP'] for r in valid],'o-');axes[0,1].set_title('Validation mAP')
    if run_name.endswith('seed29_v1'):
        reference=[json.loads(line) for line in (OUT/'superanimal_quadruped_hrnet_v1/epochs.jsonl').read_text().splitlines()]
        reference=[r for r in reference if r['mode']=='eval' and r['epoch']<=valid[-1]['epoch']]
        axes[0,0].plot([r['epoch'] for r in reference],[r['metrics']['metrics/test.rmse'] for r in reference],'--',label='seed17 matched epochs')
        axes[0,1].plot([r['epoch'] for r in reference],[r['metrics']['metrics/test.mAP'] for r in reference],'--',label='seed17 matched epochs')
    axes[0,0].legend();axes[0,1].legend()
    axes[1,0].plot([r['epoch'] for r in train],[r['loss'] for r in train],label='train epoch mean')
    axes[1,0].plot(x,[r['loss'] for r in valid],label='validation epoch mean')
    axes[1,0].set_yscale('log');axes[1,0].set_title('Loss');axes[1,0].legend()
    axes[1,1].step([r['epoch'] for r in train],[r['lr'][0] for r in train],where='post')
    axes[1,1].set_yscale('log');axes[1,1].set_title('Actual training LR (decays after100/130)')
    for ax in axes.ravel():
        ax.axvline(100,color='gray',linestyle='--',alpha=.5)
        ax.axvline(130,color='gray',linestyle='--',alpha=.5)
        ax.set_xlabel('total epochs');ax.grid(alpha=.3)
    fig.savefig(dest/'learning_curves.png',dpi=140);plt.close(fig)
    recent=valid[-5:]
    intervals=[(dt.datetime.fromisoformat(b['utc'])-dt.datetime.fromisoformat(a['utc'])).total_seconds()/(b['epoch']-a['epoch']) for a,b in zip(recent,recent[1:])]
    eta_seconds=(150-train[-1]['epoch'])*statistics.median(intervals) if intervals else None
    phase='before first LR decay' if train[-1]['epoch']<=100 else ('between planned LR decays' if train[-1]['epoch']<=130 else 'after both planned LR decays')
    report=dict(utc=now.isoformat(),run=run_name,seed=json.loads((run/'settings.json').read_text()).get('seed'),pid=state.get('pid'),pid_alive=alive,shared_GPU_lock_held=held,
        last_train=train[-1],last_validation=valid[-1],recent_validation=recent,
        estimated_training_finish_utc=(now+dt.timedelta(seconds=eta_seconds)).isoformat() if eta_seconds else None,
        estimate_caveat='Recent walltime extrapolation; final export adds time.',prior_STOP_preserved=(PRIOR/'STOP').exists(),
        decision=f'Continue planned150-epoch replication; currently {phase}. Evaluate full schedule before judging convergence. Same reused development split, not independent validation. No duplicate GPU job.',
        final_audit=f'After completion inspect best-mAP snapshot and last150. Use audit_quadruped_snapshot_20261002.py --run-name {run_name} --epoch <saved_epoch> on CPU, same281 images, raw Euclidean errors and visuals. Compare seed17; do not select solely by confidence-filtered error.',
        metric_conventions=str(OUT/'metric_conventions_20261002.json'),
        no_worker_mutation=True,no_label_changes=True,no_holdout_access=True,pseudo_gt_released=False)
    (dest/'summary.json').write_text(json.dumps(report,indent=2))
    (dest/'index.html').write_text('<meta charset="utf-8"><h1>Quadruped训练曲线</h1><p>当前继续150轮；虚线为计划降学习率位置。未按置信度筛点，不与LP原生日志混用。</p><img width="1200" src="learning_curves.png"><pre>'+json.dumps(report,indent=2)+'</pre>')
    export=Path('/mnt/c/Users/55395/Downloads/MouseGeoForge-BottomAnchor-20261002')/dest.name
    shutil.copytree(dest,export)
    write_state(latest_training_monitor=str(dest/'summary.json'),latest_decision=str(dest/'summary.json'),
        convergence_decision=report['decision'])
    print(json.dumps(dict(report=str(dest/'summary.json'),export=str(export),epoch=train[-1]['epoch'],
        validation=valid[-1],pid_alive=alive,lock_held=held,estimated_finish=report['estimated_training_finish_utc']),indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run-name');args=ap.parse_args();main(args.run_name)
