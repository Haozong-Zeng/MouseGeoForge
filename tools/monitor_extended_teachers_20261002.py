"""Read-only worker monitoring plus derived scientific curves; never launch training."""
import datetime as dt
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bottom_anchor_common_20261002 import ROOT,OUT,PRIOR,budget,write_state


def records(path):
    if not path.exists():return []
    rows=[]
    for line in path.read_text().splitlines():
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:pass  # concurrent append may expose the unfinished last line
    return rows


def main():
    plan,_,check=budget(120);check()
    state=json.loads((OUT/'state.json').read_text())
    now=dt.datetime.now(dt.timezone.utc);stamp=now.strftime('%Y%m%d_%H%M%S')
    dest=OUT/f'training_monitor_{stamp}';dest.mkdir()
    lock=(ROOT/plan['gpu_lock']).open('a+')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);held=False;fcntl.flock(lock,fcntl.LOCK_UN)
    except BlockingIOError:held=True
    proc={k:bool(state.get(k) and Path(f'/proc/{state[k]}').exists()) for k in ['pid','queue_pid','queue_child_pid']}
    rows=records(OUT/'dlc_75to150_v1/epochs.jsonl');train=[r for r in rows if r['mode']=='train'];valid=[r for r in rows if r['mode']=='eval']
    lp_run='lp_context_30to60_v2' if (OUT/'lp_context_30to60_v2').exists() else 'lp_context_30to60_v1'
    lp=records(OUT/lp_run/'epochs.jsonl')
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    if train:axes[0,0].plot([r['epoch'] for r in train],[r['loss'] for r in train],label='train')
    if valid:
        x=[r['epoch'] for r in valid]
        axes[0,0].plot(x,[r['loss'] for r in valid],label='validation')
        axes[0,1].plot(x,[r['metrics']['metrics/test.rmse'] for r in valid],marker='o',label='DLC native test.rmse (unfiltered)')
        axes[1,0].plot(x,[r['metrics']['metrics/test.mAP'] for r in valid],marker='o',label='validation mAP')
    if train:axes[1,1].step([r['epoch'] for r in train],[r['lr'][0] for r in train],where='post',label='actual train LR')
    for ax,title in zip(axes.ravel(),['Loss: may plateau while localization improves','Validation, INPUT pixels (704x768)','Validation mAP','Actual learning rate']):
        ax.set_title(title);ax.set_xlabel('total DLC epochs');ax.grid(alpha=.25);ax.legend()
    axes[0,0].set_yscale('log');axes[1,1].set_yscale('log')
    fig.suptitle('Extended DLC: fixed 281-image development cohort; no confidence filtering',fontsize=13)
    fig.savefig(dest/'dlc_learning_curve.png',dpi=150);plt.close(fig)
    report=dict(utc=now.isoformat(),active_run=state.get('active_run'),worker_pids_alive=proc,gpu_lock_held=held,
        prior_STOP_preserved=(PRIOR/'STOP').exists(),dlc_latest_train=train[-1] if train else None,
        dlc_evaluations=valid,lp_latest=lp[-1] if lp else None,
        gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv'],capture_output=True,text=True).stdout,
        interpretation='DLC validation loss is near-flat, but native localization and mAP improve across the extended phase. Do not stop on loss alone. Native test.rmse is reported verbatim in input pixels; source-coordinate EPE and severe-paw-error audit require final prediction export.',
        next_action='Finish DLC150 then run queued LP60; compare old/new on identical filtered cohort. DLC extension to200 is likely justified if final-stage validation improvement persists. Do not interrupt the queued LP stage.',
        no_checkpoint_or_training_mutation=True,locked_holdout_accessed=False,pseudo_gt_released=False)
    (dest/'summary.json').write_text(json.dumps(report,indent=2))
    page='<html><meta charset="utf-8"><h1>教师续训进度</h1><p>DLC验证loss基本不变，但定位误差与mAP改善。图中误差是704×768输入尺度，尚未进行最终原图尺度逐点核对；未按置信度筛点。LP仍按队列接续训练。旧标签和原模型保持不变。</p><img width="1200" src="dlc_learning_curve.png"><pre>'+json.dumps(report,indent=2)+'</pre></html>'
    (dest/'index.html').write_text(page)
    export=Path('/mnt/c/Users/55395/Downloads/MouseGeoForge-BottomAnchor-20261002')/dest.name
    shutil.copytree(dest,export)
    # Do not overwrite live worker state, PIDs, epoch, deadline or stage.
    write_state(latest_training_monitor=str(dest/'summary.json'),latest_training_visual_directory=str(export),
        convergence_decision='Continue both requested stages; DLC localization/mAP still improving despite nearly flat validation loss.')
    print(json.dumps(dict(report=str(dest/'summary.json'),export=str(export),dlc_epoch=train[-1]['epoch'] if train else None,
        latest_eval=valid[-1] if valid else None,lp_epoch=lp[-1]['epoch'] if lp else None),indent=2))


if __name__=='__main__':main()
