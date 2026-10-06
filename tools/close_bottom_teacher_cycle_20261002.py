"""Build evidence-backed handoff after teacher training and bounded diagnostics."""
import datetime as dt
import fcntl
import html
import json
from pathlib import Path
import shutil
import subprocess
from bottom_anchor_common_20261002 import ROOT,OUT,PRIOR,write_state


def main():
    state=json.loads((OUT/'state.json').read_text());assert state.get('pid') is None and state.get('active_run') is None
    plan=json.loads((OUT/'plan.json').read_text())
    lock=(ROOT/plan['gpu_lock']).open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    dest=OUT/'cycle_handoff';dest.mkdir(exist_ok=False)
    old=json.loads((OUT/'extended_teacher_review_v2/summary.json').read_text())
    paired=json.loads((OUT/'quadruped_seed_replication_comparison_v1/summary.json').read_text())
    metrics=dict(old['models']);metrics.update(paired['canonical_metrics'])
    runs=['dlc_75to150_v1','dlc_150to200_v1','lp_context_30to60_v2','superanimal_quadruped_hrnet_v1','superanimal_quadruped_hrnet_seed29_v1']
    completed={r:json.loads((OUT/r/'completed.json').read_text()) for r in runs}
    numeric=json.loads((OUT/'quadruped_batch_peak_native_norm_v2/summary.json').read_text())
    assert all(r['variants']['gpu_native_norm_single']['max_distance_to_sparse']==0 for r in numeric['results'])
    assert all(r['variants']['gpu_temporal8']['max_distance_to_temporal']==0 for r in numeric['results'])
    workers=[]
    names=['train_teachers_extended_20261002.py','train_superanimal_quadruped_20261002.py',
           'infer_quadruped_temporal_20261002.py','audit_quadruped_batch_peaks_20261002.py']
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:args=(p/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if args and b'python' in args[0] and any(Path(a.decode(errors='replace')).name in names for a in args[1:]):workers.append(p.name)
    assert not workers,workers
    gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
    report=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),status='teacher_training_and_diagnostics_complete',
        broader_foundation_dataset_status='NOT complete: no new strong3D/pseudo-GT release or production checkpoint replacement.',
        stop_reason='Completed requested full training stages plus Quadruped replication, uncertainty and continuous-clip checks. Further blind epoch extension is not supported. Stop early under plan rather than consume remaining budget.',
        deadline_utc=plan['deadline_utc'],completed_training=completed,models=metrics,
        cohort=dict(images=281,observed_points=1330,observed_paws=753,input_wh=[704,768],source_wh=[1080,1200],confidence_filter=False,
            sessions=['AZR2L2','FR2'],independent_test=False,locked_holdouts_accessed=False),
        incumbent='SuperAnimal-Quadruped HRNetW32 seed17 best-mAP110; retain ResNet200, LPoldbest24 and all new checkpoints for comparison.',
        model_identity='Official Quadruped backbone,1830 tensors verified; new7-channel target head, not TopViewMouse, not ImageNet fallback. Paw-center and dorsal semantics do not blindly inherit39-point decoder.',
        optimization='ResNet retained Adam moments and completed explicit extended LR phases through200. LP retained optimizer and completed60; no sustained export improvement. Quadruped seeds17/29 each completed150, LR1e-4 then3e-5 and9e-6.',
        failed_or_limited=['LP first retry failed before training due endpoint existence guard; v2 repaired native endpoint repeat behavior and completed60.',
            'ResNet late150-to200 improvement modest; LP continuation did not clearly improve. Neither failure proves architecture intrinsically unsuitable.',
            'Quadruped final checkpoints need not outperform best-mAP checkpoints; seed29 severe paw errors rise3to6 by150.',
            'Two-seed mean reduces paw mean to7.44px on development but may average incompatible anatomical hypotheses; not promoted.',
            '77 observed points share>25px error despite<=10px seed separation;67 dorsal. Agreement/smoothness not sufficient for acceptance.',
            'Whole-image uncertainty rankings are dominated by missing-label channels; unannotated is unknown, not invisible or incorrect.',
            'Continuous clips show both stable wrong localization and several-frame identity/location failures; sparse84-image validation is not denseGT.'],
        numerical_resolution='Two one-cell differences reproduced by native Albumentations normalization+GPU single inference versus tensor normalization/batch inference. Same pixels, frozen BN, near-equal neighboring peaks and integer stride4 argmax. No frame shift. Normalization difference max4.77e-7; coordinate sensitivity~6.25sourcepx. Fix evaluation conventions before sub-cell claims; no environment/model changes made.',
        recommended_next=['Human-review a small set of shared dorsal/limb ambiguities using synchronized views and exact keypoint definitions; retain uncertain labels rather than auto-relabel.',
            'Separate keypoint localization from visibility/eligibility in teacher-to-GT pipeline. Missing annotations must remain unknown; acquire explicit visibility labels if training such a head.',
            'Standardize native preprocessing/inference precision and batch settings, then separately test subpixel decoding with fixed weights and same validation cohort; do not average distinct peaks.',
            'Only after reliability gates are justified, reconsider geometry/temporal3D with held-view evaluation and rejection of ambiguous points. Independent evaluation requires a separately authorized frozen protocol.'],
        safety=dict(active_owned_workers=workers,gpu_compute_processes=gpu,shared_GPU_lock_acquired=True,prior_STOP_preserved=(PRIOR/'STOP').exists(),
            labels_modified=False,calibration_modified=False,locked_holdouts_accessed=False,pseudo_gt_released=False,git_pushed=False),
        automation='Pause mousegeoforge through app tool after verifying this handoff; preserve schedule/prompt, do not auto-extend.')
    (dest/'summary.json').write_text(json.dumps(report,indent=2))
    headers=['模型','全点原图均值','全点输入均值','四足原图均值','四足中位数','四足P90','四足>100px','底视四足均值','侧视四足均值']
    rows=[]
    for name,r in metrics.items():
        a=r['all'];p=a['paws_source']
        vals=[name,f'{a["source"]["mean"]:.2f}',f'{a["input"]["mean"]:.2f}',f'{p["mean"]:.2f}',f'{p["median"]:.2f}',f'{p["p90"]:.2f}',str(a['paws_over100']),f'{r["bottom"]["paws_source"]["mean"]:.2f}',f'{r["side"]["paws_source"]["mean"]:.2f}']
        rows.append('<tr>'+''.join(f'<td>{html.escape(v)}</td>' for v in vals)+'</tr>')
    links=[('双随机种子困难样本','quadruped_seed_replication_comparison_v1/index.html'),('一致性/置信度诊断','quadruped_uncertainty_audit_v1/index.html'),
           ('连续短片慢放','quadruped_temporal_review_v1/index.html'),('相邻五帧错误','quadruped_temporal_review_v1/focused.html')]
    page='<meta charset="utf-8"><title>MouseGeoForge 教师训练交接</title><style>body{font-family:Arial,sans-serif;margin:32px;line-height:1.6;max-width:1300px}table{border-collapse:collapse}td,th{padding:7px 12px;border:1px solid #ccc}th{background:#edf3f7}pre{white-space:pre-wrap;font-size:12px}</style><h1>MouseGeoForge 教师训练与可靠性诊断</h1>'
    page+='<p>两类教师均已完成扩展训练；四足版SuperAnimal完成两随机种子各150轮。保留seed17最佳110为当前候选，不替换正式模型。未导出伪GT或强3D。</p><p>同一281张开发图、1330个已标注点/753个四足点，未经置信度筛选。输入704×768，原图1080×1200。不是独立测试；全部均值单位为像素。</p>'
    page+='<table><tr>'+''.join(f'<th>{h}</th>' for h in headers)+'</tr>'+''.join(rows)+'</table>'
    page+='<h2>判断</h2><p>继续增加轮数不是当前首要动作。残余问题包括背部表面点语义、遮挡/左右及前后足混淆、未标注点可靠性、近等高热图峰的网格量化。稳定、置信度高或两模型一致都不能单独证明预测正确。</p>'
    page+='<h2>下一步</h2><ol><li>小规模多视角人工复核背部和足部歧义，不自动改标签。</li><li>区分定位、可见性和伪标签可用性；缺失保持unknown。</li><li>统一推理数值约定，再独立评估亚像素解码。</li><li>通过可靠性验证后再恢复时序/多视角3D，不以拟合视角残差代替独立验证。</li></ol>'
    page+='<h2>证据入口</h2><ul>'+''.join(f'<li><a href="../{p}">{label}</a></li>' for label,p in links)+'</ul>'
    page+='<h2>完整记录</h2><pre>'+html.escape(json.dumps(report,indent=2,ensure_ascii=False))+'</pre>'
    (dest/'index.html').write_text(page)
    export=Path('/mnt/c/Users/55395/Downloads/MouseGeoForge-BottomAnchor-20261002')/dest.name;shutil.copytree(dest,export)
    write_state(stage='cycle_complete_pending_automation_pause',pid=None,active_run=None,queue_pid=None,queue_child_pid=None,mode='idle',job_deadline_utc=None,
        latest_result=str(dest/'summary.json'),latest_decision=str(dest/'summary.json'),latest_training_visual_directory=str(export),
        next_action='Pause automation after handoff verification. Further work requires new user direction; no automatic retraining or pseudo-GT release.')
    print(json.dumps(dict(export=str(export),owned_workers=workers,gpu_compute_processes=gpu,models=len(metrics)),indent=2))


if __name__=='__main__':main()
