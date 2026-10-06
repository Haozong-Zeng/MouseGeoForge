"""Read-only evidence audit, tests, dual-instance RGB sheet and cycle handoff.

Does not alter labels, models or automation. Automation pause is a separate app action.
"""
import collections,datetime as dt,fcntl,hashlib,html,io,json,os,re,shutil,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'vggt-omega'),str(ROOT/'tools')]
import numpy as np
import torch
from PIL import Image,ImageDraw
from run_surface_curve_pilot_20261003 import load_item
CYCLE=ROOT/'output/core23_spatial_cycle_20261004';OUT=CYCLE/'cycle_handoff'
DEST=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-Core23-Spatial-20261004')
DOC=ROOT/'docs/pymousestudio_core23_spatial_cycle_results_20261005.md'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def write(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

def main():
    assert not OUT.exists();plan=json.loads((CYCLE/'plan.json').read_text());lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    decision=json.loads((CYCLE/'full_detector_decision_v1.json').read_text());assert decision['rgb_complete']
    for folder in ['detection_full_context_control_v1','detection_full_context_pose_audit_v1','live_view_full_detector_paired800_audit_v1','live_view_full_detector_full800_audit_v1']:
        assert json.loads((CYCLE/folder/'status.json').read_text())['stage']=='complete'
        assert not (CYCLE/folder/'failure.json').exists()
    OUT.mkdir();torch.set_num_threads(2)
    patterns=['test_core23_spatial_readout.py','test_decoder_spatial_pose_readout.py','test_spatial_detection_readout.py','test_detection_readout.py','test_matched_point_loss.py','test_spatial_readout_pose.py','test_view_context_sampling.py','test_view_token_probe.py','test_independent_view_pose_probe.py','test_shared_token_pose_probe.py']
    suite=unittest.TestSuite()
    for p in patterns:suite.addTests(unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern=p))
    stream=io.StringIO();run=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite);(OUT/'tests.txt').write_text(stream.getvalue());assert run.wasSuccessful(),stream.getvalue()
    write(OUT/'tests.json',dict(tests=run.testsRun,failures=len(run.failures),errors=len(run.errors),skipped=len(run.skipped),patterns=patterns,scope='Focused new modules, not full repository regression.'))
    # Fixed first synthetic dual-instance RGB scene, not selected for improvement.
    baseline=json.loads((CYCLE/'live_view_full_context_full800_audit_v1/results.json').read_text())['reports']
    chosen=None
    for row in baseline:
        if row['task']['record']['kind']!='synthetic' or row['task']['variant']!='all':continue
        with np.load(ROOT/row['path']) as a:
            if int(a['box_valid'].sum(-1).max())>1:chosen=row;break
    assert chosen is not None
    record=chosen['task']['record'];anchor=chosen['task']['anchor_view'];v=record['views'].index(anchor);rgb=load_item(record)['batch']['images'][0,v]
    pw=600;ph=round(rgb.shape[-2]*pw/rgb.shape[-1]);im=Image.fromarray((rgb.permute(1,2,0).numpy()*255).clip(0,255).astype('uint8')).resize((pw,ph));canvas=Image.new('RGB',(pw*3,ph+42),(20,22,28));draw=ImageDraw.Draw(canvas);colors=['#ff6a6a','#66ff88','#55ccff','#ffff55','#ee77ff','#ffa544'];dual=[]
    for col,(arm,folder) in enumerate([('baseline','live_view_full_context_full800_audit_v1'),('paired','live_view_full_detector_paired800_audit_v1'),('full','live_view_full_detector_full800_audit_v1')]):
        rows=json.loads((CYCLE/folder/'results.json').read_text())['reports'];row=next(r for r in rows if r['task']==chosen['task']);canvas.paste(im,(col*pw,42));draw.text((col*pw+8,10),f'{arm} detector / SAME full800 pose / all views',fill='white')
        with np.load(ROOT/row['path']) as a:
            scores=1/(1+np.exp(-np.clip(a['instance_objectness_logits'][0,v],-80,80)));queries=np.flatnonzero(scores>=.5)
            for q in queries:
                cx,cy,bw,bh=a['instance_boxes_cxcywh'][0,v,q];lo=np.array([cx-bw/2,cy-bh/2])*[pw,ph]+[col*pw,42];hi=np.array([cx+bw/2,cy+bh/2])*[pw,ph]+[col*pw,42];draw.rectangle(tuple(lo)+tuple(hi),outline=colors[q],width=2);draw.text((lo[0],lo[1]-12),f'q{q}:{scores[q]:.2f}',fill=colors[q])
                for k in [0,14,15,16,17]:
                    x,y=a['instance_keypoint_xy'][0,v,q,k]*[pw,ph]+[col*pw,42];draw.ellipse((x-2,y-2,x+2,y+2),fill=colors[q]);draw.text((x+3,y),str(k+1),fill=colors[q])
            for m in range(a['gt_xy'].shape[2]):
                for k in [0,14,15,16,17]:
                    if not a['valid'][0,v,m,k]:continue
                    x,y=a['gt_xy'][0,v,m,k]*[pw,ph]+[col*pw,42];draw.line((x-4,y,x+4,y),fill='#fff06a',width=2);draw.line((x,y-4,x,y+4),fill='#fff06a',width=2)
            dual.append(dict(arm=arm,queries=queries.tolist(),task=chosen['task'],prediction=row['path'],all_views_metrics=row['detector']))
    canvas.save(OUT/'synthetic_dual_all_views.jpg',quality=96);write(OUT/'synthetic_dual_manifest.json',dual)
    selected=['readout_pilot_v1','readout_softbox_v1','readout_expanded_all_soft_box_extended4800_v1','readout_post4800_control_v1','readout_post4800_reliable_heatmap_v1','real_back_weight_control_v1','bottom_train_fit_v1','bottom_balanced_extended2400_v1','view_context_control_v1','view_order_control_v1','shared_token_pose_control_v1','full_context_pose_control_v1','detection_full_context_control_v1','detection_full_context_pose_audit_v1','live_view_independent_pose_audit_v1','live_view_full_detector_paired800_audit_v1','live_view_full_detector_full800_audit_v1']
    artifacts=[];inventory=[]
    for folder in sorted(CYCLE.iterdir()):
        if not folder.is_dir() or folder==OUT:continue
        entry=dict(path=folder.relative_to(ROOT).as_posix(),results=(folder/'results.json').exists(),failure=(folder/'failure.json').exists())
        if (folder/'status.json').exists():entry['status']=json.loads((folder/'status.json').read_text())
        inventory.append(entry)
    for folder in selected:
        for name in ['protocol.json','results.json','parity.json','candidate_detector_parity.json']:
            p=CYCLE/folder/name
            if p.exists():artifacts.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)))
    for p in [DOC,ROOT/'tools/validate_core23_live_views_20261004.py',ROOT/'tools/audit_detection_context_pose_20261005.py',ROOT/'tools/train_detection_context_control_20261005.py',ROOT/'tools/train_view_context_control_20261004.py',Path(__file__)]+[CYCLE/f'{folder}/{arm}_seed17_latest.pt' for folder,arm in [('full_context_pose_control_v1','full'),('detection_view_context_control_v1','mixed'),('detection_full_context_control_v1','paired'),('detection_full_context_control_v1','full')]]:
        artifacts.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)))
    write(OUT/'artifact_hashes.json',artifacts);write(OUT/'experiment_inventory.json',inventory);shutil.copy2(DOC,OUT/DOC.name);shutil.copy2(CYCLE/'full_detector_decision_v1.json',OUT/'full_detector_decision_v1.json')
    overview=['<!doctype html><meta charset="utf-8"><title>Core23 自动研究交付</title><style>body{font:17px system-ui;max-width:1150px;margin:35px auto;padding:0 18px;color:#20242c;background:#fafaf8}img{max-width:100%}td,th{padding:9px;border:1px solid #aaa}table{border-collapse:collapse}pre{white-space:pre-wrap;font:16px system-ui;background:#eee;padding:20px}a{color:#185b93}</style><h1>Core23 空间读出与可变视角：本轮交付</h1><p>提前收尾，不耗满16小时。空间读出有效，但尚无稳定兼顾真实底部、单/多视角和多鼠检测的正式替代模型。没有修改人工原始标签，也没有自动晋升 checkpoint。</p><h2>主要发现</h2><p>参数匹配空间读出显著优于全局坐标读出；视角顺序敏感的重要来源是首视角/其他视角 special-token 差异。修复顺序稳定不等于修复足部定位。最后的全视角检测头对照减少漏检，同时增加重复框，条件定位没有同步改善。</p><h2>最后严格对照：同一姿态头、实际RGB完整真实视角</h2><table><tr><th>检测头</th><th>命中/GT</th><th>框数</th><th>精确率</th><th>固定相机误差(px)</th><th>gated PCK32</th></tr>']
    for arm,rows in decision['rgb'].items():
        r=next(r for r in rows if r['kind']=='real' and r['variant']=='all');overview.append(f"<tr><td>{arm}</td><td>{r['tp']}/{r['gt']}</td><td>{r['predictions']}</td><td>{r['precision']:.3f}</td><td>{r['anchor_error']:.2f}</td><td>{r['gated_pck32']:.3f}</td></tr>")
    overview.append('</table><p>误差为768输入坐标、GT框关联辅助条件误差，非端到端准确率。gated PCK不惩罚每个重复查询，需和精确率同看。开发集非独立外部测试。原始几何保留不等于验证3D精度。</p><h2>直观看失败</h2><p>camA：完整视角出现重复框和躯干上的足点。左右两种相机顺序稳定，但仍可能共同预测错误。</p><img src="../review_full_detector_full_order_v1/00_real_order.jpg"><p>合成双鼠：固定第一组，不根据改善挑图；全部置信度≥0.5查询，无GT数量过滤/NMS。颜色不是跨视角身份。全视角合成属于训练未覆盖的4/8视角压力条件。</p><img src="synthetic_dual_all_views.jpg"><h2>检查与复现</h2>')
    overview.append(f'<p>本次针对新模块的 {run.testsRun} 项单元测试通过；不是整个仓库测试。paired/full各84张实际RGB候选检测/姿态一致性检查通过，阈值不放宽。<a href="tests.txt">测试日志</a> · <a href="artifact_hashes.json">代码/结果/权重SHA256</a> · <a href="experiment_inventory.json">全部实验和历史失败目录</a> · <a href="full_detector_decision_v1.json">三种子及RGB完整指标</a></p>')
    for name,label in [('full_detector_summary.html','最后检测头对照'),('full_context_pose_summary.html','完整视角姿态训练'),('review_detection_full_context_pose_v1/index.html','固定camA–G及合成样例'),('review_independent_pose_order_v1/index.html','独立单视角2D诊断'),('review_full_detector_paired_order_v1/index.html','paired检测头顺序检查'),('review_full_detector_full_order_v1/index.html','full检测头顺序检查')]:
        assert (DEST/name).exists();overview.append(f'<p><a href="../{name}">{label}</a></p>')
    overview.append('<h2>完整研究记录与下一阶段假设</h2><pre>'+html.escape(DOC.read_text())+'</pre>');(OUT/'index.html').write_text('\n'.join(overview),encoding='utf-8')
    assert not (DEST/OUT.name).exists();shutil.copytree(OUT,DEST/OUT.name)
    now=dt.datetime.now(dt.timezone.utc).isoformat();next_action='Bounded cycle experiments COMPLETE; focused tests and actualRGB parity pass; reports in cycle_handoff. No more training. Inspect newlygenerated synthetic_dual_all_views.jpg and confirm final visual notes. Pause existing pymousestudio automation via official app tool (OpenAI Docs already read) preserving all fields. Then record paused completion state, verify no ownGPU jobs, and deliver concise Windows cycle_handoff/index.html link. No model promotion or label edits. Do not rerun finalizer: output is intentionally immutable.'
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='cycle_handoff_ready_pending_visual_and_automation_pause',active_job=None,active_pid=None,updated_utc=now,next=next_action,handoff=str((OUT/'index.html').relative_to(ROOT)));state['completed'].append('final_focused_tests_rgb_parity_and_evidence_handoff');write(CYCLE/'state.json',state);plan['next_command']=next_action;write(CYCLE/'plan.json',plan)
    print(json.dumps(dict(tests=run.testsRun,output=str(OUT),windows=str(DEST/OUT.name),utc=now)))

if __name__=='__main__':main()
