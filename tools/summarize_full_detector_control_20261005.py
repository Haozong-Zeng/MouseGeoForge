"""Record bounded detector-view-count control and explicit final handoff gates."""
import datetime as dt,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CYCLE=ROOT/'output/core23_spatial_cycle_20261004'
DEST=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-Core23-Spatial-20261004')

def main():
    cached=json.loads((CYCLE/'detection_full_context_pose_audit_v1/results.json').read_text())
    rgb={};parity={}
    for arm,folder in [('baseline','live_view_full_context_full800_audit_v1'),('paired','live_view_full_detector_paired800_audit_v1'),('full','live_view_full_detector_full800_audit_v1')]:
        path=CYCLE/folder/'results.json'
        if not path.exists():continue
        raw=json.loads(path.read_text());assert raw['parity_passed'] and raw['single_rgb_parity_cases']==12 and raw['shared_rgb_parity_cases']==24 and raw['full_rgb_parity_cases']==48
        parity[arm]={k:v for k,v in raw.items() if k!='reports'}
        if arm!='baseline':
            check=json.loads((path.parent/'candidate_detector_parity.json').read_text());assert check['cases']==84 and check['max_delta']<2e-5;parity[arm]['candidate_detector']=check
        groups=[]
        for kind in ['real','synthetic']:
            for variant in ['single','paired','paired_reversed','all','all_reversed']:
                rows=[r for r in raw['reports'] if r['task']['record']['kind']==kind and r['task']['variant']==variant]
                n=sum(r['metrics']['spatial/anchor']['n'] for r in rows);points=sum(r['gated_pose']['total_points'] for r in rows)
                tp=sum(r['detector']['tp'] for r in rows);pred=sum(r['detector']['predictions'] for r in rows);gt=sum(r['detector']['gt'] for r in rows)
                groups.append(dict(kind=kind,variant=variant,tp=tp,gt=gt,predictions=pred,precision=tp/max(pred,1),recall=tp/max(gt,1),anchor_error=sum(r['metrics']['spatial/anchor']['n']*r['metrics']['spatial/anchor']['mean'] for r in rows)/max(n,1),gated_pck32=sum(r['gated_pose']['pck32_count'] for r in rows)/max(points,1)))
        rgb[arm]=groups
    next_action=(
        'Final detector800x3seeds paired/full TRAIN and frozenfull800pose cacheaudit COMPLETE. Gallery review_detection_full_context_pose_v1 COMPLETE, camA/camD reviewed: full adds duplicatecamA query, hindpaw remainsontorso; camD forepaw nearhead. No promotion. '
        'ActualRGBpaired-detector audit LAUNCHED session43670 PID3761988: validate_core23_live_views_20261004.py --full-context-detector paired. CheckactualPID/status/failure andGPUlock; no duplicate. '
        'After complete run same script --full-context-detector full ONCE, output live_view_full_detector_full800_audit_v1. Both flags IMPLEMENTED. All same full800pose, same60tasks; baseline comparator live_view_full_context_full800_audit_v1. '
        'Original192cached/defaultoff gate deliberately baseline mixed800detector; candidate loads AFTERthatgate, then candidatepose/detector goldens required for12single/24sharedpair/48fullreal. candidate_detector_parity.json distinguishes correctscope. Do not misuse original-token baselineparity as candidategolden. '
        'Run review_view_order_audit_20261005.py --full-context-detector paired/full ONCE wheneachcomplete, outputs review_full_detector_{arm}_order_v1, then inspectrawfixedcamA/syntheticdual. Flags IMPLEMENTED. Rerun summarize_full_detector_control_20261005.py for finalRGB comparison. '
        'No further similar head-only training planned. If RGB confirms coverage/precision tradeoff, consolidate entirecycle finalhonesthandoff with experimentindex, success/failures, fixeddata/protocol, remainingarchitecture/data limits, no productionpromotion. Read openai-docs skill and existingautomation before PAUSING pymousestudio officialtool, preservefields. Max16h not requiredtofill. Stoptraining10:18:46UTC,hardstop10:48:46UTC. Originalannotations/lockedGR2UR2UR3 untouched.')
    if len(rgb)==3:
        next_action=('Final detectorcontext train/cache/RGB COMPLETE. Do NOT retrain or rerun validators. Check/generate missing review_full_detector_paired_order_v1 andfull_order_v1 using review_view_order_audit_20261005.py --full-context-detector paired/full ONCE. InspectfixedcamA andsyntheticdual; note allconfidencequeries. Review full_detector_decision_v1.json for baseline/paired/full SAMEfull800pose and84candidateRGBgoldens. Then consolidate entirecycle FINALhandoff, indexHTML in Downloads and repo reproduciblesummary, explain successes andfailures/limits, avoid modelpromotion andmorehead-onlyprobes. Readopenai-docs skill before officialautomation PAUSE existingpymousestudio preservingfields. Finish early ratherthanfill16h. Originaldata/lockedholdouts/productiondefaults unchanged. Harddeadline10:48:46UTC.')
    result=dict(updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),cached=cached,rgb=rgb,parity=parity,rgb_complete=len(rgb)==3,visual_cache=['camA full: full-context detector adds overlapping q3 beside q4; hindpaw predictions still on torso; does not solve anatomy.','camD full: forepaw prediction near head remains.'],no_production_promotion=True,next=next_action)
    path=CYCLE/'full_detector_decision_v1.json';path.write_text(json.dumps(result,indent=2)+'\n');shutil.copy2(path,DEST/path.name)
    page=['<!doctype html><meta charset="utf-8"><title>检测头多视角对照</title><style>body{font:17px sans-serif;max-width:1100px;margin:40px auto}td,th{border:1px solid #bbb;padding:8px}table{border-collapse:collapse}</style><h1>检测头多视角对照：召回提升，但重复框增加</h1><p>固定同一完整视角姿态头，仅改变检测头。baseline为旧mixed800；paired/full为同初始化、同800步、3种子的继续训练。两组50%单视角；full组增加25%完整合格视角条件。未改原始标注或生产默认模型。</p><h2>缓存开发验证：完整视角真实数据</h2><table><tr><th>种子/组</th><th>命中/48</th><th>预测框数</th><th>精确率</th><th>关键点误差(px)</th><th>gated PCK32</th></tr>']
    for row in cached['results']:
        if row['context']!='full':continue
        d=row['detector']['real'];m=row['metrics']['real/reliable'];page.append(f"<tr><td>{row['seed']}/{row['arm']}</td><td>{d['tp']}/48</td><td>{d['predictions']}</td><td>{d['precision']:.3f}</td><td>{m['error_mean']:.2f}</td><td>{m['gated_pck32']:.3f}</td></tr>")
    page.append('</table><p>关键点误差使用GT框辅助关联，非端到端准确率；gated PCK计入漏检但不惩罚所有重复关键点，必须同时看预测框数和精确率。误差单位为模型输入像素。FR2/AZR2L2开发集已多次使用，非独立测试。</p><p><a href="review_detection_full_context_pose_v1/index.html">固定样例：单/双/完整视角原图</a> | <a href="full_detector_decision_v1.json">完整数据及后续状态</a></p>')
    if rgb:
        page.append('<h2>实际RGB：固定seed17，姿态头一致</h2><table><tr><th>组/域/视角</th><th>TP/GT</th><th>框数</th><th>锚定相机误差</th><th>gated PCK32</th></tr>')
        for arm,rows in rgb.items():
            for row in rows:
                if row['variant'].endswith('reversed'):continue
                page.append(f"<tr><td>{arm}/{row['kind']}/{row['variant']}</td><td>{row['tp']}/{row['gt']}</td><td>{row['predictions']}</td><td>{row['anchor_error']:.2f}</td><td>{row['gated_pck32']:.3f}</td></tr>")
        page.append('</table><p>合成全视角4/8仍属未训练压力条件。共享token的2D分支保留原始多视角几何，但并不证明3D准确；额外encoder前向不是算力公平比较。辅助pose字段尚未统一，不可直接用于联合训练。</p>')
    (DEST/'full_detector_summary.html').write_text('\n'.join(page),encoding='utf-8')
    plan=json.loads((CYCLE/'plan.json').read_text());plan['next_command']=next_action;(CYCLE/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    state=json.loads((CYCLE/'state.json').read_text());state.update(next=next_action,updated_utc=result['updated_utc']);entry='full_detector_cache_3seed_and_raw_visual_review'
    if entry not in state['completed']:state['completed'].append(entry)
    (CYCLE/'state.json').write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(dict(rgb_complete=result['rgb_complete'],rgb=rgb),indent=2))

if __name__=='__main__':main()
