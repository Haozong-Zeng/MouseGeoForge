"""Record the bounded reference-order result without promoting a model."""
import datetime as dt,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CYCLE=ROOT/'output/core23_spatial_cycle_20261004'
DEST=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-Core23-Spatial-20261004')

def main():
    comparison={}
    for arm in ['fixed','random_ref']:
        path=CYCLE/f'review_reference_{arm}_order_v1/results.json'
        report=json.loads(path.read_text())
        live=json.loads((CYCLE/f'live_view_reference_{arm}800_audit_v1/results.json').read_text())
        assert live['parity_passed'] and live['single_rgb_parity_cases']==12 and not live['shared_token_probe']
        shifts={}
        for kind in ['real','synthetic']:
            for variant in ['paired','all']:
                vals=[r['mean_point_shift_px'] for r in report['changes'] if r['kind']==kind and r['variant']==variant and r['mean_point_shift_px'] is not None]
                shifts[kind+'/'+variant]=sum(vals)/len(vals)
        comparison[arm]=dict(summary=report['summary'],mean_per_view_order_shift=shifts,max_single_rgb_delta=live['max_single_rgb_delta'])
        report['replay_scope']='Changed pose weights: old checkpoint replay intentionally not run. Cached-to-live parity passed in corresponding validator.'
        report['max_original_order_delta']=None
        if arm=='random_ref':
            report.update(visual_review_pending=False,visual_review=['00_real_order.jpg: bottom camA duplicate queries persist, front/hind points still cluster on torso; visible changes under order reversal.','03_synthetic_order.jpg: full-view reversed box becomes shallow and feet shift despite identical physical image.'])
        path.write_text(json.dumps(report,indent=2)+'\n');shutil.copy2(path,DEST/path.parent.name/path.name)
    next_action=('Detector-only reference control launched: prepare_single_detection_context_20261005.py --reversed then train_detection_context_control_20261005.py --order-control; inspect actual PID/status and do not duplicate. Output reversed_detection_context_v1 and detection_reference_control_v1. Same800steps seeds17/29/43, fixed versus random_ref, initial mixed800 detector, both50percent single, treatment reverseshalfpaired. Same view_order_control schedules and original pretrained tokens. Extraction checks originalMLP and mixed800ROI goldens. Only ROI detector trained. After complete, extend audit_detection_context_pose_20261005.py with an explicit order-control mode: frozen view_order_control_v1/random_ref800 pose; baseline mixed800detector and new fixed/random_ref detectors, paired/single/reversed contexts, all138queries. Reversed baseline goldens in reversed_detection_context_v1/golden.pt, original/single baseline in detection_view_context_control_v1/predictions.npz. Pose baseline parity from view_order_control_v1/predictions.npz random_ref keys. Report detector precision/recall/counts and conditional/samequery/gated pose separately, cameras and dualsynthetic. Then add explicit reference-detector flag to fullRGB validator for same60tasks; original model parameters/defaults must remain unchanged. If no clear benefit stop repeated head-only order augmentation; consider a matched shared-special-token adaptation experiment or independent-view pose branch, with geometry path retained and explicit1/2/allview retention gates. Do not conflate symmetry with accuracy. No GTbestorder selection. Training stop10:18:46UTC,hardstop10:48:46UTC unchanged.')
    decision=dict(updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),conclusion='Pose-only reference-order augmentation gives modest cached gains but does not solve order sensitivity or bottom anatomy. Do not promote or blindly extend.',comparison=comparison,visual_cached='00_real_camA_reversed.jpg: two high-confidence boxes on one mouse in both arms; paw positions still ambiguous. No automatic label changes.',interpretation='Detector was frozen and identical between pose arms, so all TP/count differences between original/reversed contexts remain. Separate detector adaptation next rather than attributing missed detections to pose decoder.',limitations='Small development validation, not pristine external test. Inputpixel errors conditional on GTbox association. GatedPCK includes misses but not duplicate point false positives; detector precision reported separately. Three cached seeds, fullRGB seed17 only. Locked sessions untouched.',production_changed=False,next=next_action)
    (CYCLE/'reference_control_decision_v1.json').write_text(json.dumps(decision,indent=2)+'\n');shutil.copy2(CYCLE/'reference_control_decision_v1.json',DEST/'reference_control_decision_v1.json')
    page=['<!doctype html><meta charset="utf-8"><title>参考视角随机化对照</title><style>body{font:17px sans-serif;max-width:1100px;margin:40px auto}td,th{padding:8px;border:1px solid #bbb}table{border-collapse:collapse}</style><h1>参考视角随机化：小幅改善，尚未解决问题</h1><p>固定顺序与随机参考视角各800步，三种子；保持单视角/双视角数量、图像、损失一致，检测头冻结。没有修改正式模型或标注。</p><p>完整RGB复核为seed17；以下误差为输入分辨率像素、同一物理相机上的条件误差，不等于端到端精度。开发验证集并非独立外部测试。</p><table><tr><th>数据/上下文</th><th>固定顺序误差</th><th>随机参考误差</th><th>固定/随机 gated PCK32</th></tr>']
    for a,b in zip(comparison['fixed']['summary'],comparison['random_ref']['summary']):
        assert (a['kind'],a['variant'])==(b['kind'],b['variant'])
        page.append(f"<tr><td>{a['kind']}/{a['variant']}</td><td>{a['anchor_error']:.2f}</td><td>{b['anchor_error']:.2f}</td><td>{a['gated_pck32']:.3f} / {b['gated_pck32']:.3f}</td></tr>")
    page.append('</table><p>底部足点仍会聚到躯干上，重复框仍存在；交换顺序后的多视角结果仍明显变化。下一步仅调整检测头，检验框与漏检对下游姿态的影响，不盲目延长当前姿态头训练。</p><p><a href="review_reference_fixed_order_v1/index.html">固定顺序：完整RGB可视化</a> | <a href="review_reference_random_ref_order_v1/index.html">随机参考：完整RGB可视化</a> | <a href="review_view_order_control_v1/index.html">三上下文缓存样例</a></p>')
    (DEST/'reference_control_summary.html').write_text('\n'.join(page),encoding='utf-8')
    plan=json.loads((CYCLE/'plan.json').read_text());plan['next_command']=next_action;(CYCLE/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    state=json.loads((CYCLE/'state.json').read_text());state.update(next=next_action,updated_utc=decision['updated_utc'])
    item='pose_reference_control_rgb_and_visual_review_no_promotion'
    if item not in state['completed']:state['completed'].append(item)
    (CYCLE/'state.json').write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps(comparison,indent=2))

if __name__=='__main__':main()
