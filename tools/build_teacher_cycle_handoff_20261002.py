"""Read-only source integrity audit and derived HTML/JSON cycle handoff; no new experiments."""
import datetime as dt
import fcntl
import hashlib
import html
import json
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from run_teacher_cycle_dlc_20261001 import ROOT, OUT, write_state


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    started = time.time()
    plan = json.loads((OUT/'plan.json').read_text())
    deadline = min(started+600, dt.datetime.fromisoformat(plan['deadline_utc']).timestamp())
    dest = OUT/'cycle_handoff'; dest.mkdir(exist_ok=False)
    lock = (OUT/'gpu_job.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    gpu = subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
    assert not gpu, 'GPU owner exists: inspect before recording idle'
    m = json.loads((OUT/'dlc_dataset_manifest.json').read_text())
    for i,e in enumerate(m['entries']):
        assert time.time()<deadline
        assert e['session'] not in plan['locked_holdout_sessions']
        assert digest(ROOT/e['source']) == e['source_sha256'], e['source']
    base = ROOT/plan['source_return']
    ingestion = json.loads((base/'ingestion_manifest.json').read_text())
    groups = {(g['session'],g['frame']):g for g in ingestion['frame_groups']}
    with np.load(OUT/'dlc_baseline_review/predictions_source_pixels.npz') as z:baseline_gt=z['gt'].copy()
    cache = {}; max_difference=0.
    for i,e in enumerate(m['entries']):
        key = (e['session'],e['frame']); g=groups[key]
        if key not in cache:
            with np.load(base/g['normalized_file']) as z:cache[key]=z['uv'][:,0,np.array(m['ids_1based'])-1].copy()
        source=cache[key][g['cameras'].index(e['camera'])]
        assert np.array_equal(np.isnan(source),np.isnan(baseline_gt[i]))
        if np.isfinite(source).any():max_difference=max(max_difference,float(np.nanmax(np.abs(source-baseline_gt[i]))))
        assert np.allclose(source,baseline_gt[i],equal_nan=True,atol=.001,rtol=0)
    integrity=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),source_png_sha256_matches=len(m['entries']),
        source_label_groups_compared=len(cache),approved_label_points_nan_masks_match=True,max_label_coordinate_difference_px=max_difference,
        gpu_jobs_at_check=gpu,shared_gpu_lock_free=True,locked_holdout_images_read=False,
        note='Image hashes compared against cycle-startmanifest; approved7keypoint coordinates compared to cached startGT. No baseline full-calibration-file hash exists, so no invented cryptographic calibration claim.')
    context = json.loads((OUT/'lp_context_seed17_kl/summary.json').read_text())
    matched = json.loads((OUT/'lp_context_matched_ablation_v1/summary.json').read_text())
    clips = json.loads((OUT/'context_clip_geometry_review_v1/summary.json').read_text())
    methods=[('DLC ResNet50（最佳70轮）',context['models']['DLC70']),('静态 LP 单模型 seed17',context['models']['static_seed17']),
        ('静态 LP 三模型中位数',context['models']['static_ensemble_median']),('同架构时序关闭',matched['models']['off_selected']),('同架构时序开启',matched['models']['on_selected'])]
    ledger=[
        ('DLC 基线及延长训练','完成','30轮后延长45轮；最好模型在70轮。延长有收益，但仍有四足互换，原始高置信度不足以筛出所有错误。'),
        ('Lightning Pose 单帧教师','完成','最初 MSE 热图训练出现鼻/背部退化；排查输入和缺失点梯度后改用 KL，并完成17/31/73三种子。静态集成是有效基线。'),
        ('逐帧 / Anipose / 官方 EKS','完成小规模诊断','时序后处理未稳定解决共享语义错误；只看平滑或自重投影会高估效果。EKS与Anipose均未作为可发布真值来源。'),
        ('不确定性、鲁棒几何与多峰四足联合选择','完成诊断','可改善部分覆盖和定位，但严格门控会大量丢点，碰撞否决误删正常样本，多模型仍可共享高置信错误。未发布强3D。'),
        ('输入编码一致性','已修复并验证','发现已矫正H264视频与原人工PNG不逐像素一致。重新从原视频按既定K/D仅矫正一次，输出PNG/FFV1；不得对已矫正视频二次矫正。当前ON/OFF短段168项预测对齐检查均为0px。'),
        ('后处理时序联合状态','完成阴性结果','精确输入下增加部分接受点，但共同接受点的预测不变，没有修正持续性的四足身份错误。'),
        ('原生 LP MHCRNN 时序开启/关闭','完成单种子匹配对照','同架构、种子、batch4、30轮、7830步、训练划分和标签。ON用相邻5帧，OFF从训练起重复中心帧。ON降低长尾大错，但并非全面优于静态3模型集成。'),
        ('连续视频与留一视角','完成','4段×204个中心帧，24个相机片段；源52fps，导出半速26fps视频。真正屏蔽被留视角的预测后再拟合；剩余绝对误差仍过大。'),
        ('YOLO-Pose 教师','本周期未实施','本周期集中验证实际DLC/LP和时序/几何假设，不声称已经完成YOLO对照。'),
        ('尾巴分割—3D曲线—弧长取点','本周期未实施','与身体点监督分离，尾巴8–14和腹面23未加入新的强3D。'),
        ('Foundation model 下游收益','未验证','没有把本周期预测并入正式训练，也未完成独立下游消融。教师误差降低不等于foundation model训练一定获益。')]
    policy=dict(manual_2d='保留原坐标、来源、缺失掩码；待复核24组和SEM01继续隔离。缺失不是已证实不可见。',
        pseudo_2d='仅作为诊断或待人工确认候选；原生confidence不是校准概率。不得自动覆盖人工点。',
        pseudo_3d='本周期不发布强3D；单视角点、腹面23、尾部、插补点不得升级为强GT。',
        provenance_required=['source_session/frame/camera','coordinate_system/units','rectification_and_affine','teacher_checkpoint/seed','manual_vs_pseudo','observed_vs_imputed','source_view_support','LOO_error_or_quality_mask','exclusion_reason','split/animal_identity_if_known'])
    next_steps=[
        '优先复核跨视角左右/前后爪语义，以及新增SEM01（AZR2L2 frame584）。先修语义冲突，再增加同视频训练轮数。',
        '人工补标少量连续短段：所有同步视角、动作快速变化、底视/侧视冲突、遮挡转显露。保持分组独立，不能只标模型最自信帧；另留不参与选择的评估段。',
        '补充显式可见/遮挡/画外/未审核状态；对腹面不可见的背部点保留unknown或受控弱监督，不能把未标注当负样本。',
        '后续预算允许时用更多种子重复匹配时序对照，并按动物/独立视频评估。锁定GR2/UR2/UR3本周期未打开，不继续在开发集上调阈值。',
        '只有经过人工2D、跨视角一致性与质量掩码验证的候选才可设计下游伪标签试验；人标/伪标/插补分层加权，并与不用伪标签的基线比较。',
        '尾部独立推进分割/中心线/多视角3D曲线和弧长采样，保留原人工尾点作为约束，不把身体刚性时序假设直接套给尾巴。']
    links=[('匹配时序开/关：指标与18组图','../lp_context_matched_ablation_v1/index.html'),
        ('连续四段视频（OFF左/ON右）','../context_clip_geometry_review_v1/index.html'),
        ('同一时序模型输入替换：24组图及学习曲线','../lp_context_seed17_kl/index.html')]
    report=dict(prepared_utc=dt.datetime.now(dt.timezone.utc).isoformat(),cycle_start_utc=plan['start_utc'],cycle_deadline_utc=plan['deadline_utc'],
        status='prepared_before_deadline; no further experiments scheduled; finalheartbeat must verifyidle and pauseautomation',
        integrity=integrity,common_cohort='288开发图/1361人工关键点/770四足点；仅两个反复使用的开发视频，非独立测试。SEM01排除；历史293图指标不可直接混比较。',
        methods=[dict(name=n,metrics=v) for n,v in methods],matched=matched,continuous=clips,ledger=ledger,
        training_eligibility=policy,next_steps=next_steps,pseudo_gt_released=False,
        experiment_root=str(OUT),visual_root='C:/Users/55395/Downloads/MouseGeoForge-TeacherTemporal-20261001',review_links=links)
    (dest/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    (dest/'integrity_audit.json').write_text(json.dumps(integrity,indent=2))
    def esc(s):return html.escape(str(s))
    parts=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>MouseGeoForge 十小时教师试验交付</title>',
        '<style>body{font:16px/1.65 system-ui;margin:30px auto;max-width:1150px;padding:0 20px;color:#18212e}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:9px;text-align:left}th{background:#eef2f6}h2{margin-top:34px}.note{background:#fff4d9;padding:15px}a{color:#075bb5}</style>',
        '<h1>MouseGeoForge：真实视频教师与时序伪标注试验</h1>',
        '<p class="note">结论：时序有正向证据，但仍有四足身份错误和跨视角不一致。本周期没有发布强3D/伪GT，也没有证明下游foundation model收益。停止追加训练，优先补足语义复核和连续短段评估。</p>',
        f'<p>准备时间：{esc(report["prepared_utc"])}；周期截止：{esc(plan["deadline_utc"])}。此页面是交付汇总，不代表定时器已暂停。</p>',
        '<h2>检查入口</h2><ul>'+''.join(f'<li><a href="{esc(url)}">{esc(label)}</a></li>' for label,url in links)+'</ul>',
        '<h2>公平可比的2D结果</h2><p>'+esc(report['common_cohort'])+' 所有误差为原图1080×1200像素。</p>',
        '<table><tr><th>模型</th><th>平均误差</th><th>中位误差</th><th>P90</th><th>四足平均</th></tr>']
    for name,v in methods:parts.append(f'<tr><td>{esc(name)}</td><td>{v["all"]["mean"]:.2f}</td><td>{v["all"]["median"]:.2f}</td><td>{v["all"]["p90"]:.2f}</td><td>{v["paws"]["mean"]:.2f}</td></tr>')
    parts += ['</table><p>匹配OFF→ON：四足&gt;100px大错66→47/770。收益主要来自长尾；单种子、两开发视频结果不能证明总体泛化。ON没有全面超过三模型静态集成。</p>',
        '<h2>连续短段与真正留一视角</h2><p>同384个人工2D观测，均未筛除：原始2D平均37.04→28.65、P90 123.08→55.36；留一视角投影平均69.23→62.15、P90 201.77→145.35。后者绝对误差仍不足以支撑强3D标签。</p>',
        '<p>两组共用既有鲁棒几何求解器、固定均匀4px尺度；不是官方Anipose/EKS，也没有把单模型置信度冒充集成方差。无密集动作真值，因此跳变减少不能证明幅度/延迟准确。</p>',
        '<h2>尝试、失败和未完成项</h2><table><tr><th>路线</th><th>状态</th><th>实测结论/限制</th></tr>']
    for row in ledger:parts.append('<tr>'+''.join('<td>'+esc(cell)+'</td>' for cell in row)+'</tr>')
    parts += ['</table><h2>训练数据准入</h2><ul>'+''.join('<li>'+esc(policy[k])+'</li>' for k in ['manual_2d','pseudo_2d','pseudo_3d'])+'</ul>',
        '<p>候选必须携带来源、相机、帧号、坐标/单位、校正/缩放、模型版本、人工/伪标、观测/插补、视角支持、质量掩码、排除原因及数据划分。禁止只保存一组坐标丢失可信度来源。</p>',
        '<h2>下一轮优先级（本周期不自动执行）</h2><ol>'+''.join('<li>'+esc(s)+'</li>' for s in next_steps)+'</ol>',
        '<h2>完整性与复现</h2><p>'+esc(f"重新验证{integrity['source_png_sha256_matches']}张原PNG的SHA256全部匹配；{integrity['source_label_groups_compared']}组人工标签中本周期7点坐标/NaN掩码与起始缓存一致，最大差{max_difference:.8f}px。GPU无任务，共享GPU锁空闲。未打开锁定留出图像。")+'</p>',
        '<p>实验、配置、checkpoint、JSON日志：<code>'+esc(str(OUT))+'</code>。校准文件未保存起始整体hash，不能声称做过全文件哈希比对；本周期代码不写原校准。</p>',
        '<p><a href="summary.json">完整机器可读汇总</a> · <a href="integrity_audit.json">完整性检查</a></p></html>']
    (dest/'index.html').write_text('\n'.join(parts))
    export=Path('/mnt/c/Users/55395/Downloads/MouseGeoForge-TeacherTemporal-20261001/cycle_handoff')
    assert not export.exists();shutil.copytree(dest,export)
    write_state(stage='cycle_handoff_prepared_waiting_deadline',pid=None,active_run=None,mode='idle_between_heartbeats',
        latest_result=str(dest/'summary.json'),handoff_windows=str(export),integrity_audit=str(dest/'integrity_audit.json'),
        next_action='At/after07:21:52UTC verify noownedjobs, recordcyclecloseout, pause mousegeoforge automation via app tool. No new experiments.',pseudo_gt_released=False)
    print('HANDOFF_READY',export,flush=True);print(json.dumps(integrity),flush=True)


if __name__=='__main__':main()
