"""Create isolated realism project and durable 10-hour experiment plan."""
import datetime as dt, hashlib, json, shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/pymousestudio_realism_cycle_20261002'
PROJECT=ROOT/'BlenderPyMouseProjects/Cylinder/PyMouseStudio-RealCapture-Replica-20261002'
def main():
    assert not OUT.exists() and not PROJECT.exists()
    OUT.mkdir();PROJECT.mkdir()
    source=ROOT/'BlenderPyMouseProjects/Cylinder/PyMouseStudio-BrownFur-ClimbWall.blend'
    target=PROJECT/'PyMouseStudio-RealCapture-Replica-v000.blend'
    shutil.copy2(source,target)
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert digest(source)==digest(target)
    plan=dict(start_utc='2026-10-02T20:59:16+00:00',deadline_utc='2026-10-03T06:59:16+00:00',heartbeat_minutes=15,
        objective='在现有PyMouseStudio Cylinder工程基础上独立制作真实采集风格与姿态的Blender复刻工程；逐轮渲染并视觉检查，保留可复现最佳版本。不是批量训练数据发布。',
        source_blend=str(source),source_sha256=digest(source),project_directory=str(PROJECT),initial_blend=str(target),
        blender_windows='C:/Program Files/Blender Foundation/Blender 5.1/blender.exe',
        reference_groups=[dict(session=s,frame=f) for s,f in [('FR2',649),('FR2',1104),('FR2',1429),('AZR2L2',129)]],
        reference_root='dataset/PyMouseStudio/annotation_returns/20260930_full_v2/real_images/real_development',
        schedule=[dict(hours='0–1',work='审计场景、依赖、单位、相机、rig和GPU；建立固定参考板与未经调整的多视角Cycles基线。'),
            dict(hours='1–3',work='匹配透明圆筒/平台/头部装置、背景与相机、曝光和灰度采集响应；逐项A/B渲染，不凭直觉一次改全部。'),
            dict(hours='3–5',work='黑鼠短毛分布、毛向/长度/粗细、耳爪尾皮肤、粗糙度与散射；近景和全景同时对照。'),
            dict(hours='5–7',work='至少3个不同代表姿态：体轴/头颈/腹背高度、前后/左右足、接触与尾巴几何；多视角联合核对，不照搬错误或不确定3D标签。'),
            dict(hours='7–9',work='针对持续瓶颈查官方/论文/项目，试可授权现成PBR/道具、局部建模或生成；必要时独立图像域迁移小试，不覆盖物理RGB。'),
            dict(hours='9–10',work='复查最佳工程依赖与可重开性；导出原分辨率多视角和接触表、必要的短动画；整理成功失败、许可、未解决问题，截止停止并暂停自动化。')],
        review_protocol=['每轮说明差异、假设、改动、渲染参数、实际视觉观察和保留/回退理由；保存到版本独立目录。',
            '首轮4–8个参考视角含bottom与side；优化用3组开发帧，第四组FR2f1429作为本轮风格泛化检查不调参。不得使用GR2/UR2/UR3锁定留出。',
            '预览可降分辨率/采样但最终1080x1200与真实矫正图一致；若原相机比例不同先解释而非拉伸。最终至少底部+3个侧面，固定帧/相机A-B对照。',
            '区分姿态/轮廓、接触、毛发/皮肤、器材背景、成像响应、跨视角一致性；数值只能辅助，必须查看实际渲染图。',
            '尚未验证几何和语义一致性的真实3D重建只作参考，不能当GT；不移动现有真实标注。',
            '风格迁移输出单列experimental_appearance，不自动附原3D标签；检查足尖/耳/尾/背部锚点及跨视角漂移，禁止仅凭好看接受。'],
        guardrails=['原工程、dataset、相机标定、返修文件、训练模型保持不变；仅新工程及本周期输出可写。',
            '后台Blender独立进程；不得打开覆盖用户GUI未保存场景。GPU需实测OptiX或CUDA可用；不静默CPU渲染。一次最多一个GPU重任务，检查现有进程及共享锁。',
            '每个工作进程保存PID/命令/日志/局部超时，全局deadline前结束；不重复启动。只停止自己验证过的PID。',
            '免费素材检查来源、许可证与重分发条件并记录；不付费、不上传私有图片/模型到第三方服务。大于1GB单模型下载或总新增超过5GB须先询问。可先用本机现有模型。',
            '新工程每版本保存，依赖尽量打包或复制为相对路径，GUI毛发默认隐藏但最终渲染启用；不要盲目打包所有无用资源。',
            '到期不新开实验，不自动延时；收尾、展示最佳结果、报告不足并暂停本周期automation。若瓶颈先尝试安全替代，不能因第一次失败草率结束。'],
        notification='仅重要进展、完成、失败或需用户决策时通知；无变化保持安静。')
    (OUT/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'state.json').write_text(json.dumps(dict(stage='project_cloned_awaiting_baseline_audit',active_pid=None,best_blend=str(target),source_unchanged=True,updated_utc=dt.datetime.now(dt.timezone.utc).isoformat()),indent=2))
    refs=OUT/'references';refs.mkdir()
    for g in plan['reference_groups']:
        src=ROOT/plan['reference_root']/g['session']/f"frame_{g['frame']:06d}";dst=refs/g['session']/src.name
        dst.mkdir(parents=True)
        for p in src.glob('*.png'):shutil.copy2(p,dst/p.name)
    print(json.dumps(dict(project=str(target),plan=str(OUT/'plan.json'),deadline=plan['deadline_utc']),indent=2))
if __name__=='__main__':main()
