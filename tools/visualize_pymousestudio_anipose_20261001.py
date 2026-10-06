"""Before/after review and cautious tail-curve design; no accepted-label changes."""
import html
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_pymousestudio_manual_multiview import projection
from audit_pymousestudio_anipose_20261001 import OUT,BASE,ROOT,BODY,ACTIVE

FONT='/mnt/c/Windows/Fonts/msyh.ttc'
def font(n):return ImageFont.truetype(FONT,n)


def overlay(raw,uv,proj,color):
    im=raw.copy();d=ImageDraw.Draw(im)
    for k in sorted(ACTIVE|{22}):
        if not np.isfinite(uv[k]).all():continue
        x,y=uv[k];green='#ffd352' if k==22 else '#6fff70'
        d.ellipse((x-5,y-5,x+5,y+5),outline=green,width=2)
        d.text((x+7,y-21),str(k+1),font=font(21),fill=green,stroke_fill='black',stroke_width=1)
        if k==22 or not np.isfinite(proj[k]).all():continue
        u,v=proj[k]
        d.line((x,y,u,v),fill=color,width=1)
        d.line((u-5,v-5,u+5,v+5),fill=color,width=2)
        d.line((u-5,v+5,u+5,v-5),fill=color,width=2)
    return im


def main():
    summary=json.loads((OUT/'summary.json').read_text());groups=json.loads((OUT/'frame_manifest.json').read_text())
    obs=json.loads((OUT/'observation_metrics.json').read_text());config=json.loads((OUT/'config.json').read_text())
    names=['nose','dorsal thorax','dorsal lumbar','left forepaw','right forepaw','left hindpaw','right hindpaw']
    keys=[1,6,7,15,16,17,18]
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
    x=np.arange(len(keys));width=.37
    a=[summary['keypoints'][str(k)]['paired_loo']['before']['median'] for k in keys]
    b=[summary['keypoints'][str(k)]['paired_loo']['after']['median'] for k in keys]
    ax.bar(x-width/2,a,width,label='Previous LS');ax.bar(x+width/2,b,width,label='Aniposelib robust')
    ax.set_xticks(x,names,rotation=20,ha='right');ax.set_ylabel('Leave-one-view-out median error (px)')
    ax.set_title('Same observations | 276 groups | 24 pending-review groups excluded\nFixed cameras; no temporal/length smoothing; tail and abdomen unchanged')
    ax.legend();fig.savefig(OUT/'paired_loo_by_keypoint.png',dpi=160);plt.close(fig)
    paired=defaultdict(list)
    for r in obs:
        if r['active'] and r['split']=='real_development' and r['baseline_ls_loo'] is not None and r['anipose_robust_loo'] is not None:
            paired[(r['session'],r['frame'])].append(r['anipose_robust_loo']-r['baseline_ls_loo'])
    ranked=sorted((float(np.median(v)),k) for k,v in paired.items() if len(v)>=20)
    chosen=[]
    for session in sorted({g['session'] for g in groups}):
        g=min((g for g in groups if g['session']==session),key=lambda g:g['frame'])
        chosen.append((session,g['frame'],'该序列首个未排除抽帧'))
    for val,k in [ranked[0],ranked[-1]]:
        if k not in [(s,f) for s,f,_ in chosen]:chosen.append((*k,f'开发集逐观测LOO变化中位数 {val:+.2f}px'))
    galleries=[]
    for session,frame,reason in chosen:
        g=next(g for g in groups if g['session']==session and g['frame']==frame)
        with np.load(OUT/g['diagnostic_path']) as z:a=dict(z)
        with np.load(BASE/'geometry_diagnostics_NOT_GT'/g['split']/session/f'frame_{frame:06d}.npz') as z:old=dict(z)
        folder=OUT/'examples'/f'{session}_frame_{frame:06d}';folder.mkdir(parents=True,exist_ok=True)
        cams=g['cameras'];sheet=Image.new('RGB',(1600,110+math.ceil(len(cams)/2)*530),'#17212c');sd=ImageDraw.Draw(sheet)
        sd.text((15,10),f'{session} 源帧{frame:06d} | 每格左=旧LS / 右=Anipose稳健优化',font=font(27),fill='white')
        sd.text((15,52),'绿圈=人工；粉×=旧LS；蓝×=Anipose；黄色23腹面原样保留。原标注未覆盖。',font=font(24),fill='#cdddeb')
        for ci,c in enumerate(cams):
            vi=a['camera_names'].tolist().index(c)
            src=BASE/'real_images'/g['split']/session/f'frame_{frame:06d}'/f'cam{c}.png'
            raw=Image.open(src).convert('RGB');ann=json.loads(src.with_suffix('.json').read_text())
            xy=np.array([p for s in ann['shapes'] for p in s['points']])
            lo=np.maximum(xy.min(0)-40,0);hi=np.minimum(xy.max(0)+40,[1080,1200]);box=tuple(map(int,[*lo,*hi]))
            previous=np.full((23,2),np.nan)
            for k in ACTIVE:
                if np.isfinite(old['xyz_all_ls_mm'][k]).all():previous[k]=projection(old['xyz_all_ls_mm'][k],old['K'][ci:ci+1],old['world_to_camera_mm'][ci:ci+1])[0][0]
            left=overlay(raw,a['uv_original'][vi],previous,'#ff78d1').crop(box)
            right=overlay(raw,a['uv_original'][vi],a['uv_projection_all'][vi],'#50d5ff').crop(box)
            detail=Image.new('RGB',(left.width*2,left.height+60),'#17212c')
            detail.paste(left,(0,60));detail.paste(right,(left.width,60))
            ImageDraw.Draw(detail).text((10,10),f'{session} {frame:06d} cam{c} | 旧LS / Anipose | 23不动',font=font(25),fill='white')
            detail.save(folder/f'cam{c}_comparison.jpg',quality=95)
            detail.thumbnail((785,510));sheet.paste(detail,((ci%2)*800,110+(ci//2)*530))
        sheet.save(folder/'multiview_comparison.jpg',quality=95)
        galleries.append(dict(session=session,frame=frame,split=g['split'],reason=reason,folder=folder.relative_to(OUT).as_posix(),cameras=cams))
    m=summary['approved_non_tail'];p=m['paired_loo']
    report=[
        'PyMouseStudio Anipose非尾部对照 | 2026-10-01',
        '结论：稳健优化小幅改善，不足以证明真实3D精度提高；原人工标注不覆盖。',
        f"排除D01–D06及C01–C18，共24组多视角帧；仅处理剩余{len(groups)}组。所有排除帧不参与拟合、评估和可视化选例。",
        '保留范围：08–14所有尾部点、23腹面点（无论几个视角）、任何仅0/1视角观测的点原样保留。',
        f"已验证原样保留尾/腹观测{summary['preserved_tail_and_abdomen_observations']}个，单视角非尾部观测{summary['preserved_singleview_body_observations']}个。未补不可见点作为人工监督。",
        '实际调用已安装aniposelib 0.8.0（本机JAX版triangulate，源文件SHA256见config.json），triangulate + optim_points；不是自写算法冒称Anipose。',
        'soft_l1、15px阈值；相机固定；不启用RANSAC筛选。稀疏抽帧不做时序平滑；骨长约束为空，避免把体表/暂缓定义槽位当刚性骨架。',
        '批处理只为提高速度，所有点独立，时序/骨长损失均为0。标定D=0；库中的undistort仅做像素转归一化射线，没有再次畸变校正图像。',
        '完整计算包含15个非尾非腹槽位；主要结论只看已确认的7个槽位：01、06、07、15、16、17、18。其他槽位结果不等于定义已验收。',
        f"主要槽位拟合内中位误差：{m['baseline_ls']['fit']['median']:.2f} → {m['anipose_robust']['fit']['median']:.2f}px。",
        f"同一批{p['n']}个留一视角观测中位误差：{p['before']['median']:.2f} → {p['after']['median']:.2f}px；P90 {p['before']['p90']:.2f} → {p['after']['p90']:.2f}px。",
        f"逐观测变化中位数{p['median_after_minus_before_px']:+.3f}px；改善>2px占{p['improved_over2px_fraction']:.1%}，变差>2px占{p['worsened_over2px_fraction']:.1%}。两种中位数的差并不等于逐观测差的中位数。",
        'LOO=被评估相机的人工坐标不参与初始化、优化或观测筛选；不是用拟合后的2D坐标自证误差为零。',
        '评估参照仍是带噪人工2D，不存在独立精确3D真值；相机系统偏差、折射或语义选点漂移不会因重投影自动消失。',
        '开发/留出分离，固定算法都只作审计，无训练、无相机调整、无自动改左右。',
        '输出diagnostic_npz_NOT_GT含276组：原始观测、建议重投影坐标、更新掩码、3D毫米坐标及LOO投影；train_allowed=false。',
        '官方接口：https://anipose.readthedocs.io/en/latest/aniposelib-api.html',
        '',
        '尾巴路线建议（设计评估，尚未运行精细分割或新3D尾巴重建）',
        '1. 将人工尾点作为软参考/分割提示，不再把不同视角相同编号的中间点强制视为同一个物理位置。尾根/真实尾尖先确认，遮挡终点不是尾尖。',
        '2. 优先在尾部高分辨率裁剪图上，用SAM2正点+负点+局部框精分割；以现有人工多边形作辅助。负点排除玻璃边缘、反射、爪和躯干；不得只沿原始错误折线涂粗管来假装精细分割。',
        '3. 将裁剪结果映射回原1080×1200坐标；每视角保留visible/occluded/uncertain范围。分割不存在并不意味着被遮挡的3D尾段不存在。',
        '4. 对清晰尾段提取中心线及半径，保留自交分支候选，不把2D骨架分支任意连接。二维相同比例弧长因透视缩短和遮挡不对应同一三维位置。',
        '5. 用多视角极线约束、顺序单调对应和至少第三视角验证初始化3D中心线；再以投影中心线/渲染细管与分割的一致性优化3D样条。可见尾段双向覆盖，遮挡尾段不强制解释所有背景。',
        '6. 辅以柔性的曲率、半径渐变和长度先验；不要用过强平滑把真实卷曲拉直。不必先得到完整鼠体网格；视觉外壳仅作范围约束，容易在遮挡处变粗或断裂。',
        '7. 在毫米3D中心线上累计弧长s，再于s/L=r的位置采样。是沿曲线的比例，不是端点直线插值，也不是每张2D图分别等分后再三角化。',
        '8. 定义需先统一：当前合成Core23尾部来自Tail0/2/4/6/8/10/11_M的骨头位置，不保证等距，也不保证最后骨头头端是真实尾尖。不能直接把新等分点塞入旧tail_q通道并声称定义一致。',
        '9. 推荐新建tail_centerline_arclength_v1；可选7点比例0,1/6,2/6,3/6,4/6,5/6,1，但这只是待确认的新定义。要兼容现有通道，先测量统一静止模型的实际比例并对齐尾尖；合成和真实必须使用同一契约。',
        '10. 新点先作为pseudo-GT：保存分割置信度、多视角支持、留一相机曲线误差、相机子集重建离散度、遮挡/插值标记。整段长度或尾尖不可靠时，不输出虚假的全尾归一化比例真值。',
        '11. 验证先用合成已知3D中心线/分割和真实少量清晰尾段，单独测尾根、尾尖、弯曲、自交/遮挡病例。稀疏标注不跨帧强平滑；真正连续视频以后再加时序约束。',
        '12. 本轮24组待复核帧继续隔离，腹部不动。先复核camA/camF标定适用性，再试点精分割与3D曲线，可避免尾曲线通过错误相机拟合出看似一致但偏移的形状。',
        'SAM2点/框/掩码提示官方实现：https://github.com/facebookresearch/sam2/blob/main/sam2/sam2_image_predictor.py',
        '多视角曲线重建相关工作（非鼠尾现成方案）：https://openaccess.thecvf.com/content/WACV2025/papers/Zheng_3D_Edge_Sketch_from_Multiview_Images_WACV_2025_paper.pdf',
    ]
    for split,s in summary['splits'].items():
        report.append(f"{split} LOO：{s['paired_loo']['before']['median']:.2f} → {s['paired_loo']['after']['median']:.2f}px")
    (OUT/'结果与尾巴路线.txt').write_text('\n'.join(report),encoding='utf-8-sig')
    (OUT/'visualization_manifest.json').write_text(json.dumps(galleries,ensure_ascii=False,indent=2))
    contents=''.join(f'<p>{html.escape(t)}</p>' for t in report[:18])
    (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Anipose非尾部对照</title><style>body{background:#17212c;color:white;font:18px sans-serif;line-height:1.6;margin:24px}a{color:#7cd7ff}img{max-width:100%}</style><h1>Anipose 非尾部重投影对照</h1>'+contents+'<a href="结果与尾巴路线.txt">完整结果与尾巴路线</a><p><img src="paired_loo_by_keypoint.png"></p>'+''.join(f'<h2>{g["session"]} 源帧{g["frame"]:06d}</h2><p>{g["reason"]}</p><a href="{g["folder"]}/multiview_comparison.jpg"><img loading="lazy" src="{g["folder"]}/multiview_comparison.jpg"></a><p>'+' · '.join(f'<a href="{g["folder"]}/cam{c}_comparison.jpg">cam{c}放大对照</a>' for c in g['cameras'])+'</p>' for g in galleries),encoding='utf-8')
    dest=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-Anipose-Review-20261001')
    assert not dest.exists();shutil.copytree(OUT,dest)
    print(json.dumps(dict(galleries=len(galleries),windows_path=str(dest)),indent=2))


if __name__=='__main__':main()
