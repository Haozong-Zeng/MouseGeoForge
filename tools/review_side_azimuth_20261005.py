"""Review the matched side-azimuth data control without altering prior reports."""
import json, shutil, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import torch
import numpy as np
from PIL import Image,ImageDraw
import review_pixel_adapter_four_arm_20261005 as gallery
from train_pixel_adapter_matched_20261005 import prepare_rows,KP
from run_surface_curve_pilot_20261003 import load_item
from run_depth_pose_ablation import write_json
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005'
BASE=CYCLE/'pixel_adapter_matched_pilot_v1'
NEW=CYCLE/'single_side_azimuth_control_v1'
OUT=CYCLE/'single_side_azimuth_review_v1'
DEST=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005')/OUT.name
def read(p):return torch.load(p,map_location='cpu',weights_only=False)
def main():
    OUT.mkdir(exist_ok=True);DEST.mkdir(exist_ok=True);torch.set_num_threads(2)
    old=json.loads((BASE/'results.json').read_text())['summary'];new=json.loads((NEW/'results.json').read_text())
    summary={k:old[k] for k in ['baseline','head_only']};summary['side_azimuth']=new['summary']
    data=read(BASE/'predictions.pt');pred={k:data['predictions'][k] for k in summary if k!='side_azimuth'}
    pred['side_azimuth']=read(NEW/'predictions.pt')['predictions'];rows=prepare_rows()
    gallery.OUT=OUT;gallery.ARMS=list(summary);gallery.COLORS={'baseline':'#222222','head_only':'#266abf','side_azimuth':'#bd337e'}
    curves=[r for r in json.loads((BASE/'curves.json').read_text()) if r['arm']=='head_only']
    curves += [dict(r,arm='side_azimuth') for r in json.loads((NEW/'curves.json').read_text())]
    gallery.chart(curves,summary)
    error={a:((p[:,:,KP]-data['gt'])*data['pixels'][:,None,None]).norm(dim=-1) for a,p in pred.items()}
    examples=[]
    for label,predicate in [('real_bottom',lambda m:m['physical_bottom']),('real_side',lambda m:m['kind']=='real' and not m['physical_bottom']),('dual',lambda m:m['kind']=='dual')]:
        candidates=[(float(error['side_azimuth'][i,q,k]-error['head_only'][i,q,k]),i,q,k) for i,m in enumerate(data['metadata']) if m['stage']=='validation' and predicate(m) for q in range(6) for k in range(1,5) if data['valid'][i,q,k]]
        for mode,entry in [('best',min(candidates)),('worst',max(candidates))]:
            delta,i,q,k=entry;r=rows[i];m=r['metadata'];item=load_item(r['record']);vi=r['record']['views'].index(m['view']);rgb=item['batch']['images'][0,vi]
            image=Image.fromarray((rgb.permute(1,2,0).numpy().clip(0,1)*255).round().astype(np.uint8));w,h=image.size
            tile=Image.new('RGB',(1830,800),'white');td=ImageDraw.Draw(tile)
            for ci,(name,p) in enumerate(pred.items()):
                panel=image.copy();d=ImageDraw.Draw(panel);x,y=(p[i,q,KP[k]]*torch.tensor([w,h])).tolist();gx,gy=(data['gt'][i,q,k]*torch.tensor([w,h])).tolist()
                d.line((x,y,gx,gy),fill='yellow',width=2);d.ellipse((x-8,y-8,x+8,y+8),outline='red',width=4);d.line((gx-10,gy,gx+10,gy),fill='#00ff66',width=4);d.line((gx,gy-10,gx,gy+10),fill='#00ff66',width=4)
                panel.thumbnail((594,650));tile.paste(panel,(ci*610+(610-panel.width)//2,130))
                td.text((ci*610+12,12),name,font=gallery.font(24),fill='black');td.text((ci*610+12,48),f'error {error[name][i,q,k]:.2f} input px',font=gallery.font(22),fill='black');td.text((ci*610+12,83),f'{label} view{m["view"]} q{q} point{KP[k]+1}',font=gallery.font(20),fill='black')
            file=f'{label}_{mode}.jpg';tile.save(OUT/file,quality=94);examples.append(dict(file=file,category=label,selection=mode,delta=delta,index=i,query=q,point=KP[k]+1,metadata=m,errors={a:float(e[i,q,k]) for a,e in error.items()}))
    write_json(OUT/'results.json',dict(summary=summary,cost={k:new[k] for k in ['train_seconds','total_seconds','peak_gpu_bytes']},frozen_before=new['frozen_before'],frozen_after=new['frozen_after']))
    write_json(OUT/'visual_examples.json',examples)
    page=['<!doctype html><meta charset="utf-8"><title>侧面方位角覆盖对照</title><style>body{font:17px system-ui;max-width:1750px;margin:30px auto;padding:15px}img{max-width:100%}td,th{border:1px solid #aaa;padding:7px}table{border-collapse:collapse}</style>',
    '<h1>增加侧面方位角：有局部收益，但没有解决底视</h1><p>固定43个原TRAIN合成单鼠姿态、400步、初始化、真实/双鼠采样和相机方向类别。400次单鼠抽样仍为111顶部、203侧面、86底部，仅148次侧面相机更换；9次新视图没有合格匹配点，仍前向且损失为零，不重采样。不是新增独立视频，也不是提高底视曝光。</p>',
    '<p>开发总体均值：原模型22.267、原head-only微调22.048、扩充侧面21.665px。真实为31.985/34.461/33.227，底脚为43.465/45.853/46.644；因此整体均值改善不能当作真实或底视改善。单种子、反复使用的开发验证、检测框匹配后的条件误差，不是AP或外部测试。保留原模型，不晋升。</p>',
    '<p>本数据对照与近期head-only使用相同五点MSE及优化器，因而相互可比；但与原基础模型训练配方不同。原配方使用按输入像素尺度归一的SmoothL1、更多合成点、较低分组学习率和不同真实/合成权重。这些差异需拆开验证，不能归咎于单一因素。</p>']
    for phase in ['calibration','validation']:
        page.append(f'<h2>{phase}</h2><table><tr><th>范围</th><th>方法</th><th>known</th><th>mean↓</th><th>P95↓</th><th>PCK32↑</th></tr>')
        for kind in ['all','real','single','dual','real_bottom_feet']:
            for arm,s in summary.items():
                v=s[phase][kind];page.append(f'<tr><td>{kind}</td><td>{arm}</td><td>{v["known"]}</td><td>{v["mean"]:.3f}</td><td>{v["p95"]:.3f}</td><td>{v["pck32"]:.3f}</td></tr>')
        page.append('</table>')
    page += ['<h2>收敛曲线</h2><img src="calibration_curves.png"><img src="fit_curves.png">','<h2>新增顶部图的TRAIN检测诊断</h2><p>21张Cylinder顶部图均有1个有效mask实例和5个有效点；16张因原预测框IoU不足0.5而没有合格点，并非缺标签。10张出现两个高置信预测。顶部RGB明显偏暗；未加入本侧面方位角实验。图中框是模型预测，不是GT裁剪推理。</p><img src="top_qualification_examples.jpg">','<h2>相对head-only的最好/最坏变化</h2><p>事后极值示例，不代表发生频率。红圈预测，绿十字评价GT，单位为输入像素。</p>']
    for e in examples:page.append(f'<h3>{e["category"]} {e["selection"]}: {e["delta"]:+.2f}px</h3><img src="{e["file"]}">')
    page.append('<p><a href="results.json">指标与成本</a> | <a href="visual_examples.json">图例索引</a> | <a href="top_qualification_audit.json">顶部诊断</a></p>')
    (OUT/'index.html').write_text('\n'.join(page),encoding='utf-8')
    for name in ['top_qualification_examples.jpg','top_qualification_audit.json']:shutil.copy2(CYCLE/'single_train_allview_rgb_inputs_v1'/name,OUT/name)
    for p in OUT.iterdir():
        if p.is_file():shutil.copy2(p,DEST/p.name)
    print(DEST/'index.html')
if __name__=='__main__':main()
