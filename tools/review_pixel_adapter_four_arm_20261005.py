"""Four-arm decomposition report; preserves the earlier matched pilot report."""
import html,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from PIL import Image,ImageDraw
from review_pixel_adapter_20261005 import font
from train_pixel_adapter_matched_20261005 import prepare_rows,KP
from run_surface_curve_pilot_20261003 import load_item
from run_depth_pose_ablation import write_json
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';BASE=CYCLE/'pixel_adapter_matched_pilot_v1';SOLO=CYCLE/'pixel_adapter_only_pilot_v1'
OUT=CYCLE/'pixel_adapter_four_arm_review_v1';DEST=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005')/OUT.name
ARMS=['baseline','lora_only','head_only','head_plus_lora'];COLORS={'baseline':'#222222','lora_only':'#249966','head_only':'#266abf','head_plus_lora':'#bd337e'}
def chart(curves,summary):
    for phase in ['fit','calibration']:
        canvas=Image.new('RGB',(1440,950),'white');d=ImageDraw.Draw(canvas)
        for n,kind in enumerate(['real','single','dual','real_bottom_feet']):
            x=70+(n%2)*710;y=65+(n//2)*425;w=580;h=300
            vals=[r[phase][kind]['mean'] for r in curves if r[phase][kind]['mean'] is not None];lo=0;upper=max(10,np.ceil(max(vals)/10)*10)
            d.text((x,y-33),kind+' '+phase+' mean px',font=font(21),fill='black')
            for j in range(6):
                v=upper*j/5;yy=y+h-h*j/5;d.line((x,yy,x+w,yy),fill='#ddd');d.text((x-48,yy-8),f'{v:.0f}',font=font(15),fill='black')
            baseline=summary['baseline']['fit_monitor' if phase=='fit' else phase][kind]['mean'];yy=y+h-baseline/upper*h
            d.line((x,yy,x+w,yy),fill=COLORS['baseline'],width=2)
            for arm in ARMS[1:]:
                pts=[(x+r['step']/400*w,y+h-r[phase][kind]['mean']/upper*h) for r in curves if r['arm']==arm]
                if len(pts)>1:d.line(pts,fill=COLORS[arm],width=3)
                for xx,yy in pts:d.ellipse((xx-4,yy-4,xx+4,yy+4),fill=COLORS[arm])
            for step in [0,100,200,300,400]:d.text((x+step/400*w-8,y+h+10),str(step),font=font(15),fill='black')
        for j,arm in enumerate(ARMS):d.text((35+j*1400/len(ARMS),905),arm,font=font(19),fill=COLORS[arm])
        canvas.save(OUT/f'{phase}_curves.png')
def main():
    assert json.loads((SOLO/'status.json').read_text())['stage']=='complete'
    OUT.mkdir(exist_ok=True);DEST.mkdir(exist_ok=True);torch.set_num_threads(2)
    mainres=json.loads((BASE/'results.json').read_text());solo=json.loads((SOLO/'results.json').read_text());summary={**mainres['summary'],'lora_only':solo['summary']}
    data=torch.load(BASE/'predictions.pt',map_location='cpu',weights_only=False);sdata=torch.load(SOLO/'predictions.pt',map_location='cpu',weights_only=False)
    assert data['evaluated_ids']==sdata['evaluated_ids'];pred={**data['predictions'],'lora_only':sdata['predictions']};rows=prepare_rows()
    curves=json.loads((BASE/'curves.json').read_text())+[dict(r,arm='lora_only') for r in json.loads((SOLO/'curves.json').read_text())];chart(curves,summary)
    error={a:((p[:,:,KP]-data['gt'])*data['pixels'][:,None,None]).norm(dim=-1) for a,p in pred.items()};chosen=[]
    for label,predicate in [('real_bottom',lambda m:m['physical_bottom']),('real_side',lambda m:m['kind']=='real' and not m['physical_bottom']),('dual',lambda m:m['kind']=='dual'),('single',lambda m:m['kind']=='single')]:
        candidates=[(float(error['lora_only'][i,q,k]-error['baseline'][i,q,k]),i,q,k) for i,m in enumerate(data['metadata']) if m['stage']=='validation' and predicate(m) for q in range(6) for k in range(1,5) if data['valid'][i,q,k]]
        for mode,entry in [('best',min(candidates)),('worst',max(candidates))]:
            delta,i,q,k=entry;r=rows[i];m=r['metadata'];item=load_item(r['record']);vi=r['record']['views'].index(m['view']);rgb=item['batch']['images'][0,vi]
            image=Image.fromarray((rgb.permute(1,2,0).numpy().clip(0,1)*255).round().astype(np.uint8));w,h=image.size
            allxy=[pred[a][i,q,KP[k]]*torch.tensor([w,h]) for a in ARMS]+[data['gt'][i,q,k]*torch.tensor([w,h])]
            pts=torch.stack(allxy);left=max(0,int(pts[:,0].min())-35);top=max(0,int(pts[:,1].min())-35);right=min(w,int(pts[:,0].max())+36);bottom=min(h,int(pts[:,1].max())+36)
            tile=Image.new('RGB',(2040,1040),'white');draw=ImageDraw.Draw(tile)
            for ci,a in enumerate(ARMS):
                panel=image.copy();d=ImageDraw.Draw(panel);xy=allxy[ci];gt=allxy[-1];xx,yy=map(float,xy);gx,gy=map(float,gt)
                d.line((xx,yy,gx,gy),fill='yellow',width=2);d.ellipse((xx-7,yy-7,xx+7,yy+7),outline='red',width=3);d.line((gx-8,gy,gx+8,gy),fill='#00ff66',width=3);d.line((gx,gy-8,gx,gy+8),fill='#00ff66',width=3)
                crop=panel.crop((left,top,right,bottom));crop.thumbnail((480,245));small=panel.copy();small.thumbnail((490,555))
                tile.paste(small,(ci*510+(510-small.width)//2,125));tile.paste(crop,(ci*510+(510-crop.width)//2,740))
                draw.text((ci*510+12,10),a,font=font(23),fill=COLORS[a]);draw.text((ci*510+12,45),f'error {error[a][i,q,k]:.2f} input px',font=font(21),fill='black');draw.text((ci*510+12,79),f'{label} view{m["view"]} q{q} point{KP[k]+1}',font=font(18),fill='black')
                draw.text((ci*510+12,705),'Evaluation-only same-region detail',font=font(17),fill='#555')
            file=f'{label}_{mode}.jpg';tile.save(OUT/file,quality=94);chosen.append(dict(category=label,selection=mode,delta_lora_minus_original=delta,index=i,query=q,point=KP[k]+1,metadata=m,errors={a:float(error[a][i,q,k]) for a in ARMS},file=file))
    ids=data['evaluated_ids'];quality=torch.load(CYCLE/'point_quality_inputs_v1/inputs.pt',map_location='cpu',weights_only=False);conf=quality['confident'][ids];drift={}
    for arm in ARMS[1:]:
        e=((pred[arm][ids]-pred['baseline'][ids])*data['pixels'][ids,None,None]).norm(dim=-1)
        drift[arm]={label:dict(mean=float(e[:,:,kp][conf].mean()),max=float(e[:,:,kp][conf].max())) for label,kp in [('nose_feet',[0,14,15,16,17]),('tail',list(range(7,14))),('abdomen',[22]),('all23',list(range(23)))]}
    payload=dict(summary=summary,confident_query_drift=drift,scope='Known-point conditional error; drift is not accuracy. Reused development one seed. LoRA-only added after two-arm inspection, not preregistered pristine test. All arms use same400imageschedule, extra cost/params notmatched.')
    write_json(OUT/'results.json',payload);write_json(OUT/'visual_examples.json',chosen)
    page=['<!doctype html><meta charset="utf-8"><title>图像适配与姿态头四组分解</title><style>body{font:17px system-ui;max-width:1950px;margin:24px auto;padding:18px}img{max-width:100%}td,th{border:1px solid #aaa;padding:7px}table{border-collapse:collapse}</style>',
      '<h1>图像适配与姿态头：四组分解</h1><p>baseline（全部冻结）、LoRA-only（仅最后图像骨干层适配）、head-only（仅姿态头）、head+LoRA。训练组同初始状态、源划分、400次图像采样计划、学习率及loss。LoRA-only是首轮结果后的探索性补充，不冒充事前注册。原骨干/对应冻结头的权重哈希已核验不变。</p>',
      '<p>768实际RGB，固定原预测框而非GT框；鼻尖/四脚条件定位，不是完整检测或3D指标。完整6实例/138点前向，GT仅用于loss/评价匹配。缺失标注不算负例。真实开发FR2/AZR2L2反复使用、基础模型已见原TRAIN；内部校验仅对本阶段更新留出。单种子，不能凭小幅波动宣称架构更好。正式模型与数据不变。</p>']
    page+=['<h2>当前结论：冻结头更稳定，但尚无整体提升</h2><p>LoRA-only开发整体22.272px，原模型22.267px，基本持平；真实32.124对31.985、单鼠19.307对19.225略差，双鼠20.365对20.425略好。底视脚42.332对43.465略好，但仅24个已知点，PCK32仍为0.5。不能宣称已解决底视。</p>',
      '<p>内部校验底脚31.130→30.354px，未出现更新姿态头时41–42px的回退；整体24.790→24.831并未改善。可见脚仍会被预测到腹部；最好底脚102.68→86.71px仍明显错误，某侧面脚89.26→125.73px恶化。只更新图像适配也不保证所有点安全。</p>',
      '<p>仅适配图像时，原高置信query的尾巴坐标平均漂移2.48px、腹部2.26px，比更新头时小，但只是漂移不是精度。下一步为400→最多1600步的低学习率1e-5续训检查，每200步检查内部校验，连续两次超过预设误差/尾巴漂移上限即停止；不以开发集选epoch。不是与400步头部训练的同预算胜负对比。</p>']
    for phase in ['calibration','validation']:
        page.append(f'<h2>{phase}（模型输入像素）</h2><table><tr><th>范围</th><th>方法</th><th>已知/未知</th><th>mean↓</th><th>P95↓</th><th>PCK32↑</th></tr>')
        for kind in ['all','real','single','dual','real_bottom_feet']:
            for arm in ARMS:
                r=summary[arm][phase][kind];page.append(f'<tr><td>{kind}</td><td>{arm}</td><td>{r["known"]}/{r["unknown"]}</td><td>{r["mean"]:.3f}</td><td>{r["p95"]:.3f}</td><td>{r["pck32"]:.3f}</td></tr>')
        page.append('</table>')
    page+=['<h2>固定72图训练监视、220图内部校验</h2><p>不按开发选择步数；黑线是冻结原模型。底视内部校验71个已知脚点，开发仅24个，不能混用。</p><img src="fit_curves.png"><img src="calibration_curves.png">',
      '<h2>LoRA-only相对原模型的极值图例</h2><p>各范围事后选择最好/最坏脚点变化，不代表频率。红圈预测、绿十字GT。下方相同局部区域仅为评价展示，不进入模型，也不是GT裁剪推理。标题仍使用未显示缩放的模型输入像素。</p>']
    for e in chosen:page.append(f'<h3>{e["category"]} {e["selection"]}: {e["delta_lora_minus_original"]:+.2f}px</h3><img src="{e["file"]}">')
    page+=['<h2>未监督输出的漂移</h2><p>以下只统计原预测score≥0.5的query，包含缺少真值者。它是相对原模型坐标的变化量，不是尾巴/腹部的误差；不能据此生成新标注。</p><pre>'+html.escape(json.dumps(drift,indent=2))+'</pre>',
      '<p><a href="results.json">四组结果与漂移</a> | <a href="visual_examples.json">图例编号</a> | <a href="../pixel_adapter_review_v1/index.html">前一轮完整方案、成本与负结果</a></p>']
    (OUT/'index.html').write_text('\n'.join(page),encoding='utf-8')
    for p in OUT.iterdir():
        if p.is_file():shutil.copy2(p,DEST/p.name)
    print(json.dumps(summary['lora_only'],indent=2));print(DEST/'index.html')
if __name__=='__main__':main()
