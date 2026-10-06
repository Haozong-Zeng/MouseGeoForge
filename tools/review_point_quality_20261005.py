"""Quality pilot visual review: ranking, calibration and confidently wrong points."""
import json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from run_surface_curve_pilot_20261003 import load_item
from run_depth_pose_ablation import write_json
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';OUT=CYCLE/'point_quality_review_v1';DEST=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005')/OUT.name
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf';KP=[0,14,15,16,17]


def font(n):return ImageFont.truetype(FONT,n)


def chart(curves,results):
    im=Image.new('RGB',(1530,1040),'white');draw=ImageDraw.Draw(im);colors={'scalar/fit':'#1b72c2','scalar/calibration':'#df4429','semantic/fit':'#159966','semantic/calibration':'#9b38a4'}
    for ri,seed in enumerate([17,29,43]):
        for ci,kind in enumerate(['real','single','dual']):
            x0=60+ci*510;y0=48+ri*320;w=405;h=235
            draw.text((x0,y0-28),f'{kind} seed{seed} Brier',font=font(18),fill='black')
            for value in [0,.1,.2,.3]:
                y=y0+h-value/.3*h;draw.line((x0,y,x0+w,y),fill='#ddd');draw.text((x0-42,y-6),f'{value:.1f}',font=font(13),fill='black')
            for label,color in colors.items():
                arm,stage=label.split('/');rows=[r for r in curves if r['seed']==seed and r['arm']==arm and r['stage']==stage]
                draw.line([(x0+r['step']/800*w,y0+h-r['metrics'][kind]['brier']/.3*h) for r in rows],fill=color,width=3)
            for step in [0,400,800]:draw.text((x0+step/800*w-8,y0+h+4),str(step),font=font(13),fill='black')
    for i,(label,color) in enumerate(colors.items()):draw.text((65+i*365,1005),label,font=font(18),fill=color)
    im.save(OUT/'quality_brier_curves.png')
    im=Image.new('RGB',(1450,900),'white');draw=ImageDraw.Draw(im);colors={'objectness':'#888888','scalar_seed17':'#2673c8','semantic_seed17':'#b747ba','monotonic_std':'#179b5d'}
    for ci,kind in enumerate(['real','single','dual','real_bottom_feet']):
        x0=75+(ci%2)*725;y0=55+(ci//2)*410;w=590;h=300
        draw.text((x0,y0-29),kind+' risk / coverage',font=font(20),fill='black')
        for value in [0,20,40,60,80]:
            y=y0+h-value/80*h;draw.line((x0,y,x0+w,y),fill='#ddd');draw.text((x0-45,y-8),str(value),font=font(14),fill='black')
        for name,color in colors.items():
            rows=results[name]['validation'][kind]['risk_coverage'];points=[(x0+r['actual_coverage']*w,y0+h-r['mean_error']/80*h) for r in rows if r['mean_error'] is not None]
            if len(points)>1:draw.line(points,fill=color,width=3)
            for a,b in points:draw.ellipse((a-3,b-3,a+3,b+3),fill=color)
        for f in [0,.25,.5,.75,1]:draw.text((x0+f*w-5,y0+h+6),f'{f:.0%}',font=font(14),fill='black')
    for i,(name,color) in enumerate(colors.items()):draw.text((65+i*345,865),name,font=font(17),fill=color)
    im.save(OUT/'quality_risk_coverage.png')


def examples(data,scores):
    metadata=data['metadata'];chosen=[]
    for label,predicate in [('real_bottom',lambda m:m['kind']=='real' and m['physical_bottom']),('real_side',lambda m:m['kind']=='real' and not m['physical_bottom']),('dual',lambda m:m['kind']=='dual'),('single',lambda m:m['kind']=='single')]:
        candidates=[(float(scores['monotonic_std'][i,q,k]),i,q,k) for i,m in enumerate(metadata) if m['stage']=='validation' and predicate(m)
                    for q in range(6) for k in range(1,5) if data['valid'][i,q,k] and data['error'][i,q,k]>32]
        score,i,q,k=max(candidates);chosen.append(dict(category=label,index=i,query=q,k_index=k,point=KP[k]+1,error=float(data['error'][i,q,k]),
            scores={name:float(scores[name][i,q,k]) for name in ['objectness','scalar_seed17','semantic_seed17','monotonic_std']},metadata=metadata[i]))
    base={role:torch.load(CYCLE/'singleview_fusion_inputs_seed17_v1'/f'{role}.pt',map_location='cpu',weights_only=False) for role in ['validation']}
    lookup={(m['record']['scene'],int(m['view'])):(i,m) for i,m in enumerate(base['validation']['metadata'])}
    dual=torch.load(CYCLE/'dual_allscene_association_audit_v1/groups.pt',map_location='cpu',weights_only=False)
    canvas=Image.new('RGB',(1600,1920),'white');draw=ImageDraw.Draw(canvas)
    for ci,e in enumerate(chosen):
        m=e['metadata'];q=e['query'];kp=e['point']-1
        if m['kind']=='dual':
            g=dual[m['group_index']];vi=g['record']['views'].index(m['view']);image=Image.open(CYCLE/'dual_allscene_association_audit_v1'/g['images'][vi]).convert('RGB')
            d={k:g['evaluation_data'][k][vi] for k in ['gt_boxes','box_valid','instance_valid','gt_xy']};boxes=g['predictions']['boxes'][vi];logits=g['predictions']['objectness_logits'][vi]
        else:
            bi,meta=lookup[m['scene'],m['view']];item=load_item(meta['record']);vi=meta['record']['views'].index(m['view']);rgb=item['batch']['images'][0,vi]
            image=Image.fromarray((rgb.permute(1,2,0).numpy().clip(0,1)*255).round().astype(np.uint8));b=base['validation'];d={k:b[k][bi] for k in ['gt_boxes','box_valid','instance_valid','gt_xy']};boxes=b['boxes'][bi];logits=b['objectness_logits'][bi]
        from mousegeoforge.detection_readout_loss import match_detection
        qs,ts=match_detection(dict(instance_boxes_cxcywh=boxes[None],instance_objectness_logits=logits[None]),d['gt_boxes'][None],(d['box_valid']&d['instance_valid'])[None])[0];target=dict(zip(qs.tolist(),ts.tolist()))[q]
        w,h=image.size;di=ImageDraw.Draw(image);cx,cy,bw,bh=boxes[q].tolist();di.rectangle(((cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h),outline='white',width=3)
        xy=data['xy'][e['index'],q,kp];gt=d['gt_xy'][target,kp];a,b=float(xy[0])*w,float(xy[1])*h;c,f=float(gt[0])*w,float(gt[1])*h
        di.line((a,b,c,f),fill='yellow',width=2);di.ellipse((a-9,b-9,a+9,b+9),outline='red',width=4);di.line((c-10,f,c+10,f),fill='#00ff66',width=4);di.line((c,f-10,c,f+10),fill='#00ff66',width=4)
        x=ci%2*800;y=ci//2*960
        draw.text((x+15,y+15),f'{e["category"]} view{m["view"]} q{q} point{kp+1}',font=font(22),fill='black')
        draw.text((x+15,y+50),f'Error {e["error"]:.1f}px | mono P32 {e["scores"]["monotonic_std"]:.3f}',font=font(20),fill='black')
        draw.text((x+15,y+80),f'scalar {e["scores"]["scalar_seed17"]:.3f} / semantic {e["scores"]["semantic_seed17"]:.3f}',font=font(19),fill='black')
        draw.text((x+15,y+110),'Red circle=unchanged prediction; green +=GT evaluation only',font=font(17),fill='black')
        canvas.paste(image,(x+(800-w)//2,y+160))
    canvas.save(OUT/'confident_wrong_feet.jpg',quality=94);write_json(OUT/'visual_examples.json',chosen)


def main():
    OUT.mkdir(exist_ok=True);DEST.mkdir(exist_ok=True)
    pilot=json.loads((CYCLE/'point_quality_pilot_v1/results.json').read_text());simple=json.loads((CYCLE/'point_quality_simple_baselines_v1/results.json').read_text());results={**pilot['results'],**simple['results']}
    curves=json.loads((CYCLE/'point_quality_pilot_v1/curves.json').read_text());data=torch.load(CYCLE/'point_quality_inputs_v1/inputs.pt',map_location='cpu',weights_only=False)
    scores={**torch.load(CYCLE/'point_quality_pilot_v1/scores.pt',map_location='cpu',weights_only=False),**torch.load(CYCLE/'point_quality_simple_baselines_v1/scores.pt',map_location='cpu',weights_only=False)}
    chart(curves,results);examples(data,scores)
    write_json(OUT/'results.json',dict(results=results,simple_fit=simple['fit'],training_runs=pilot['runs']))
    page=['<!doctype html><meta charset="utf-8"><title>逐关键点可靠性</title><style>body{font:17px system-ui;max-width:1580px;margin:25px auto;padding:15px}img{max-width:100%}td,th{border:1px solid #aaa;padding:7px}table{border-collapse:collapse}</style>',
        '<h1>逐点可靠性：简单方差校准胜过新小头，但仍有自信的错误</h1><p>此轮只预测 P(二维误差≤32个模型输入像素)，不改坐标、框或标签，也不是可见性分类。主模型冻结；所有6×5点前向，只有原预测置信≥0.5、框IoU≥0.5、存在点标注者参与loss/评价。缺失及身份不明点不充当负例。</p>',
        '<p>新阶段fit1024图、校准220图、开发144图。校准真实FR3整序列、合成爬墙整动作、双鼠训练源内5场景；双鼠2视角缓存替换成4视角而非重复加入。21张来源不明确的curve-kind训练图暂未纳入。基础姿态模型见过原TRAIN，所以只是新置信头的阶段性留出；开发集多次使用，不是全新外测。</p>',
        '<h2>结论</h2><p>不能仅凭低校准误差认定置信度有用：恒定预测0.805的整体ECE仅0.002，却完全不能区分好坏（AUC0.5）。两个参数的单调方差校准整体Brier0.140/AUC0.730，优于六个新小头Brier约0.153–0.156/AUC0.622–0.682。真实底视脚仅24个可评价点，结果不稳；不能据此部署门控。</p>',
        '<p>神经对照同初始化/采样/800步/3种子，scalar屏蔽semantic输入。名义参数同9697，有效参数scalar1473、semantic9697，容量并不相同。BCE按图内已知点平均，再按8真实/4单鼠合成/4双鼠合成采样。简单基线按同一期望目标精确拟合，只用fit；是首轮失败后补充的诊断，不伪称事前注册。</p>',
        '<h2>开发指标</h2><table><tr><th>范围</th><th>方法</th><th>已知/未知</th><th>AUC↑</th><th>Brier↓</th><th>ECE↓</th></tr>']
    for kind in ['all','real','single','dual','real_bottom_feet']:
        for name in ['objectness','inverse_std_rank','balanced_fit_constant','monotonic_std','scalar_seed17','semantic_seed17','scalar_seed29','semantic_seed29','scalar_seed43','semantic_seed43']:
            m=results[name]['validation'][kind];fmt=lambda x:'—' if x is None else f'{x:.3f}'
            page.append(f'<tr><td>{kind}</td><td>{name}</td><td>{m["known"]}/{m["unknown"]}</td><td>{fmt(m["auc"])}</td><td>{fmt(m["brier"])}</td><td>{fmt(m["ece"])}</td></tr>')
    page+=['</table><p>未知数指可信检测内缺失点或身份不能可靠评价者，不计正确也不计错误。真实142已知点（底脚24）比此前175（底脚32）少，因为本轮新增框IoU≥0.5的身份资格，不是标注减少；不能直接比较两个子集的定位均值。热图方差只作排名时不是概率，因此不报Brier/ECE。</p>',
        '<h2>训练/内部校准曲线</h2><p>semantic继续改善fit与真实校准，却明显恶化整段爬墙校准；不是全面未收敛。暂不增加步数，也不按开发集挑最好种子或epoch。</p><img src="quality_brier_curves.png">',
        '<h2>风险—覆盖率</h2><p>横轴在全部预测可信点中保留的实际比例，未知点仍参与排序；纵轴仅统计其中已知标签点的平均误差。低覆盖的小样本波动不能当作安全保证；分数相同者一起保留，常数预测不会虚构少量优选。</p><img src="quality_risk_coverage.png">',
        '<h2>高置信却错误的脚点</h2><p>各范围事后取单调校准分数最高的错误脚点（误差&gt;32px），用于找漏洞，不代表发生频率。所有红色坐标仍是原模型输出，绿色仅作评价。</p><img src="confident_wrong_feet.jpg">',
        '<h2>后续</h2><p>保留单调方差校准作为轻量参考，不自动用于修改坐标或生成新真值。停止重复相似的小头试验；下一步先审计当前冻结编码器、训练数据覆盖与既有适配路线，判断是否该从像素特征适配/更丰富的训练姿态入手。原先方差加权三角化已有负结果，不能把换一层校准包装成已验证的融合改进。</p>',
        '<p><a href="results.json">全部结果与简单校准参数</a> | <a href="input_manifest.json">划分与输入来源</a> | <a href="visual_examples.json">图例对应记录</a></p>']
    (OUT/'index.html').write_text('\n'.join(page),encoding='utf-8');shutil.copy2(CYCLE/'point_quality_inputs_v1/manifest.json',OUT/'input_manifest.json')
    for path in OUT.iterdir():
        if path.is_file():shutil.copy2(path,DEST/path.name)
    print(DEST/'index.html')


if __name__=='__main__':main()

