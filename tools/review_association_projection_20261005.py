"""Review convergence, identity association and actual accepted image groups."""
import json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from run_depth_pose_ablation import write_json
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';SOURCE=CYCLE/'association_projection_pilot_v1';ASSOC=CYCLE/'learned_partial_association_v1'
OUT=CYCLE/'association_projection_review_v1';DEST=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005')/OUT.name
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def font(n):return ImageFont.truetype(FONT,n)


def curves_plot(curves):
    im=Image.new('RGB',(1400,1080),'white');draw=ImageDraw.Draw(im);colors={'rgb/fit':'#155aab','rgb/calibration':'#e23d26','gray/fit':'#16a878','gray/calibration':'#953da8'}
    for ri,seed in enumerate([17,29,43]):
        for ci,(metric,lo,hi) in enumerate([('contrastive_loss',0,3.6),('same_identity_auc',.35,1.02)]):
            x0=75+ci*700;y0=58+ri*345;w=585;h=250
            draw.text((x0,y0-32),f'Seed {seed}: {metric}',font=font(19),fill='black')
            for t in np.linspace(lo,hi,5):
                y=y0+h-(t-lo)/(hi-lo)*h;draw.line((x0,y,x0+w,y),fill='#ddd');draw.text((x0-52,y-7),f'{t:.2f}',font=font(13),fill='black')
            for key,color in colors.items():
                arm,split=key.split('/');rs=[r for r in curves if r['seed']==seed and r['arm']==arm and r['split']==split]
                points=[(x0+r['step']/800*w,y0+h-(r['metrics'][metric]-lo)/(hi-lo)*h) for r in rs]
                draw.line(points,fill=color,width=3)
            for step in [0,100,400,800]:draw.text((x0+step/800*w-10,y0+h+5),str(step),font=font(13),fill='black')
    for i,(key,color) in enumerate(colors.items()):draw.text((70+i*325,1040),key,font=font(20),fill=color)
    im.save(OUT/'fit_calibration_curves.png')


def group_panel(row,g,name):
    im=Image.new('RGB',(1536,984),'white');draw=ImageDraw.Draw(im);palette=['#00cf91','#e891ff','#ffff00']
    draw.text((16,12),f'{row["model"]} / {row["mode"]} / scene {row["group_index"]}',font=font(22),fill='black')
    draw.text((16,43),'Box colors = accepted association groups; gray = rejected. ID label is evaluation only.',font=font(18),fill='black')
    assigned={n:i for i,group in enumerate(row['groups']) for n in group['nodes']}
    for vi,view in enumerate(g['record']['views']):
        x=vi%2*768;y=78+vi//2*453;img=Image.open(CYCLE/'dual_allscene_association_audit_v1'/g['images'][vi]).convert('RGB')
        assert img.size==(768,432);im.paste(img,(x,y+21));draw.text((x+5,y),f'physical view {view}',font=font(16),fill='black')
        for ni,node in enumerate(row['node_list']):
            if node['view']!=view or ni not in row['node_indices']:continue
            q=node['query'];cx,cy,bw,bh=g['predictions']['boxes'][vi,q].tolist();rect=(x+(cx-bw/2)*768,y+21+(cy-bh/2)*432,x+(cx+bw/2)*768,y+21+(cy+bh/2)*432)
            color=palette[assigned[ni]%len(palette)] if ni in assigned else '#9b9b9b';draw.rectangle(rect,outline=color,width=4)
            identity=row['identity_diagnostic'][ni];label=f'q{q} ID={identity if identity is not None else "?"} '+(f'G{assigned[ni]}' if ni in assigned else 'reject')
            tx=max(x,min(x+600,rect[0]));ty=max(y+21,min(y+420,rect[1]-23 if q%2 else rect[3]+2));draw.rectangle((tx,ty,tx+165,ty+22),fill='black');draw.text((tx+3,ty+2),label,font=font(15),fill=color)
    im.save(OUT/name,quality=94)


def main():
    OUT.mkdir(exist_ok=True);DEST.mkdir(exist_ok=True);r=json.loads((SOURCE/'results.json').read_text());curves=json.loads((SOURCE/'curves.json').read_text());a=json.loads((ASSOC/'results.json').read_text());p=json.loads((SOURCE/'protocol.json').read_text())
    data=torch.load(SOURCE/'inputs.pt',map_location='cpu',weights_only=False);d=data['train'];floors=[]
    for i in p['fit_indices']:
        labels=d['labels'][i];views=d['views'][i];same=labels[:,None]==labels[None,:];known=(labels>=0)[:,None]&(labels>=0)[None,:]
        pos=(same&known&(views[:,None]!=views[None,:])).sum(-1);neg=(~same&known).sum(-1);active=(pos>0)&(neg>0)
        floors.append(float(pos[active].float().log().mean()))
    audit=dict(theoretical_fit_contrastive_lower_bound=float(np.mean(floors)),rgb_gray_feature_max_delta=float((d['features']['rgb']-d['features']['gray']).abs().max()),
        same_feature_storage=d['features']['rgb'].data_ptr()==d['features']['gray'].data_ptr(),
        explanation='Multiple positive anchors impose mean(log positive-count) loss floor. Similar final RGB/gray fitting losses do not mean aliased features; final fit AUC=1, calibration worsens. No extra training warranted.')
    assert not audit['same_feature_storage'] and audit['rgb_gray_feature_max_delta']>0
    write_json(OUT/'convergence_audit.json',audit);curves_plot(curves)
    groups=torch.load(CYCLE/'dual_allscene_association_audit_v1/groups.pt',map_location='cpu',weights_only=False)
    candidates=[x for x in a['details'] if x['role']=='validation'];examples=[]
    # Declared post-hoc illustration categories, not checkpoint/threshold selection.
    queries=[('accepted_rgb',lambda x:x['model']=='rgb_seed17_trained' and x['mode']=='natural' and x['groups']),
        ('unknown_rgb',lambda x:x['model']=='rgb_seed43_trained' and x['mode']=='natural' and any(g['status']=='unknown' for g in x['groups'])),
        ('fault_first_scene',lambda x:x['model']=='rgb_seed17_trained' and x['mode']=='highest_confidence' and x['group_index']==0),
        ('gray_first_scene',lambda x:x['model']=='gray_seed17_trained' and x['mode']=='natural' and x['group_index']==0)]
    for name,predicate in queries:
        row=next(x for x in candidates if predicate(x));file=name+'.jpg';group_panel(row,groups[row['group_index']],file);examples.append(dict(file=file,row=row))
    write_json(OUT/'visual_examples.json',examples)
    page=['<!doctype html><meta charset="utf-8"><title>关联投影与保守分组</title><style>body{font:17px system-ui;max-width:1500px;margin:28px auto;padding:12px}td,th{padding:7px;border:1px solid #bbb}table{border-collapse:collapse}img{max-width:100%}small{color:#555}</style>',
        '<h1>小型关联头：有改善，但不足以安全全面融合</h1><p>冻结主检测器、姿态头和图像骨干，仅训练1024→64的线性投影（65,536参数）。RGB/灰度分别3种子、800步，同初始化及采样；16训练场景+5训练内校准场景。最终统一查看16个开发场景。全部6个预测槽位保留，裁剪由预测框产生；GT身份只用于训练loss和事后评价，不进入特征前向。</p>',
        '<p>来源已核对：TRAIN动作视频021709_A25_Block22_BCfe1_t，开发动作视频021309_A29_Block16_BCma1_t，视频源分离；仍共用Blender场景/模型/毛色/相机。内部5场景与训练同源，主姿态模型曾见过它们。这不是独立真实视频或同毛色验证。</p>',
        '<h2>收敛：不继续加步数</h2><p>约100步后拟合集身份AUC已接近1；训练loss达到多正样本对比损失的理论下界附近，校准loss随后变差。RGB与灰度末期训练loss几乎相同是拟合饱和，并非输入误复用。开发集没有参与选步数；本报告仍使用预先固定的800步终点，不改挑最好的一步。</p>',
        f'<p>拟合loss理论下界 {audit["theoretical_fit_contrastive_lower_bound"]:.6f}；RGB/灰度缓存最大差 {audit["rgb_gray_feature_max_delta"]:.3f}，存储独立。六次训练总耗时约{sum(x["seconds"] for x in r["runs"]):.1f}秒（仅小头缓存训练，不含先前裁剪编码）。</p><img src="fit_calibration_curves.png">',
        '<h2>开发集身份关联（未知项不计正确，也不隐藏）</h2><table><tr><th>种子</th><th>输入</th><th>阶段</th><th>身份AUC</th><th>互选正确/已知</th><th>未知/全部互选</th></tr>']
    for row in r['results']:
        if row['split']!='validation':continue
        if row['phase']=='raw' and row['seed']!=17:continue
        m=row['metrics'];u=m['mutual'];page.append(f'<tr><td>{row["seed"]}</td><td>{row["arm"]}</td><td>{row["phase"]}</td><td>{m["same_identity_auc"]:.3f}</td><td>{u["correct"]}/{u["known"]}</td><td>{u["unknown"]}/{u["proposed"]}</td></tr>')
    page+=['</table><h2>保守分组：先冻结校准阈值，再评价开发和故障注入</h2><p>5个校准场景仅43条已知异鼠边。预先声明1%经验负边预算取整为0，以严格小于最小负边距离为阈值；再要求跨相机互选、至少3相机完整组，并拒绝重叠组。所有未知节点依旧参加推理。不是“已证明错误率低于1%”：样本小且边相关。</p><table><tr><th>模型 / 场景</th><th>组数</th><th>全已知同鼠</th><th>已知混鼠</th><th>含未知</th><th>已分配节点</th></tr>']
    for key,m in a['summary'].items():
        if '/validation/' not in key:continue
        page.append(f'<tr><td>{key}</td><td>{m["groups"]}</td><td>{m["pure_known"]}</td><td>{m["mixed"]}</td><td>{m["unknown"]}</td><td>{m["assigned_nodes"]}/{m["nodes"]}</td></tr>')
    page+=['</table><p>RGB自然场景仅覆盖15–32/128个节点。未观察到已知混组，不意味着所有未知组正确。灰度覆盖更低；去色不去亮度，也不能证明同毛色泛化。故障注入仅屏蔽logit、模拟每相机只保留一个检测，不是真实遮挡发生率。原严格几何控制及原始裁剪余弦都没有接受3视角组。本次未改任何姿态坐标、生产默认或标注。</p><h2>实际图像复核</h2><p>下图按预先说明的示例类别事后选取：首个接受RGB组、首个含未知RGB组、第一场景故障拒绝、第一场景灰度拒绝。不是随机性能样本；完整数量以上表为准。框颜色标识关联组而非query身份。</p>']
    for e in examples:page.append(f'<h3>{e["file"]}</h3><img src="{e["file"]}">')
    page+=['<h2>下一步</h2><p>不延长拟合，不放宽开发集阈值。固定历史seed17的已冻结规则，做关联驱动、默认关闭的离线候选融合对照；保留未关联点原样，分别检查原始/局部/带关联融合及人工故障，尤其检查足点跨鼠跳跃和未知身份组。收益不足则保留为负结果，不部署。</p><p><a href="training_results.json">训练结果</a> | <a href="association_results.json">全部分组明细</a> | <a href="calibration.json">冻结校准</a> | <a href="visual_examples.json">示例来源</a></p>']
    (OUT/'index.html').write_text('\n'.join(page),encoding='utf-8')
    for src,name in [(SOURCE/'results.json','training_results.json'),(SOURCE/'curves.json','curves.json'),(ASSOC/'results.json','association_results.json'),(ASSOC/'calibration.json','calibration.json')]:shutil.copy2(src,OUT/name)
    for path in OUT.iterdir():
        if path.is_file():shutil.copy2(path,DEST/path.name)
    print(DEST/'index.html');print(json.dumps(audit))


if __name__=='__main__':main()
