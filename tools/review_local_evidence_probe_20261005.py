"""Raw-image review of query-independent candidate maps, no new inference."""
import sys,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import torch,numpy as np
from PIL import Image,ImageDraw
from review_pixel_adapter_20261005 import font
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';RUN=CYCLE/'local_evidence_probe_v1';INPUT=CYCLE/'highres_preflight_inputs_v1';OUT=CYCLE/'local_evidence_review_v1';DEST=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005')/OUT.name
def read(p):return torch.load(p,weights_only=False,map_location='cpu')
def main():
    global RUN,INPUT,OUT,DEST
    rotation4000='--rotation4000' in sys.argv
    rotation='--rotation' in sys.argv or rotation4000
    pool_control='--pool-control' in sys.argv or rotation
    mixed='--mixed' in sys.argv or pool_control
    if mixed:
        RUN=CYCLE/'mixed_local_backbone800_v1';INPUT=CYCLE/'mixed_local_rgb_inputs_v1';OUT=CYCLE/'mixed_local_backbone800_review_v1';DEST=DEST.parent/OUT.name
    if pool_control:
        RUN=CYCLE/'mixed_real_pool_control_v1';OUT=CYCLE/'mixed_real_pool_review_v1';DEST=DEST.parent/OUT.name
    if rotation:
        RUN=CYCLE/'mixed_rotation_control_v1';OUT=CYCLE/'mixed_rotation_review_v1';DEST=DEST.parent/OUT.name
    if rotation4000:
        RUN=CYCLE/'mixed_rotation4000_v1';OUT=CYCLE/'mixed_rotation4000_review_v1';DEST=DEST.parent/OUT.name
    OUT.mkdir(exist_ok=True);DEST.mkdir(exist_ok=True);result=json.loads((RUN/'results.json').read_text());curves=json.loads((RUN/'curves.json').read_text());ev={r['index']:r for r in read(INPUT/'evaluation_only.pt')};manifest={r['index']:r for r in json.loads((INPUT/'inference_manifest.json').read_text())}
    if rotation4000:
        preceding=json.loads((CYCLE/'mixed_rotation_control_v1/curves.json').read_text());curves=list({(z['arm'],z['step']):z for z in preceding+curves}.values());curves.sort(key=lambda z:(z['arm'],z['step']))
    if mixed:
        arms=({'native':'native_rgb','rotated':'quarter_turn'} if rotation else {'same':'same_pool','expanded':'expanded_real'}) if pool_control else {'frozen':'frozen_official_backbone','adapted':'train_official_backbone'};pred={short:read(RUN/(arm+'_predictions.pt')) for short,arm in arms.items()};result['summary']={short:result['summary'][arm] for short,arm in arms.items()};curves=[dict(z,arm=next(short for short,arm in arms.items() if arm==z['arm'])) for z in curves];vis={i:dict(masks=dict(visible=r['coordinate_valid']&r['visibility_valid']&r['visible'])) for i,r in ev.items()};ev={i:dict(r,gt=r['target'],valid=r['coordinate_valid']) for i,r in ev.items()}
    else:vis=read(CYCLE/'highres_visibility_audit_v1/evaluation_visibility.pt');pred=read(RUN/'predictions.pt')
    for key in ['loss_mean','nearest_top6_mean']:
        fig,axes=plt.subplots(2,3,figsize=(15,8))
        for row,stage in enumerate(['fit','calibration']):
            for col,kind in enumerate(['real','single','dual']):
                ax=axes[row,col]
                for arm in pred:
                    cc=[z for z in curves if z['arm']==arm];ax.plot([z['step'] for z in cc],[z['metrics'][stage][kind]['eligible'][key] for z in cc],label=arm,marker='o')
                ax.set_title(stage+' '+kind);ax.set_ylabel(key);ax.legend()
        fig.tight_layout();fig.savefig(OUT/(key+'.png'),dpi=140);plt.close(fig)
    colors=['#ff4422','#00bbff','#ee44dd','#99ff33','#ffbb00'];ids=[1,15,16,17,18];selected=[]
    for stage in ['fit','calibration']:
        for kind in ['real','single','dual']:
            i=next(i for i,e in ev.items() if e['metadata']['stage']==stage and e['metadata']['kind']==kind and 'probability_map' in pred[next(iter(pred))][i]);e=ev[i];im=Image.open(ROOT/manifest[i]['image']).convert('RGB');w,h=im.size;tile=Image.new('RGB',(2220,1840),'white');draw=ImageDraw.Draw(tile);eligible=e['valid'] if kind=='real' else vis[i]['masks']['visible']
            panels=[]
            arm0,arm1=pred.keys()
            for name in ['GT_eligible_only',arm0+'_top6',arm1+'_top6','raw_RGB',arm0+'_heat',arm1+'_heat']:
                p=im.copy();d=ImageDraw.Draw(p)
                if name=='GT_eligible_only':
                    for q,k in eligible.nonzero().tolist():
                        x,y=(e['gt'][q,k]*e['pixels']).tolist();d.line((x-6,y,x+6,y),fill=colors[k],width=3);d.line((x,y-6,x,y+6),fill=colors[k],width=3);d.text((x+6,y),str(ids[k]),font=font(14),fill=colors[k])
                elif name.endswith('_top6'):
                    arm=name.split('_')[0];a=pred[arm][i]
                    for k in range(5):
                        for rank in range(6):
                            if not a['available'][k,rank]:continue
                            x,y=(a['xy'][k,rank]*e['pixels']).tolist();radius=5 if rank==0 else 2;d.ellipse((x-radius,y-radius,x+radius,y+radius),outline=colors[k],width=2)
                            if rank==0:d.text((x+6,y),str(ids[k]),font=font(12),fill=colors[k])
                elif name.endswith('_heat'):
                    arm=name.split('_')[0];a=pred[arm][i]['probability_map'].sum(0).numpy();a=np.log1p(a/max(float(a.max()),1e-12)*100)/np.log(101);heat=np.asarray(Image.fromarray((plt.get_cmap('inferno')(a)[...,:3]*255).astype('uint8')).resize((a.shape[1]*4,a.shape[0]*4),Image.Resampling.BILINEAR).crop((0,0,w,h)));p=Image.fromarray((np.asarray(im)*.55+heat*.45).astype('uint8'))
                panels.append((name,p))
            for slot,(name,p) in enumerate(panels):
                row,col=divmod(slot,3);p.thumbnail((720,795));tile.paste(p,(col*740+(740-p.width)//2,row*900+90));draw.text((col*740+12,row*900+12),name,font=font(24),fill='black');draw.text((col*740+12,row*900+48),f'{stage} {kind} index{i}',font=font(20),fill='black')
            draw.text((10,1810),'Heat: per-panel normalized log-sum probability, not confidence. Fixed6 peaks/class; no instance assignment.',font=font(21),fill='black');name=f'{stage}_{kind}_{i}.jpg';tile.save(OUT/name,quality=95);selected.append(dict(index=i,metadata=e['metadata'],file=name))
    (OUT/'visual_examples.json').write_text(json.dumps(selected,indent=2)+'\n');(OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n');shutil.copy2(RUN/'protocol.json',OUT/'protocol.json')
    if not mixed:shutil.copy2(RUN/'completion_verification.json',OUT/'completion_verification.json')
    page=['<!doctype html><meta charset="utf-8"><title>局部关键点证据诊断</title><style>body{font:17px system-ui;margin:30px}img{max-width:100%}td,th{border:1px solid #aaa;padding:7px}table{border-collapse:collapse}</style><h1>去掉实例查询后，特征能找到足部吗？</h1><p>同42图/300步/19,973参数浅层卷积/初始化/样本顺序，对比两种冻结HRNet特征。前向不接收GT框、数量、查询或数据域。合成训练仅用明确可见点；真实点可见性未知，不改写标签。不是最终姿态模型、不是端到端AP。</p><p>原始单鼠微调特征在真实内部校准中Top1约2.44px，但合成单鼠约75.80px、双鼠候选覆盖约249.27px。原始四足预训练特征在双鼠内部校准候选覆盖约21.25px，但真实/合成单鼠失败。当前局部证据本身存在明显域差异，不能只归因于实例query分配。关闭固定浅层特征探针，不将其加入主模型；接下来考虑有可验证保留约束的混合域视觉训练。</p><p>两个300步训练及结果均保存；结束时路径拼接的只读哈希校验报错，已修正并独立验证84个缓存哈希、所有曲线/端点/概率。未重跑训练，原失败记录保留。</p><img src="loss_mean.png"><img src="nearest_top6_mean.png"><table><tr><th>特征</th><th>阶段</th><th>组</th><th>N</th><th>最近Top6↓</th><th>一对一Top6↓</th><th>单实例Top1↓</th><th>8px概率质量↑</th></tr>']
    if mixed:
        page=['<!doctype html><meta charset="utf-8"><title>混合域视觉骨干对照</title><style>body{font:17px system-ui;margin:30px}img{max-width:100%}td,th{border:1px solid #aaa;padding:7px}table{border-collapse:collapse}</style><h1>真实 + 合成：冻结骨干与全骨干训练</h1><p>同273张FIT/89张内部校准、同官方四足动物HRNetW32初始化、同新五类局部头、同800步/样本顺序/目标/PyTorch2.7.1运行环境。唯一训练配置差异是是否训练29,305,536个骨干参数；BN运行统计始终冻结。更多参数/计算不是等算力消融。原数据、checkpoint和主模型默认未改。</p><p>只评估鼻尖和四足局部候选，不处理实例关联。前向不接收GT框、数量、查询或数据域。合成仅明确可见点参与本诊断训练，真实坐标可见性未知；缺失标签忽略。内部校准不是独立测试，未读取开发验证/锁定序列。</p><img src="loss_mean.png"><img src="nearest_top6_mean.png"><table><tr><th>方法</th><th>阶段</th><th>组</th><th>N</th><th>最近Top6↓</th><th>一对一Top6↓</th><th>单实例Top1↓</th><th>8px概率质量↑</th></tr>']
    if pool_control:
        page=['<!doctype html><meta charset="utf-8"><title>真实训练数据覆盖对照</title><style>body{font:17px system-ui;margin:30px}img{max-width:100%}td,th{border:1px solid #aaa;padding:7px}table{border-collapse:collapse}</style><h1>多训练还是多样本：匹配续训对照</h1><p>同一800步模型和AdamW状态出发，两组均追加800步到1600；相同末端学习率、合成采样、底视/非底视400/400次。仅真实FIT候选池128→821张；本次实际抽到128/406张不同图。评价集保持原273FIT/89内部校准，无新验证图。不是独立外部测试。</p><p>same＝沿用128张；expanded＝从821张池采样。前向不接收GT框、数量、查询或数据域。两组均训练全骨干和局部五类头、冻结BN运行统计。候选覆盖不等于最终实例姿态，未替换主模型。</p><img src="loss_mean.png"><img src="nearest_top6_mean.png"><table><tr><th>方法</th><th>阶段</th><th>组</th><th>N</th><th>最近Top6↓</th><th>一对一Top6↓</th><th>单实例Top1↓</th><th>8px概率质量↑</th></tr>']
    if rotation:
        page=['<!doctype html><meta charset="utf-8"><title>方向增强对照</title><style>body{font:17px system-ui;margin:30px}img{max-width:100%}td,th{border:1px solid #aaa;padding:7px}table{border-collapse:collapse}</style><h1>像素精确旋转：方向依赖控制实验</h1><p>两组同一800步模型和AdamW状态，追加同800步；使用相同821张真实FIT候选池、相同406张实际抽样、相同合成采样和底视比例。native没有增强；rotated固定种子43逐图选择0/90/180/270°旋转。无镜像、插值或左右标签交换。全骨干训练、BN统计冻结，学习率和监督不变。</p><p>评估始终使用未旋转的同一362图，其中89张为反复使用的内部校准，非独立测试。该模型只预测鼻尖和四足候选，不是最终多实例姿态模型。未变更正式模型或原始数据。</p><img src="loss_mean.png"><img src="nearest_top6_mean.png"><table><tr><th>方法</th><th>阶段</th><th>组</th><th>N</th><th>最近Top6↓</th><th>一对一Top6↓</th><th>单实例Top1↓</th><th>8px概率质量↑</th></tr>']
    if rotation4000:
        page=['<!doctype html><meta charset="utf-8"><title>方向增强4000步对照</title><style>body{font:17px system-ui;margin:30px}img{max-width:100%}td,th{border:1px solid #aaa;padding:7px}table{border-collapse:collapse}</style><h1>方向增强：匹配训练轨迹至4000步</h1><p>两组共同祖先为原800步模型，此后增强开/关各训练3200步到4000。1600→4000这一段分别保留各自模型及AdamW状态；不是相同1600初始化。新增2400步使用相同样本顺序、学习率、底视比例和监督，真实821张池本段实际抽到681张，旋转种子71；不更改架构。无镜像、插值或左右标签交换，全骨干训练、BN统计冻结。</p><p>曲线含前段800→1600及本段1600→4000，表格为预定4000终点，不选择最优中间步。评估仍为同一未增强362图，其中89张反复使用的内部校准，不是外部测试。仅鼻尖/四足候选，无最终实例关联，不改变正式模型或标注。</p><img src="loss_mean.png"><img src="nearest_top6_mean.png"><table><tr><th>方法</th><th>阶段</th><th>组</th><th>N</th><th>最近Top6↓</th><th>一对一Top6↓</th><th>单实例Top1↓</th><th>8px概率质量↑</th></tr>']
    for arm,stages in result['summary'].items():
        for stage,kinds in stages.items():
            for kind,d in kinds.items():
                z=d['eligible'];page.append('<tr>'+''.join('<td>'+str(x)+'</td>' for x in [arm,stage,kind,z['n']])+''.join('<td>'+('—' if z[k] is None else f'{z[k]:.3f}')+'</td>' for k in ['nearest_top6_mean','unique_top6_mean','top1_single_mean','mass8_mean'])+'</tr>')
    page.append('</table><p>Top6是固定候选集合，最近距离使用真值作事后评价，允许多个点落入同一候选，是乐观覆盖率而非实例定位精度；同时报告一对一候选评价。所有坐标均为输入像素。内部校准并非外部测试。</p><h2>每组首张图，未按误差挑选</h2><p>点颜色表示部位，不表示实例；大圆为Top1，其余小圆为Top2–6。GT面板仅画本诊断采用的坐标，不进入前向。</p>')
    for z in selected:page.append('<h3>'+z['metadata']['stage']+' '+z['metadata']['kind']+'</h3><img src="'+z['file']+'">')
    if mixed and not pool_control:page.append('<h2>800步结论与下一项对照</h2><p>冻结/训练骨干的内部校准最近Top6误差：真实184.38/52.46、合成单鼠58.92/9.76、双鼠74.51/15.64px；单实例Top1真实247.05/92.70、合成单鼠81.24/20.65px。骨干训练有效，但真实左右足混淆与玻璃误响应仍明显，不能作为合格最终模型。400→800真实校准改善较慢，训练/校准差距扩大。下一项从同一800步模型和优化器出发，追加相同800步，对比真实训练池128张与821张；固定底视比例和合成采样，以区分步数与样本覆盖，不改校准集。</p>')
    page.append('<a href="results.json">完整结果</a> | <a href="protocol.json">协议</a>'+('' if mixed else ' | <a href="completion_verification.json">完成校验</a>'));(OUT/'index.html').write_text('\n'.join(page),encoding='utf-8')
    for p in OUT.iterdir():
        if p.is_file():shutil.copy2(p,DEST/p.name)
    print(json.dumps(selected,indent=2))
if __name__=='__main__':main()
