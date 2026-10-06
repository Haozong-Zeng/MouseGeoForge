"""Reusable small-cohort high-resolution control report (no new inference)."""
import sys,json,shutil,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import torch
from PIL import Image,ImageDraw
from review_pixel_adapter_20261005 import font
from mousegeoforge.models.highres_instance_readout import KP
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';INPUT=CYCLE/'highres_preflight_inputs_v1'
def read(p):return torch.load(p,weights_only=False,map_location='cpu')
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',default='highres_unit_control_v1');parser.add_argument('--output',default='highres_unit_review_v1');parser.add_argument('--control',default='frozen_unit');parser.add_argument('--candidate',default='adapt_unit');parser.add_argument('--official',action='store_true');args=parser.parse_args()
    if args.official:
        args.source='official_quadruped_readout_pilot_v1';args.output='official_quadruped_readout_review_v1';args.control='finetuned_features';args.candidate='official_features'
    source=CYCLE/args.source;out=CYCLE/args.output;dest=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005')/out.name;out.mkdir(exist_ok=True);dest.mkdir(exist_ok=True)
    result=json.loads((source/'results.json').read_text());protocol=json.loads((source/'protocol.json').read_text());pred=read(source/'predictions.pt');ev={r['index']:r for r in read(INPUT/'evaluation_only.pt')};inf={r['index']:r for r in read(INPUT/'inference.pt')};manifest={r['index']:r for r in json.loads((INPUT/'inference_manifest.json').read_text())};arms=[args.control,args.candidate];curves=json.loads((source/'curves.json').read_text());fig,axs=plt.subplots(2,3,figsize=(15,8))
    if args.official:
        old=CYCLE/'highres_preflight_pilot_v1';old_result=json.loads((old/'results.json').read_text());old_pred=read(old/'predictions.pt');old_protocol=json.loads((old/'protocol.json').read_text())
        assert protocol['schedule']==old_protocol['schedule'] and protocol['initial_sha256']==old_protocol['initial_sha256']
        result=dict(summary={'baseline':result['summary']['baseline'],args.control:old_result['summary']['appearance'],args.candidate:result['summary']['appearance']},fit_gate=result['fit_gate'],performance=result['performance'])
        pred={'baseline':pred['baseline'],args.control:old_pred['appearance'],args.candidate:pred['appearance']}
        curves=[dict(z,arm=args.candidate) for z in curves if z['arm']=='appearance']+[dict(z,arm=args.control) for z in json.loads((old/'curves.json').read_text()) if z['arm']=='appearance']
    for row,stage in enumerate(['fit','calibration']):
        for col,kind in enumerate(['real','single','dual']):
            ax=axs[row,col]
            for arm in arms:
                cc=[z for z in curves if z['arm']==arm];ax.plot([z['step'] for z in cc],[z['metrics'][stage][kind]['mean'] for z in cc],label=arm,marker='o')
            ax.axhline(result['summary']['baseline'][stage][kind]['mean'],color='gray',ls='--');ax.set_title(stage+' '+kind);ax.set_ylabel('input pixel error');ax.legend()
    fig.tight_layout();fig.savefig(out/'curves.png',dpi=140);plt.close(fig);selected=[];colors=['#ff7744','#00bbff','#ff77dd','#aaee44','#ffaa00','#7777ff']
    for kind in ['real','single','dual']:
        entries=[]
        for i,r in ev.items():
            if r['metadata']['stage']!='calibration' or r['metadata']['kind']!=kind:continue
            errors={a:float(((p[i][:,KP]-r['gt'])*r['pixels']).norm(dim=-1)[r['valid']].mean()) for a,p in pred.items()};entries.append(dict(index=i,kind=kind,metadata=r['metadata'],errors=errors,delta=errors[args.candidate]-errors['baseline']))
        z=max(entries,key=lambda z:z['delta']);i=z['index'];r=ev[i];im=Image.open(ROOT/manifest[i]['image']).convert('RGB');w,h=im.size;tile=Image.new('RGB',(2960,960),'white');d=ImageDraw.Draw(tile)
        for col,name in enumerate(['baseline',*arms,'GT_evaluation_only']):
            panel=im.copy();di=ImageDraw.Draw(panel)
            for q,b in enumerate(inf[i]['boxes']):
                cx,cy,bw,bh=(b*torch.tensor([w,h,w,h])).tolist();di.rectangle((cx-bw/2,cy-bh/2,cx+bw/2,cy+bh/2),outline=colors[q],width=1)
                for k,kp in enumerate(KP):
                    if name=='GT_evaluation_only':
                        if not r['valid'][q,k]:continue
                        x,y=(r['gt'][q,k]*r['pixels']).tolist();di.line((x-5,y,x+5,y),fill='#00ff66',width=3);di.line((x,y-5,x,y+5),fill='#00ff66',width=3)
                    else:
                        x,y=(pred[name][i][q,kp]*r['pixels']).tolist();di.ellipse((x-3,y-3,x+3,y+3),outline=colors[q],width=2)
                    if r['valid'][q,k]:di.text((x+5,y),str(kp+1),font=font(13),fill='#00ff66' if name=='GT_evaluation_only' else colors[q])
            panel.thumbnail((720,795));tile.paste(panel,(col*740+(740-panel.width)//2,100));d.text((col*740+10,10),name,font=font(24),fill='black');d.text((col*740+10,50),f'CAL {kind} worst change',font=font(20),fill='black')
        d.text((15,915),'All6 queries; GT for evaluation only; no development validation.',font=font(23),fill='black');z['file']=f'{kind}_worst_{i}.jpg';tile.save(out/z['file'],quality=95);selected.append(z)
    write(out/'visual_examples.json',selected);write(out/'results.json',result);write(out/'protocol.json',protocol)
    page=['<!doctype html><meta charset="utf-8"><title>视觉适配同预算对照</title><style>body{font:17px system-ui;margin:30px}img{max-width:100%}td,th{border:1px solid #aaa;padding:7px}table{border-collapse:collapse}</style><h1>高分辨率视觉适配：冻结对照与学习分支</h1><p>同一24张FIT/18张内部校准、同初始化/300步/样本顺序/目标，在同一PyTorch运行环境比较。原预测框/查询和另外18点不变，GT仅用于独立训练损失与评价，未读取开发验证。更多可训练参数与计算量见protocol/results；不是等参数架构消融。BN运行统计固定，原checkpoint不变。</p><p>结果：适配末级视觉单元改善拟合，真实和合成单鼠通过原定拟合门槛，双鼠未通过；合成校准明显退步，呈现过拟合，不能晋升或直接加训。关闭当前窄域教师末级适配支线；后续先审计本地现存的原始SuperAnimal预训练骨干，区别通用预训练与小鼠单域微调后的特征。</p><img src="curves.png"><table><tr><th>阶段</th><th>方法</th><th>组</th><th>mean↓</th><th>PCK32↑</th></tr>']
    if args.official:
        page=['<!doctype html><meta charset="utf-8"><title>原始四足动物预训练骨干对照</title><style>body{font:17px system-ui;margin:30px}img{max-width:100%}td,th{border:1px solid #aaa;padding:7px}table{border-collapse:collapse}</style><h1>原始 SuperAnimal Quadruped 与单鼠微调特征</h1><p>同24张FIT/18张内部校准、同300步/初始化/样本顺序/29,792参数读出头/目标/PyTorch2.11环境，仅替换冻结的32通道特征。未使用预训练关键点头作为Core23标注，未读取开发验证。预训练经历不同，不是架构消融；内部校准不等于独立测试。</p><p>结果：原始预训练特征解决本小样本中的全零响应，但三组均未通过原定拟合门槛；真实校准显著退步，合成单鼠好于微调特征但仍差于原模型，双鼠未获益。不扩大此固定读出头的训练，不晋升生产；接下来按已知可见性审计误差。</p><img src="curves.png"><table><tr><th>阶段</th><th>方法</th><th>组</th><th>mean↓</th><th>PCK32↑</th></tr>']
    for stage in ['fit','calibration']:
        for a,s in result['summary'].items():
            for kind in ['real','single','dual','real_bottom_feet']:
                t=s[stage][kind];page.append(f'<tr><td>{stage}</td><td>{a}</td><td>{kind}</td><td>{t["mean"]:.3f}</td><td>{t["pck32"]:.3f}</td></tr>')
    page.append('</table><h2>每组内部校准最差变化</h2><p>最差也可能仍为改进；极值不是发生率，内部校准曾被底层模型部分见过，不是外部测试。</p>')
    for z in selected:page.append(f'<h3>{z["kind"]}: {z["delta"]:+.2f}px</h3><img src="{z["file"]}">')
    page.append('<a href="results.json">完整结果</a> | <a href="protocol.json">预算与配置</a>');(out/'index.html').write_text('\n'.join(page),encoding='utf-8')
    for p in out.iterdir():
        if p.is_file():shutil.copy2(p,dest/p.name)
    print(json.dumps(selected,indent=2))
if __name__=='__main__':main()
