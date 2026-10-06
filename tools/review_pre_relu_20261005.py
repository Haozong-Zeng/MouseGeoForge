"""Matched pre/post-ReLU feature tap review without new evaluation inputs."""
import sys,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import torch,numpy as np
from PIL import Image,ImageDraw
from review_pixel_adapter_20261005 import font
from mousegeoforge.models.highres_instance_readout import KP
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';INPUT=CYCLE/'highres_preflight_inputs_v1';OLD=CYCLE/'highres_preflight_pilot_v1';NEW=CYCLE/'highres_pre_relu_pilot_v1';OUT=CYCLE/'highres_pre_relu_review_v1';DEST=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005')/OUT.name
def read(p):return torch.load(p,weights_only=False,map_location='cpu')
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def main():
    OUT.mkdir(exist_ok=True);DEST.mkdir(exist_ok=True);ev={r['index']:r for r in read(INPUT/'evaluation_only.pt')};inf={r['index']:r for r in read(INPUT/'inference.pt')};manifest={r['index']:r for r in json.loads((INPUT/'inference_manifest.json').read_text())};old=read(OLD/'predictions.pt');new=read(NEW/'predictions.pt');pred=dict(baseline=old['baseline'],post_relu=old['appearance'],pre_relu=new['appearance']);a=json.loads((OLD/'results.json').read_text());b=json.loads((NEW/'results.json').read_text());summary=dict(baseline=a['summary']['baseline'],zero_appearance=a['summary']['zero_appearance'],post_relu=a['summary']['appearance'],pre_relu=b['summary']['appearance']);write(OUT/'results.json',dict(summary=summary,fit_gate=b['fit_gate'],pre_relu_performance=b['performance'],limits='Same42TRAIN/internalcal images, same300steps/init/schedule/readout/objective. Only frozen32channel feature tap changes; no validation tuning or defaults change.'))
    fig,axs=plt.subplots(2,3,figsize=(15,8));curves={'post_relu':json.loads((OLD/'curves.json').read_text()),'pre_relu':json.loads((NEW/'curves.json').read_text())}
    for row,stage in enumerate(['fit','calibration']):
        for col,kind in enumerate(['real','single','dual']):
            ax=axs[row,col]
            for name,cc in curves.items():
                cc=[z for z in cc if z['arm']=='appearance'];ax.plot([z['step'] for z in cc],[z['metrics'][stage][kind]['mean'] for z in cc],label=name,marker='o')
            ax.axhline(summary['baseline'][stage][kind]['mean'],color='gray',ls='--');ax.set_title(stage+' '+kind);ax.set_ylabel('input pixel error');ax.legend()
    fig.tight_layout();fig.savefig(OUT/'curves.png',dpi=140);plt.close(fig)
    selections=[];colors=['#ff7744','#00bbff','#ff77dd','#aaee44','#ffaa00','#7777ff']
    for kind in ['real','single','dual']:
        entries=[]
        for i,r in ev.items():
            if r['metadata']['stage']!='calibration' or r['metadata']['kind']!=kind:continue
            errors={name:float(((p[i][:,KP]-r['gt'])*r['pixels']).norm(dim=-1)[r['valid']].mean()) for name,p in pred.items()};entries.append(dict(index=i,kind=kind,metadata=r['metadata'],errors=errors,delta=errors['pre_relu']-errors['baseline']))
        z=max(entries,key=lambda z:z['delta']);i=z['index'];r=ev[i];im=Image.open(ROOT/manifest[i]['image']).convert('RGB');w,h=im.size;tile=Image.new('RGB',(2960,960),'white');d=ImageDraw.Draw(tile)
        for col,name in enumerate(['baseline','post_relu','pre_relu','GT_evaluation_only']):
            panel=im.copy();di=ImageDraw.Draw(panel)
            for q,box in enumerate(inf[i]['boxes']):
                cx,cy,bw,bh=(box*torch.tensor([w,h,w,h])).tolist();di.rectangle((cx-bw/2,cy-bh/2,cx+bw/2,cy+bh/2),outline=colors[q],width=1)
                for k,kp in enumerate(KP):
                    if name=='GT_evaluation_only':
                        if not r['valid'][q,k]:continue
                        x,y=(r['gt'][q,k]*r['pixels']).tolist();di.line((x-5,y,x+5,y),fill='#00ff66',width=3);di.line((x,y-5,x,y+5),fill='#00ff66',width=3)
                    else:
                        x,y=(pred[name][i][q,kp]*r['pixels']).tolist();di.ellipse((x-3,y-3,x+3,y+3),outline=colors[q],width=2)
                    if r['valid'][q,k]:di.text((x+5,y),str(kp+1),font=font(13),fill='#00ff66' if name=='GT_evaluation_only' else colors[q])
            panel.thumbnail((720,795));tile.paste(panel,(col*740+(740-panel.width)//2,100));d.text((col*740+10,10),name,font=font(24),fill='black');d.text((col*740+10,50),f'CAL {kind} worst change',font=font(20),fill='black')
        d.text((15,915),'All6 predicted queries; GT evaluation only. No development validation.',font=font(23),fill='black');z['file']=f'{kind}_worst_{i}.jpg';tile.save(OUT/z['file'],quality=95);selections.append(z)
    write(OUT/'visual_examples.json',selections)
    page=['<!doctype html><meta charset="utf-8"><title>ReLU前后特征对照</title><style>body{font:17px system-ui;margin:30px}img{max-width:100%}td,th{border:1px solid #aaa;padding:7px}table{border-collapse:collapse}</style><h1>ReLU前后32通道特征：同预算定位对照</h1><p>相同24张FIT/18张内部校准、同初始化/300步/样本顺序/目标/读出；仅把原末级特征换成最后一次ReLU前的32通道值。42图逐一验证ReLU(pre)与原末级特征完全一致，原模型参数和BN缓冲不变。9项读出单测仍适用。未读取开发验证、未改变标注或生产默认。</p><p>结果：真实组仍较好，合成单鼠拟合略改善但未减半；双鼠拟合未改善，两个合成组校准明显退步。三组拟合门槛仍仅真实组通过；不能把非零特征或绕过ReLU当作已解决域失配。下一步先设计/验证受控视觉适配，不扩大当前固定特征头的训练。</p><img src="curves.png"><table><tr><th>阶段</th><th>方法</th><th>组</th><th>mean↓</th></tr>']
    for stage in ['fit','calibration']:
        for name,s in summary.items():
            for kind in ['real','single','dual','real_bottom_feet']:page.append(f'<tr><td>{stage}</td><td>{name}</td><td>{kind}</td><td>{s[stage][kind]["mean"]:.3f}</td></tr>')
    page.append('</table><h2>每组校准变化最差的原图</h2><p>极值非发生率；最差也可能仍是改进。内部校准曾被底层冻结模型部分见过，不是独立外部测试。</p>')
    for z in selections:page.append(f'<h3>{z["kind"]}: {z["delta"]:+.2f}px</h3><img src="{z["file"]}">')
    page.append('<a href="results.json">完整结果</a>');(OUT/'index.html').write_text('\n'.join(page),encoding='utf-8')
    for p in OUT.iterdir():
        if p.is_file():shutil.copy2(p,DEST/p.name)
    print(json.dumps(selections,indent=2))
if __name__=='__main__':main()
