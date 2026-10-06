"""Small preflight metrics, raw RGB/feature collapse and fixed-extreme galleries."""
import sys,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import numpy as np,torch
from PIL import Image,ImageDraw
from review_pixel_adapter_20261005 import font
from mousegeoforge.models.highres_instance_readout import KP
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';INPUT=CYCLE/'highres_preflight_inputs_v1';SRC=CYCLE/'highres_preflight_pilot_v1';OUT=CYCLE/'highres_preflight_review_v1';DEST=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005')/OUT.name
def read(p):return torch.load(p,weights_only=False,map_location='cpu')
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')

def main():
    torch.set_num_threads(2);OUT.mkdir(exist_ok=True);DEST.mkdir(exist_ok=True);result=json.loads((SRC/'results.json').read_text());pred=read(SRC/'predictions.pt');ev={r['index']:r for r in read(INPUT/'evaluation_only.pt')};inf={r['index']:r for r in read(INPUT/'inference.pt')};manifest={r['index']:r for r in json.loads((INPUT/'inference_manifest.json').read_text())};per=[];stats=[]
    for i,r in ev.items():
        errors={a:((p[i][:,KP]-r['gt'])*r['pixels']).norm(dim=-1) for a,p in pred.items()};z=dict(index=i,metadata=r['metadata'],errors={a:float(e[r['valid']].mean()) for a,e in errors.items()});z['delta']=z['errors']['appearance']-z['errors']['baseline'];per.append(z)
        f=read(CYCLE/'highres_preflight_features_v1'/f'{i:04d}.pt')['features'];h,w=manifest[i]['image_hw'];fh,fw=f.shape[-2:];yy,xx=torch.meshgrid((torch.arange(fh)+.5)*4,(torch.arange(fw)+.5)*4,indexing='ij');grid=torch.stack([xx/w,yy/h],-1);valid=(xx<w)&(yy<h);boxes=inf[i]['boxes'];roi=torch.zeros_like(valid)
        for q in r['valid'].any(-1).nonzero().flatten():
            b=boxes[q];roi|=((grid>=b[:2]-b[2:]/2)&(grid<=b[:2]+b[2:]/2)).all(-1)&valid
        energy=f.abs().amax(0);stats.append(dict(index=i,metadata=r['metadata'],max_abs=float(f.abs().max()),all_feature_zero_fraction=float((f==0).float().mean()),qualified_box_spatial_cells=int(roi.sum()),roi_allchannels_zero_fraction=float((energy[roi]==0).float().mean()),roi_energy_mean=float(energy[roi].mean()),roi_energy_max=float(energy[roi].max()),roi_channel_spatial_std_mean=float(f[:,roi].std(-1).mean())))
    write(OUT/'feature_statistics.json',stats);write(OUT/'per_image.json',per)
    grouped={}
    for kind in ['real','single','dual']:
        ss=[s for s in stats if s['metadata']['kind']==kind];grouped[kind]=dict(images=len(ss),median_box_zero_fraction=float(np.median([s['roi_allchannels_zero_fraction'] for s in ss])),median_box_energy_mean=float(np.median([s['roi_energy_mean'] for s in ss])),median_box_spatialstd=float(np.median([s['roi_channel_spatial_std_mean'] for s in ss])))
    result['feature_statistics']=grouped;write(OUT/'results.json',result)
    selected=[]
    for label,fn in [('real_bottom',lambda m:m['kind']=='real' and m['physical_bottom']),('single',lambda m:m['kind']=='single'),('dual',lambda m:m['kind']=='dual')]:
        group=[z for z in per if z['metadata']['stage']=='calibration' and fn(z['metadata'])]
        for mode,z in [('best_change',min(group,key=lambda z:z['delta'])),('worst_change',max(group,key=lambda z:z['delta']))]:
            z=dict(z,label=label,mode=mode);i=z['index'];r=ev[i];im=Image.open(ROOT/manifest[i]['image']).convert('RGB');w,h=im.size;tile=Image.new('RGB',(2960,960),'white');d=ImageDraw.Draw(tile);colors=['#ff7744','#00bbff','#ff77dd','#aaee44','#ffaa00','#7777ff']
            for col,name in enumerate(['baseline','zero_appearance','appearance','GT_evaluation_only']):
                panel=im.copy();di=ImageDraw.Draw(panel)
                for q,b in enumerate(inf[i]['boxes']):
                    cx,cy,bw,bh=(b*torch.tensor([w,h,w,h])).tolist();di.rectangle((cx-bw/2,cy-bh/2,cx+bw/2,cy+bh/2),outline=colors[q],width=1)
                    for k,kp in enumerate(KP):
                        if name=='GT_evaluation_only':
                            if not r['valid'][q,k]:continue
                            x,y=(r['gt'][q,k]*r['pixels']).tolist();di.line((x-5,y,x+5,y),fill='#00ff66',width=3);di.line((x,y-5,x,y+5),fill='#00ff66',width=3)
                        else:
                            x,y=(pred[name][i][q,kp]*r['pixels']).tolist();di.ellipse((x-3,y-3,x+3,y+3),outline=colors[q],width=2)
                        if r['valid'][q,k]:di.text((x+5,y),f'{kp+1}',font=font(13),fill='#00ff66' if name=='GT_evaluation_only' else colors[q])
                panel.thumbnail((720,795));tile.paste(panel,(col*740+(740-panel.width)//2,100));d.text((col*740+10,10),name,font=font(24),fill='black');d.text((col*740+10,50),f'CAL {label} {mode}',font=font(18),fill='black')
            d.text((15,915),'All6 queries; GT only evaluation. TRAIN/calibration only, NO developmentVAL.',font=font(23),fill='black');z['file']=f'{label}_{mode}_{i}.jpg';tile.save(OUT/z['file'],quality=95);selected.append(z)
    write(OUT/'visual_examples.json',selected)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Rectangle
    fig,axs=plt.subplots(3,2,figsize=(12,13))
    for row,kind in enumerate(['real','single','dual']):
        i=next(i for i,r in ev.items() if r['metadata']['kind']==kind and r['metadata']['stage']=='fit');r=ev[i];im=np.asarray(Image.open(ROOT/manifest[i]['image']).convert('RGB'));h,w=im.shape[:2];cache=read(CYCLE/'highres_preflight_features_v1'/f'{i:04d}.pt');energy=cache['features'].abs().amax(0).numpy();ph,pw=cache['pad_hw']
        axs[row,0].imshow(im);axs[row,1].imshow(np.maximum(energy,1e-6),norm=LogNorm(vmin=1e-6,vmax=10),cmap='magma',extent=(0,w+pw,h+ph,0))
        for col in [0,1]:
            ax=axs[row,col];ax.set_xlim(0,w);ax.set_ylim(h,0);ax.set_title(f'{kind} FIT index{i}: '+('RGB' if col==0 else 'max abs feature (fixed log scale)'))
            for q in r['valid'].any(-1).nonzero().flatten():
                cx,cy,bw,bh=(inf[i]['boxes'][q]*torch.tensor([w,h,w,h])).tolist();ax.add_patch(Rectangle((cx-bw/2,cy-bh/2),bw,bh,fill=False,color='lime',lw=1))
    fig.tight_layout();fig.savefig(OUT/'feature_energy.png',dpi=150);plt.close(fig)
    curves=json.loads((SRC/'curves.json').read_text());fig,axs=plt.subplots(2,3,figsize=(15,8))
    for row,stage in enumerate(['fit','calibration']):
        for col,kind in enumerate(['real','single','dual']):
            ax=axs[row,col]
            for arm in ['appearance','zero_appearance']:
                cc=[c for c in curves if c['arm']==arm];ax.plot([c['step'] for c in cc],[c['metrics'][stage][kind]['mean'] for c in cc],label=arm,marker='o')
            ax.axhline(result['summary']['baseline'][stage][kind]['mean'],color='gray',ls='--');ax.set_title(stage+' '+kind);ax.set_ylabel('input pixel error');ax.legend()
    fig.tight_layout();fig.savefig(OUT/'curves.png',dpi=150);plt.close(fig)
    page=['<!doctype html><meta charset="utf-8"><title>高分辨率实例分支预检</title><style>body{font:17px system-ui;margin:30px}img{max-width:100%}td,th{border:1px solid #aaa;padding:8px}table{border-collapse:collapse}</style><h1>高分辨率实例条件化分支：小样本可行性预检</h1><p>24张TRAIN拟合、18张内部校准，未读取开发验证图。共享stride4、32通道HRNet特征，原实例语义token/预测框/粗点条件化，64px高斯软先验的零初始化残差；不是硬框裁剪。与图像特征置零、保留坐标及token的相同名义结构对照。两组固定300步、同样本顺序、同初始化。9项单测通过；默认关闭/零初始化精确保持原结果，其他18点不变。教师及原模型见过部分底层TRAIN数据，内部校准不是独立泛化测试。</p><p>预设拟合门槛为三个数据组均≤12px且较原模型至少减半；只有真实组通过，合成单/双鼠未通过，不扩大缓存或自动追加训练。检查特征是否存在域失配/空间退化，再决定架构，而非继续微调同类头。</p><img src="curves.png"><table><tr><th>阶段</th><th>方法</th><th>范围</th><th>known</th><th>mean↓</th></tr>']
    for stage in ['fit','calibration']:
        for arm,s in result['summary'].items():
            for kind in ['real','single','dual','real_bottom_feet']:
                x=s[stage][kind];page.append(f'<tr><td>{stage}</td><td>{arm}</td><td>{kind}</td><td>{x["known"]}</td><td>{x["mean"]:.3f}</td></tr>')
    page+=['</table><h2>共享特征统计（原预测框内，仅评价）</h2><p>14张单鼠中8张、14张双鼠中13张，其合格实例预测框内全部空间位置的32通道均精确为零；14张真实图没有整框全零。单鼠8张全零均属于拟合子集。不能把这是何种归一化/训练导致的原因直接认定为已证实，下一步需逐层定位。以下按每组第一个FIT输入展示，未按误差挑选。</p><pre>'+json.dumps(grouped,indent=2)+'</pre><img src="feature_energy.png"><h2>校准样本极值</h2>']
    for z in selected:page.append(f'<h3>{z["label"]} {z["mode"]} {z["delta"]:+.2f}px</h3><img src="{z["file"]}">')
    page.append('<p><a href="results.json">完整结果</a> | <a href="feature_statistics.json">逐图特征统计</a></p>');(OUT/'index.html').write_text('\n'.join(page),encoding='utf-8')
    for p in OUT.iterdir():
        if p.is_file():shutil.copy2(p,DEST/p.name)
    print(json.dumps(dict(feature_statistics=grouped,fit_gate=result['fit_gate'],examples=selected),indent=2))
if __name__=='__main__':main()
