"""Feature provenance comparison on existing42 small TRAIN/calibration RGBs."""
import json,sys,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np
from PIL import Image
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';INPUT=CYCLE/'highres_preflight_inputs_v1';SOURCE=CYCLE/'official_quadruped_features_v2';OUT=CYCLE/'official_quadruped_feature_review_v2';DEST=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005')/OUT.name
def read(p):return torch.load(p,weights_only=False,map_location='cpu')
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def main():
    OUT.mkdir(exist_ok=True);DEST.mkdir(exist_ok=True);torch.set_num_threads(2);ev={r['index']:r for r in read(INPUT/'evaluation_only.pt')};inf={r['index']:r for r in read(INPUT/'inference.pt')};manifest={r['index']:r for r in json.loads((INPUT/'inference_manifest.json').read_text())};stats=[]
    for i,r in ev.items():
        for name,folder in [('finetuned','highres_preflight_features_v1'),('official','official_quadruped_features_v2')]:
            cache=read(CYCLE/folder/f'{i:04d}.pt');f=cache['features'];h,w=cache['image_hw'];fh,fw=f.shape[-2:];yy,xx=torch.meshgrid((torch.arange(fh)+.5)*4,(torch.arange(fw)+.5)*4,indexing='ij');grid=torch.stack([xx/w,yy/h],-1);valid=(xx<w)&(yy<h);mask=torch.zeros_like(valid)
            for q in r['valid'].any(-1).nonzero().flatten():
                b=inf[i]['boxes'][q];mask|=((grid>=b[:2]-b[2:]/2)&(grid<=b[:2]+b[2:]/2)).all(-1)&valid
            vals=f[:,mask];n=vals.shape[-1];zeros=int((vals==0).all(0).sum());stats.append(dict(index=i,model=name,metadata=r['metadata'],cells=n,zero_cells=zeros,all_zero=zeros==n,zero_fraction=zeros/n,rms=float(vals.square().mean().sqrt()),spatialstd=float(vals.std(-1,unbiased=False).mean())))
    grouped=[]
    for stage in ['fit','calibration']:
        for kind in ['real','single','dual']:
            for name in ['finetuned','official']:
                s=[z for z in stats if z['metadata']['stage']==stage and z['metadata']['kind']==kind and z['model']==name];grouped.append(dict(stage=stage,kind=kind,model=name,images=len(s),entire_box_zero=sum(z['all_zero'] for z in s),median_zero_fraction=float(np.median([z['zero_fraction'] for z in s])),median_rms=float(np.median([z['rms'] for z in s])),median_spatialstd=float(np.median([z['spatialstd'] for z in s]))))
    write(OUT/'statistics.json',stats);write(OUT/'results.json',dict(inference=json.loads((SOURCE/'results.json').read_text()),summary=grouped,limits='Feature response not poseaccuracy; same architecture different pretrained/finetuned weights andBNbuffers. This alone cannot isolate whichtrainingcomponent changed.'))
    fig,axs=plt.subplots(3,3,figsize=(14,13))
    for row,kind in enumerate(['real','single','dual']):
        i=next(i for i,r in ev.items() if r['metadata']['kind']==kind and r['metadata']['stage']=='fit');im=np.asarray(Image.open(ROOT/manifest[i]['image']).convert('RGB'));h,w=im.shape[:2]
        for col,name in enumerate(['RGB','finetuned','official']):
            ax=axs[row,col]
            if name=='RGB':ax.imshow(im)
            else:
                folder='highres_preflight_features_v1' if name=='finetuned' else 'official_quadruped_features_v2';f=read(CYCLE/folder/f'{i:04d}.pt')['features'];energy=f.abs().amax(0).numpy();ax.imshow(np.maximum(energy,1e-6),norm=LogNorm(vmin=1e-6,vmax=100),cmap='magma',extent=(0,energy.shape[1]*4,energy.shape[0]*4,0))
            ax.set_xlim(0,w);ax.set_ylim(h,0);ax.set_title(f'{kind} FIT {i}\n{name}');ax.set_xticks([]);ax.set_yticks([])
            for q in ev[i]['valid'].any(-1).nonzero().flatten():
                cx,cy,bw,bh=(inf[i]['boxes'][q]*torch.tensor([w,h,w,h])).tolist();ax.add_patch(Rectangle((cx-bw/2,cy-bh/2),bw,bh,fill=False,color='lime',lw=1))
    fig.tight_layout();fig.savefig(OUT/'feature_comparison.png',dpi=140);plt.close(fig)
    page=['<!doctype html><meta charset="utf-8"><title>通用预训练与单域微调特征</title><style>body{font:17px system-ui;margin:30px}img{max-width:100%}td,th{border:1px solid #aaa;padding:8px}table{border-collapse:collapse}</style><h1>原始SuperAnimal四足预训练 vs 单鼠微调后的骨干</h1><p>现有本地官方权重，1830个骨干张量严格加载核验，不下载、不使用其关键点头/语义映射。相同42张既定TRAIN/内部校准RGB、相同归一化/补边/stride4，全部冻结。GT只在推理完成后的框统计中用于原有实例资格；未读取开发验证或锁定数据。</p><p>下图每组第一个FIT样本，固定绝对对数色阶。非零特征不代表定位准确；骨干权重和BN缓冲都可能在微调中变化，仅此对照不能区分具体训练因素。</p><img src="feature_comparison.png"><table><tr><th>阶段</th><th>组</th><th>模型</th><th>全零图/总图</th><th>零比例中位数</th><th>RMS中位数</th><th>空间标准差</th></tr>']
    for s in grouped:page.append(f'<tr><td>{s["stage"]}</td><td>{s["kind"]}</td><td>{s["model"]}</td><td>{s["entire_box_zero"]}/{s["images"]}</td><td>{s["median_zero_fraction"]:.4f}</td><td>{s["median_rms"]:.4f}</td><td>{s["median_spatialstd"]:.4f}</td></tr>')
    page.append('</table><a href="results.json">完整结果和来源哈希</a>');(OUT/'index.html').write_text('\n'.join(page),encoding='utf-8')
    for p in OUT.iterdir():
        if p.is_file():shutil.copy2(p,DEST/p.name)
    print(json.dumps(grouped,indent=2))
if __name__=='__main__':main()
