"""Posthoc qualified-query grouping for the frozen TRAIN layer diagnostic."""
import sys,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from PIL import Image
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';INPUT=CYCLE/'highres_preflight_inputs_v1';SRC=CYCLE/'highres_layer_audit_v2';OUT=CYCLE/'highres_layer_review_v2';DEST=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005')/OUT.name
def read(p):return torch.load(p,weights_only=False,map_location='cpu')
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')

def main():
    OUT.mkdir(exist_ok=True);DEST.mkdir(exist_ok=True);ev={r['index']:r for r in read(INPUT/'evaluation_only.pt')};inf={r['index']:r for r in read(INPUT/'inference.pt')};manifest={r['index']:r for r in json.loads((INPUT/'inference_manifest.json').read_text())};stats=json.loads((SRC/'statistics.json').read_text());layers=['act2','layer1','stage2','stage3']+sorted({s['layer'] for s in stats if s['layer'].startswith('stage4_unit')})+['stage4_pre_relu','stage4'];summary=[]
    for kind in ['real','single','dual']:
        for variant in ['rgb','gray']:
            for layer in layers:
                ss=[s for s in stats if ev[s['index']]['metadata']['kind']==kind and s['variant']==variant and s['layer']==layer];entries=[]
                for s in ss:
                    r=ev[s['index']];assert r['metadata']['stage']=='fit';qq=[b for b in s['boxes'] if r['valid'][b['query']].any() and b['cells']>0];n=sum(b['cells'] for b in qq)
                    entries.append(dict(index=s['index'],zero=all(b['allchannels_zero_cells']==b['cells'] for b in qq),nonpositive=all(b['allchannels_nonpositive_cells']==b['cells'] for b in qq),weighted_zero=sum(b['allchannels_zero_cells'] for b in qq)/n,rms=sum(b['rms']*b['cells'] for b in qq)/n,spatialstd=sum(b['spatialstd']*b['cells'] for b in qq)/n))
                summary.append(dict(kind=kind,variant=variant,layer=layer,images=len(entries),entire_qualified_boxes_zero=sum(z['zero'] for z in entries),entire_qualified_boxes_nonpositive=sum(z['nonpositive'] for z in entries),median_zero_fraction=float(np.median([z['weighted_zero'] for z in entries])),median_rms=float(np.median([z['rms'] for z in entries])),median_spatialstd=float(np.median([z['spatialstd'] for z in entries]))))
    result=dict(inference=json.loads((SRC/'results.json').read_text()),summary=summary,scope='24FIT only; qualifiedquery grouping evaluation-only. Predboxes may overlap; weighted averages count overlap more than once. These are diagnostic counts, not population rates.');write(OUT/'results.json',result)
    first={k:next(i for i,r in ev.items() if r['metadata']['kind']==k and r['metadata']['stage']=='fit') for k in ['real','single','dual']}
    for mode,columns in [('layers',[('rgb','RGB'),('rgb','act2'),('rgb','layer1'),('rgb','stage2'),('rgb','stage3'),('rgb','stage4')]),('gray_control',[('rgb','RGB'),('rgb','stage3'),('rgb','stage4'),('gray','stage3'),('gray','stage4')])]:
        fig,axs=plt.subplots(3,len(columns),figsize=(3.8*len(columns),12))
        for row,(kind,i) in enumerate(first.items()):
            im=np.asarray(Image.open(ROOT/manifest[i]['image']).convert('RGB'));h,w=im.shape[:2];data={v:np.load(SRC/'energy'/f'{i:04d}_{v}.npz') for v in ['rgb','gray']}
            for col,(variant,layer) in enumerate(columns):
                ax=axs[row,col]
                if layer=='RGB':ax.imshow(im)
                else:
                    z=data[variant][layer];ax.imshow(np.maximum(z,1e-6),norm=LogNorm(vmin=1e-6,vmax=100),cmap='magma',extent=(0,z.shape[1]*4,z.shape[0]*4,0))
                ax.set_xlim(0,w);ax.set_ylim(h,0);ax.set_title(f'{kind} {i}\n{variant} {layer}');ax.set_xticks([]);ax.set_yticks([])
                for q in ev[i]['valid'].any(-1).nonzero().flatten():
                    cx,cy,bw,bh=(inf[i]['boxes'][q]*torch.tensor([w,h,w,h])).tolist();ax.add_patch(Rectangle((cx-bw/2,cy-bh/2),bw,bh,fill=False,color='lime',lw=1))
        fig.tight_layout();fig.savefig(OUT/f'{mode}.png',dpi=130);plt.close(fig)
    page=['<!doctype html><meta charset="utf-8"><title>HRNet逐层特征诊断</title><style>body{font:17px system-ui;margin:30px}img{max-width:100%}td,th{border:1px solid #aaa;padding:7px}table{border-collapse:collapse}</style><h1>冻结HRNet：特征在哪里消失？</h1><p>只用24张既定FIT图，真实/合成单鼠/双鼠各8；所有权重和BN统计保持eval冻结。每张分别原RGB与固定PIL灰度复制三通道，不逐图选择输入。推理不读GT；推理后按原匹配合格实例汇总全部预测框统计。末级原RGB特征与前轮缓存精确一致。下图每组第一个FIT样本，并非按结果挑选；各层统一绝对响应对数色阶。</p><img src="layers.png"><h2>灰度对照</h2><img src="gray_control.png"><h2>层级统计</h2><p>全零图数指图中所有合格实例框内每个像素的全部通道精确为零。pre_relu使用非正图数判断ReLU截断；RMS为框内原激活，不做特征归一化。</p><table><tr><th>组</th><th>输入</th><th>层</th><th>全零图/8</th><th>全非正图/8</th><th>零比例中位数</th><th>RMS中位数</th></tr>']
    for s in summary:page.append(f'<tr><td>{s["kind"]}</td><td>{s["variant"]}</td><td>{s["layer"]}</td><td>{s["entire_qualified_boxes_zero"]}</td><td>{s["entire_qualified_boxes_nonpositive"]}</td><td>{s["median_zero_fraction"]:.4f}</td><td>{s["median_rms"]:.4f}</td></tr>')
    page.append('</table><p>本诊断不证明BN是原因，也不证明非零特征足够准确定位。未改变checkpoint、标注或默认模型。</p><a href="results.json">完整统计与校验</a>');(OUT/'index.html').write_text('\n'.join(page),encoding='utf-8')
    for p in OUT.iterdir():
        if p.is_file():shutil.copy2(p,DEST/p.name)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
