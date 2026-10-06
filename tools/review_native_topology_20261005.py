"""Native decoder-topology result and frozen-path visual review."""
import json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from PIL import Image,ImageDraw
from training.finetune_blendedmvs import load_scene
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';OUT=CYCLE/'native_topology_control_v2';DEST=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005')/OUT.name;KP=[0,14,15,16,17]


def main():
    r=json.loads((OUT/'results.json').read_text());d=torch.load(OUT/'validation_eval_inputs.pt',map_location='cpu',weights_only=False);pred=np.load(OUT/'predictions.npz');ps=json.loads((OUT/'structured_union_validation_points.json').read_text())
    bottom=[p for p in ps if d['metadata'][p['row']]['kind']=='real' and d['metadata'][p['row']]['view']==0 and p['point']!=1];worst=sorted(bottom,key=lambda p:p['error']-p['baseline_error'],reverse=True)
    dual=next(i for i,m in enumerate(d['metadata']) if m['scene'].endswith('frame_000287') and m['view']==0);ids=list(dict.fromkeys([0,worst[0]['row'],dual]));gallery=[]
    for i in ids:
        m=d['metadata'][i];w,h=map(int,d['pixels'][i]);f=load_scene(ROOT/m['scene']).frames[m['view']];rgb=Image.open(f.image_path).convert('RGB').resize((w,h));canvas=Image.new('RGB',(3*w,h+44),'white');draw=ImageDraw.Draw(canvas)
        for col,arm in enumerate(['baseline','full','structured_union']):
            canvas.paste(rgb,(col*w,44));draw.text((col*w+10,12),arm,fill='black')
            for q in np.flatnonzero(d['logits'][i].sigmoid().numpy()>=.5):
                color=['#ff5555','#00ccff','#eecc00','#ee00cc','#00ee99','#ff9900'][q]
                for k in KP:
                    x,y=pred['validation_'+arm][i,q,k]*[w,h]+[col*w,44];draw.ellipse((x-5,y-5,x+5,y+5),outline=color,width=3);draw.text((x+5,y),str(k+1),fill=color)
            for t in range(2):
                for k in KP:
                    if d['point_valid'][i,t,k]:
                        x,y=d['gt_xy'][i,t,k].numpy()*[w,h]+[col*w,44];draw.line((x-4,y-4,x+4,y+4),fill='white',width=2);draw.line((x-4,y+4,x+4,y-4),fill='white',width=2)
        name=f'row{i:03d}_view{m["view"]}.jpg';canvas.save(OUT/name,quality=92);gallery.append(dict(row=i,path=name,**m))
    curves={}
    for arm in ['full','structured_union']:
        values=[p['loss'] for p in json.loads((OUT/f'{arm}_history.json').read_text())];curves[arm]=dict(first100=float(np.mean(values[:100])),previous100=float(np.mean(values[-200:-100])),last100=float(np.mean(values[-100:])))
    review=dict(gallery=gallery,curves=curves,worst_bottom=worst[:6],decision='No clear structured-union advantage in this bounded trial; no promotion or extension of this variant. Do not infer all grouped attention fails. Preserve negative controls. Next prioritize actual-RGB integration audit of already-promising bounded candidate fusion, not another similar head/loss sweep.',prior_negative_heatmap='OLD/review_reliable_heatmap_v1/decision.json rejects matched reliable Gaussian-KL in3seeds; no repeat sigma/weight search justified here.')
    (OUT/'review.json').write_text(json.dumps(review,indent=2),encoding='utf-8')
    page=['<!doctype html><meta charset="utf-8"><title>原生单视角注意力拓扑对照</title><style>body{font:17px system-ui;margin:30px}img{max-width:100%}td,th{padding:8px;border:1px solid #aaa}</style><h1>结构限制没有显示明确优势</h1><p>冻结原始模型，1119744个新分支参数；两组同初始化、同800步/训练图像/坐标损失，仅自注意力连接方式不同。受限组允许同实例或同名关键点直接交互。不是GroupPose完整复现。仅鼻点和四足可改变，其余点不变。</p><table><tr><th>方案/数据</th><th>真实可靠点</th><th>底视足点</th><th>合成单鼠</th><th>合成双鼠</th></tr>']
    for row in r['results']:
        m=row['metrics'];vals=[f'{m[k]["mean_error"]:.2f}' for k in ['real/reliable','real/view0/feet','synthetic_single/reliable','synthetic_dual/reliable']];page.append(f'<tr><td>{row["arm"]}/{row["role"]}</td>'+''.join(f'<td>{x}</td>' for x in vals)+'</tr>')
    page.append('</table><p>真实开发验证约33.3px，双鼠约22.4px；结构两组接近，不能宣称分组拓扑优于全连接。合成训练池为153张单鼠与42张双鼠，不同于先前候选头461张单鼠训练池；不得跨实验当作严格同预算对照。</p><h2>安全与计算量</h2><pre>'+json.dumps(dict(gates=r['gates'],compute=r['compute'],parameters=r['parameters_trainable'],curves=curves),indent=2)+'</pre><p>耗时为GPU缓存上的新增分支训练，不含原1B编码器提取成本。v1在训练前因21张curve类型合成图分组校验停止，修正为非真实单鼠分组后v2完成；失败保留。curve图只使用鼻足点，不使用曲线标签。</p><p>复核历史发现真实足鼻点高斯热图KL已有三种子失败对照，暂不重复搜索sigma/权重。先完成已取得部分收益的有界候选融合在真实RGB中的集成与多视角核验，保留本次负结果。所有正式模型、数据标签、锁定划分均未改变。</p><p><a href="results.json">完整结果</a> | <a href="review.json">曲线与诊断样本</a></p>')
    for g in gallery:page.append(f'<h2>{g["path"]}</h2><img src="{g["path"]}">')
    (OUT/'index.html').write_text('\n'.join(page),encoding='utf-8');DEST.mkdir(exist_ok=True)
    for name in ['index.html','results.json','review.json','protocol.json']+[g['path'] for g in gallery]:shutil.copy2(OUT/name,DEST/name)
    print(json.dumps(review))


if __name__=='__main__':main()
