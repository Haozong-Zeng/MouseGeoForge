"""Read-only prediction gallery and handoff for independent unary diagnostic."""
import json
import shutil
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from PIL import Image,ImageDraw
from training.finetune_blendedmvs import load_scene
from run_depth_pose_ablation import write_json
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005'
OUT=CYCLE/'independent_unary_eligibility_audit_v1'
DEST=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005')
KP=[0,14,15,16,17]


def main():
    data=torch.load(CYCLE/'local_candidate_selector_pilot_v1/validation_inputs.pt',map_location='cpu',weights_only=False)
    results=json.loads((OUT/'results.json').read_text())
    points=json.loads((OUT/'validation_independent_unique_points.json').read_text())
    syn=[p for p in points if data['metadata'][p['row']]['kind']=='synthetic']
    best=sorted(syn,key=lambda p:p['error']-p['baseline_error'])
    worst=list(reversed(best))
    ids=list(dict.fromkeys([next(i for i,m in enumerate(data['metadata']) if m['kind']=='synthetic'),best[0]['row'],worst[0]['row']]))
    pred=np.load(OUT/'predictions.npz');gallery=[]
    for i in ids:
        m=data['metadata'][i];f=load_scene(ROOT/m['scene']).frames[m['view']]
        w,h=map(int,data['pixels'][i]);rgb=Image.open(f.image_path).convert('RGB').resize((w,h))
        canvas=Image.new('RGB',(2*w,h+44),'white');draw=ImageDraw.Draw(canvas)
        for col,arm in enumerate(['baseline','independent_unique']):
            canvas.paste(rgb,(col*w,44));draw.text((col*w+10,10),arm,fill='black')
            for q in np.flatnonzero(data['logits'][i].sigmoid().numpy()>=.5):
                color=['#ff5555','#00ccff','#eecc00','#ee00cc','#00ee99','#ff9900'][q]
                for kp in KP:
                    x,y=pred['validation_'+arm][i,q,kp]*[w,h]+[col*w,44]
                    draw.ellipse((x-5,y-5,x+5,y+5),outline=color,width=3);draw.text((x+5,y),str(kp+1),fill=color)
            for t in range(2):
                for kp in KP:
                    if data['point_valid'][i,t,kp]:
                        x,y=data['gt_xy'][i,t,kp].numpy()*[w,h]+[col*w,44]
                        draw.line((x-5,y-5,x+5,y+5),fill='white',width=2);draw.line((x-5,y+5,x+5,y-5),fill='white',width=2)
        name=f'row{i:03d}_view{m["view"]}.jpg';canvas.save(OUT/name,quality=92);gallery.append(dict(row=i,path=name,**m))
    write_json(OUT/'gallery.json',dict(images=gallery,selection='First synthetic validation image and greatest individual improvement/harm; diagnostic, not representative random examples.',best_points=best[:5],worst_points=worst[:5]))
    page=['<!doctype html><meta charset="utf-8"><title>真正单视角候选选择检查</title><style>body{font:17px system-ui;margin:30px}img{max-width:100%}td,th{padding:8px;border:1px solid #aaa}</style><h1>去除跨视角启用门槛：冻结模型，不新增训练</h1><p>独立启用条件仅为本图恰好一个预测框置信度≥0.5，不读取GT数量、不需要其他相机。仅更新鼻点与四足；其余点与检测器不变。真实数据验证结果保持，合成开发验证开始真正启用修正。</p><table><tr><th>数据/方案</th><th>真实可靠点</th><th>底视足点</th><th>合成可靠点</th></tr>']
    for r in results['results']:
        m=r['metrics'];page.append(f'<tr><td>{r["role"]}/{r["arm"]}</td><td>{m["real/reliable"]["mean_error"]:.2f}</td><td>{m["real/view0/feet"]["mean_error"]:.2f}</td><td>{m["synthetic/reliable"]["mean_error"]:.2f}</td></tr>')
    page.append('</table><p>误差为各自输入图片像素。GT框匹配仅在预测后用于评价；不是端到端AP。这仍是单鼠队列，尚不能证明双鼠或独立视频泛化。FR2/AZR2L2重复使用的开发验证集，不是外部测试。单图、打乱顺序、重复图片、无效几何输入均通过一致性检查。生产默认不变。</p><h2>完整指标与一致性检查</h2><p><a href="results.json">results.json</a></p><pre>'+json.dumps(results['tests'],indent=2)+'</pre><p>下面为合成验证首图和最大改善/最大新增误差诊断图。白叉为标注；彩色圆点为全部置信度≥0.5的预测，不以GT数量筛选。</p>')
    for g in gallery:page.append(f'<h2>{g["path"]}</h2><p>{g["scene"]}</p><img src="{g["path"]}">')
    (OUT/'index.html').write_text('\n'.join(page),encoding='utf-8')
    delivery=DEST/OUT.name;delivery.mkdir(exist_ok=True)
    for name in ['index.html','results.json','gallery.json']+[g['path'] for g in gallery]:shutil.copy2(OUT/name,delivery/name)
    print(json.dumps(dict(report=str(delivery/'index.html'),gallery=gallery,worst=worst[:3]),ensure_ascii=False))


if __name__=='__main__':main()
