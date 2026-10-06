"""Matched retention report using current saved coordinates and original synthetic GT."""
import json,sys,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import torch,numpy as np
from PIL import Image,ImageDraw
from review_pose_recipe_20261005 import full_targets
import review_pixel_adapter_four_arm_20261005 as gallery
from train_pixel_adapter_matched_20261005 import prepare_rows,KP
from run_surface_curve_pilot_20261003 import load_item
from run_depth_pose_ablation import write_json
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';SRC=CYCLE/'synthetic_supervision_retention_v1';REF=CYCLE/'pose_recipe_factorial_v1';OUT=CYCLE/'synthetic_retention_review_v1'
DEST=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005')/OUT.name
def main():
    OUT.mkdir(exist_ok=True);DEST.mkdir(exist_ok=True);torch.set_num_threads(2);rows=prepare_rows();gt,valid=full_targets(rows);pixels=torch.stack([r['pixels'] for r in rows]);raw=json.loads((SRC/'results.json').read_text());pred=torch.load(SRC/'predictions.pt',map_location='cpu',weights_only=False)['predictions'];arms=list(pred);summary=raw['summary']
    gallery.OUT=OUT;gallery.ARMS=arms;gallery.COLORS=dict(zip(arms,['#222222','#266abf','#bb337e']));curves=[r for r in json.loads((REF/'curves.json').read_text()) if r['arm']=='huber_low']+json.loads((SRC/'curves.json').read_text());gallery.chart(curves,summary)
    auxiliary={}
    for a,p in pred.items():
        e=((p-gt)*pixels[:,None,None]).norm(dim=-1);auxiliary[a]={}
        for phase in ['calibration','validation']:
            for kind in ['single','dual']:
                ids=[i for i,r in enumerate(rows) if r['metadata']['stage']==phase and r['metadata']['kind']==kind];v=valid[ids].clone();v[:,:,KP]=False;z=e[ids][v];auxiliary[a][phase+'/'+kind]=dict(n=z.numel(),mean=float(z.mean()),p95=float(z.quantile(.95)))
    write_json(OUT/'auxiliary_accuracy.json',auxiliary);write_json(OUT/'results.json',raw)
    chosen=json.loads((CYCLE/'pose_recipe_review_v1/visual_examples.json').read_text());chosen=[e for e in chosen if e['category'] in ['real_bottom','real_side']]
    for e in chosen:
        i=e['index'];q=e['query'];kp=e['point']-1;r=rows[i];item=load_item(r['record']);vi=r['record']['views'].index(r['metadata']['view']);rgb=item['batch']['images'][0,vi];im=Image.fromarray((rgb.permute(1,2,0).numpy()*255).clip(0,255).round().astype(np.uint8));tile=Image.new('RGB',(2160,875),'white');d=ImageDraw.Draw(tile);e['errors']={}
        for col,(a,p) in enumerate(pred.items()):
            panel=im.copy();di=ImageDraw.Draw(panel);x,y=(p[i,q,kp]*pixels[i]).tolist();gx,gy=(gt[i,q,kp]*pixels[i]).tolist();err=float(((p[i,q,kp]-gt[i,q,kp])*pixels[i]).norm());e['errors'][a]=err
            di.line((x,y,gx,gy),fill='yellow',width=2);di.ellipse((x-8,y-8,x+8,y+8),outline='red',width=4);di.line((gx-10,gy,gx+10,gy),fill='#00ff66',width=4);di.line((gx,gy-10,gx,gy+10),fill='#00ff66',width=4);tile.paste(panel,(col*720,100));d.text((col*720+10,10),a+f' {err:.2f}px',font=gallery.font(22),fill='black');d.text((col*720+10,50),f'{e["category"]} view{r["metadata"]["view"]} point{kp+1}',font=gallery.font(21),fill='black')
        tile.save(OUT/e['file'],quality=95)
    write_json(OUT/'visual_examples.json',chosen)
    page=['<!doctype html><meta charset="utf-8"><title>合成非目标点监督保留对照</title><style>body{font:17px system-ui;max-width:2000px;margin:30px auto}img{max-width:100%}td,th{border:1px solid #aaa;padding:8px}table{border-collapse:collapse}</style><h1>辅助合成监督：主要足点结果基本不变</h1><p>同400步、初始化、原图采样、低学习率Huber配方。唯一新增项为0.25×既有合成非目标点损失，不稀释鼻尖/四脚主损失；真实辅助损失为零，双鼠尾巴禁用。4项保护单测通过，冻结权重哈希不变，训练120.01秒、峰值4.051GB。</p><p>开发主指标控制/辅助：真实31.789/31.834，单鼠18.414/18.448，双鼠19.430/19.455，底脚43.465/43.448px；底脚PCK32均0.458。内部底脚32.803/32.911，也未优于原31.130。没有解决底视，不晋升、不加步数；停止相近头部微调，下一步检查已有专门定位教师。</p><h2>非目标点的合成真值误差</h2><p>不是坐标漂移。缺失点不评价，双鼠尾巴不评价；真实背部无伪GT。开发集非外部测试，单种子条件定位指标。</p><table><tr><th>范围</th><th>方法</th><th>n</th><th>mean↓</th><th>P95↓</th></tr>']
    for group in auxiliary['baseline']:
        for a in arms:
            z=auxiliary[a][group];page.append(f'<tr><td>{group}</td><td>{a}</td><td>{z["n"]}</td><td>{z["mean"]:.3f}</td><td>{z["p95"]:.3f}</td></tr>')
    page+=['</table><h2>主指标校验曲线</h2><img src="calibration_curves.png"><h2>沿用上一报告固定图例</h2><p>不是根据新结果重选。红圈预测、绿十字当前GT。</p>']
    for e in chosen:page.append(f'<h3>{e["file"]}</h3><img src="{e["file"]}">')
    page.append('<p><a href="results.json">主指标</a> | <a href="auxiliary_accuracy.json">非目标点精度</a> | <a href="visual_examples.json">图例编号</a></p>');(OUT/'index.html').write_text('\n'.join(page),encoding='utf-8')
    for p in OUT.iterdir():
        if p.is_file():shutil.copy2(p,DEST/p.name)
    print(json.dumps(auxiliary,indent=2))
if __name__=='__main__':main()
