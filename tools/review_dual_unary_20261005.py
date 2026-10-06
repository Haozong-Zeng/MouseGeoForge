"""Raw all-query dual diagnostic gallery; no GT-filtered predictions."""
import json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from PIL import Image,ImageDraw
from training.finetune_blendedmvs import load_scene
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';OUT=CYCLE/'dual_unary_diagnostic_v1';DEST=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005')/OUT.name
KP=[0,14,15,16,17]

def main():
    d=torch.load(OUT/'inputs.pt',map_location='cpu',weights_only=False);r=json.loads((OUT/'results.json').read_text());points=json.loads((OUT/'per_query_points.json').read_text());a=np.load(OUT/'predictions.npz')
    worst=sorted(points,key=lambda p:p['error']-p['baseline_error'],reverse=True);cross=[p for p in worst if p.get('new_other_closer')]
    ids=list(dict.fromkeys([0,worst[0]['row'],cross[0]['row']]));gallery=[]
    for i in ids:
        m=d['metadata'][i];w,h=map(int,d['pixels'][i]);f=load_scene(ROOT/m['scene']).frames[m['view']];rgb=Image.open(f.image_path).convert('RGB').resize((w,h));canvas=Image.new('RGB',(2*w,h+44),'white');draw=ImageDraw.Draw(canvas)
        for col,arm in enumerate(['baseline','per_query']):
            canvas.paste(rgb,(col*w,44));draw.text((col*w+10,12),arm,fill='black')
            for q in np.flatnonzero(d['logits'][i].sigmoid().numpy()>=.5):
                color=['#ff5555','#00ccff','#eecc00','#ee00cc','#00ee99','#ff9900'][q];cx,cy,bw,bh=d['boxes'][i,q].numpy()*[w,h,w,h]
                draw.rectangle((col*w+cx-bw/2,44+cy-bh/2,col*w+cx+bw/2,44+cy+bh/2),outline=color,width=2);draw.text((col*w+cx-bw/2,44+cy-bh/2),f'q{q}',fill=color)
                for k in KP:
                    x,y=a[arm][i,q,k]*[w,h]+[col*w,44];draw.ellipse((x-5,y-5,x+5,y+5),outline=color,width=3);draw.text((x+5,y),str(k+1),fill=color)
            for t in range(2):
                for k in KP:
                    if d['point_valid'][i,t,k]:
                        x,y=d['gt_xy'][i,t,k].numpy()*[w,h]+[col*w,44];draw.line((x-4,y-4,x+4,y+4),fill='white',width=2);draw.line((x-4,y+4,x+4,y-4),fill='white',width=2)
        name=f'row{i:03d}_view{m["view"]}.jpg';canvas.save(OUT/name,quality=92);gallery.append(dict(row=i,path=name,**m))
    summary=dict(gallery=gallery,worst_points=worst[:8],new_other_nearer_points=cross,selection='First image, max additional error, max additional error among new-other-nearer points; diagnostic selection, not random.')
    (OUT/'gallery.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    page=['<!doctype html><meta charset="utf-8"><title>双鼠冻结检查</title><style>body{font:17px system-ui;margin:30px}img{max-width:100%}td,th{border:1px solid #aaa;padding:8px}</style><h1>单鼠训练的选择头不能直接推广到双鼠</h1><p>冻结权重、32张合成双鼠开发验证图片，64只GT、64个预测框、45个IoU≥0.5正确匹配。全部图片恰有两个可信预测框，因此unique-query门控全部回退；per-query独立修正是真正的分布外诊断，未再训练。</p><table><tr><th>方案</th><th>320个可靠点平均误差</th><th>P95</th><th>检测门控PCK32</th><th>增加误差&gt;16px</th></tr>']
    for row in r['results']:
        v=row['metrics']['synthetic/reliable'];page.append(f'<tr><td>{row["arm"]}</td><td>{v["mean_error"]:.2f}</td><td>{v["p95_error"]:.2f}</td><td>{v["gated_pck32"]:.3f}</td><td>{v["harmed_over16"]}</td></tr>')
    page.append('</table><p>条件误差不是端到端AP。比目标实例更接近另一只鼠同名点：24→28，新增7点；该接近度仅用于诊断，不等同准确的ID交换标签。拒绝直接对双鼠启用当前单鼠选择头，生产默认不变。下一步先约束候选选择对原概率的偏离，再根据证据决定是否需要多实例TRAIN监督。颜色是本图query编号，不能当作跨相机实例ID；白叉为已有GT。显示所有可信预测，无GT数量筛选。</p><p><a href="results.json">完整结果</a> | <a href="gallery.json">诊断样本与误差</a></p>')
    for g in gallery:page.append(f'<h2>{g["path"]}</h2><p>{g["scene"]}</p><img src="{g["path"]}">')
    (OUT/'index.html').write_text('\n'.join(page),encoding='utf-8');DEST.mkdir(exist_ok=True)
    for name in ['index.html','results.json','gallery.json']+[g['path'] for g in gallery]:shutil.copy2(OUT/name,DEST/name)
    print(json.dumps(summary))

if __name__=='__main__':main()
