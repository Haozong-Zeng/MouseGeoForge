"""Summarize controlled prior experiment and raw bottom-view failure checks."""
import json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from PIL import Image,ImageDraw
from training.finetune_blendedmvs import load_scene
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';OUT=CYCLE/'bounded_prior_candidate_pilot_v1';DEST=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005')/OUT.name
KP=[0,14,15,16,17]

def main():
    r=json.loads((OUT/'results.json').read_text());d=torch.load(CYCLE/'local_candidate_selector_pilot_v1/validation_inputs.pt',map_location='cpu',weights_only=False)
    points=json.loads((OUT/'geometry_validation_matched_points.json').read_text())
    bottom=[p for p in points if d['metadata'][p['row']]['kind']=='real' and d['metadata'][p['row']]['view']==0 and p['point']!=1]
    worst=sorted(bottom,key=lambda p:p['error']-p['baseline_error'],reverse=True)
    ids=list(dict.fromkeys([0,worst[0]['row'],5]));pred=np.load(OUT/'predictions.npz');gallery=[]
    for i in ids:
        m=d['metadata'][i];w,h=map(int,d['pixels'][i]);f=load_scene(ROOT/m['scene']).frames[m['view']];rgb=Image.open(f.image_path).convert('RGB').resize((w,h));canvas=Image.new('RGB',(3*w,h+44),'white');draw=ImageDraw.Draw(canvas)
        for col,arm in enumerate(['baseline','unary','geometry']):
            xy=d['base_xy'][i].numpy() if arm=='baseline' else pred[arm+'_validation_matched'][i]
            canvas.paste(rgb,(col*w,44));draw.text((col*w+10,12),arm,fill='black')
            for q in np.flatnonzero(d['logits'][i].sigmoid().numpy()>=.5):
                color=['#ff5555','#00ccff','#eecc00','#ee00cc','#00ee99','#ff9900'][q]
                for k in KP:
                    x,y=xy[q,k]*[w,h]+[col*w,44];draw.ellipse((x-5,y-5,x+5,y+5),outline=color,width=3);draw.text((x+5,y),str(k+1),fill=color)
            for t in range(2):
                for k in KP:
                    if d['point_valid'][i,t,k]:
                        x,y=d['gt_xy'][i,t,k].numpy()*[w,h]+[col*w,44];draw.line((x-4,y-4,x+4,y+4),fill='white',width=2);draw.line((x-4,y+4,x+4,y-4),fill='white',width=2)
        name=f'row{i:03d}_view{m["view"]}.jpg';canvas.save(OUT/name,quality=92);gallery.append(dict(row=i,path=name,**m))
    curves={}
    for arm in ['unary','geometry']:
        a=[v['loss'] for v in json.loads((OUT/f'{arm}_history.json').read_text())];curves[arm]=dict(first100=float(np.mean(a[:100])),previous100=float(np.mean(a[-200:-100])),last100=float(np.mean(a[-100:])))
    summary=dict(curves=curves,gallery=gallery,worst_bottom_points=worst[:8],decision='Probability prior mitigates catastrophic geometry misses but does not solve dual out-of-distribution association. Do not promote; verify multi-view subset/duplicate/ordering and training-seed repeats before larger budgets.')
    (OUT/'review.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    page=['<!doctype html><meta charset="utf-8"><title>有界概率先验对照</title><style>body{font:17px system-ui;margin:30px}img{max-width:100%}td,th{padding:8px;border:1px solid #aaa}</style><h1>有界概率先验：缓解极端错误，但并非全部解决</h1><p>相同冻结基线、同初始化、同800步样本顺序和监督，仅改变候选评分：原概率对数先验 + 预先固定±2的学习修正。没有根据验证集调阈值。置信相对质量低于exp(-2)的候选不能击败原始点；这不能保证身份或解剖正确。</p><table><tr><th>对照</th><th>真实可靠点误差</th><th>真实底视足点</th><th>合成可靠点</th></tr>']
    for row in r['results']:
        m=row['metrics'];vals=[f'{m[k]["mean_error"]:.2f}' if k in m else '-' for k in ['real/reliable','real/view0/feet','synthetic/reliable']];page.append(f'<tr><td>{row["tag"]}</td>'+''.join(f'<td>{v}</td>' for v in vals)+'</tr>')
    page.append('</table><p>参考：原始基线真实验证33.25/底视48.46；无先验unary29.51/36.12；无先验geometry29.59/61.16。有先验geometry为26.32/39.70：比同先验unary30.44/41.20改善，但底视仍不如旧unary。不得按验证视角挑选不同方案。</p><p>双鼠原始23.30、无先验unary26.06、有先验unary25.66；仍退步，不直接启用。均为匹配后的条件误差，不是AP。复用开发验证集，未证明独立视频泛化。单/双鼠差异和几何融合仍需重复种子与视角压力测试。生产默认不变。</p><h2>损失</h2><pre>'+json.dumps(curves,indent=2)+'</pre><p><a href="results.json">完整结果</a> | <a href="review.json">底视新增错误</a></p><p>以下包括原严重失败样本及本轮最大新增误差底视样本。白叉为已有人工标注，彩点为全部可信预测；不是代表性随机图集。</p>')
    for g in gallery:page.append(f'<h2>{g["path"]}</h2><p>{g["scene"]}</p><img src="{g["path"]}">')
    (OUT/'index.html').write_text('\n'.join(page),encoding='utf-8');DEST.mkdir(exist_ok=True)
    for name in ['index.html','results.json','review.json','protocol.json']+[g['path'] for g in gallery]:shutil.copy2(OUT/name,DEST/name)
    print(json.dumps(summary))

if __name__=='__main__':main()
