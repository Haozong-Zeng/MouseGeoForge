"""Show actual preprocessed RGB, original queries and composite readout."""
import json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from PIL import Image,ImageDraw
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';OUT=CYCLE/'composite_candidate_rgb_audit_v1';DEST=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005')/OUT.name;KP=[0,14,15,16,17]


def main():
    r=json.loads((OUT/'results.json').read_text());cache=json.loads((CYCLE/'composite_candidate_cache_audit_v1/results.json').read_text());groups=torch.load(OUT/'fresh_groups.pt',map_location='cpu',weights_only=False);d=torch.load(CYCLE/'singleview_fusion_inputs_seed17_v1/validation.pt',map_location='cpu',weights_only=False);gallery=[]
    selected=[(0,0),(0,len(groups[0]['views'])-1),(1,0),(2,0),(3,1)]
    for gi,vi in selected:
        g=groups[gi];row=g['rows'][vi];rgb=Image.open(OUT/g['images'][vi]).convert('RGB');w,h=rgb.size;canvas=Image.new('RGB',(3*w,h+44),'white');draw=ImageDraw.Draw(canvas)
        for col,arm in enumerate(['baseline','local','composite']):
            canvas.paste(rgb,(col*w,44));draw.text((col*w+10,12),arm,fill='black');p=g['predictions']
            for q in np.flatnonzero(p['objectness_logits'][vi].sigmoid().numpy()>=.5):
                color=['#ff5555','#00ccff','#eecc00','#ee00cc','#00ee99','#ff9900'][q];cx,cy,bw,bh=p['boxes'][vi,q].numpy()*[w,h,w,h];draw.rectangle((col*w+cx-bw/2,44+cy-bh/2,col*w+cx+bw/2,44+cy+bh/2),outline=color,width=2)
                for k in KP:
                    x,y=g['outputs'][arm]['xy'][vi,q,k].numpy()*[w,h]+[col*w,44];draw.ellipse((x-5,y-5,x+5,y+5),outline=color,width=3);draw.text((x+5,y),str(k+1),fill=color)
            for t in range(2):
                for k in KP:
                    if d['point_valid'][row,t,k]:
                        x,y=d['gt_xy'][row,t,k].numpy()*[w,h]+[col*w,44];draw.line((x-4,y-4,x+4,y+4),fill='white',width=2);draw.line((x-4,y+4,x+4,y-4),fill='white',width=2)
        name=f'group{gi:02d}_view{g["views"][vi]}_comparison.jpg';canvas.save(OUT/name,quality=92);gallery.append(dict(group=gi,view=g['views'][vi],path=name,scene=g['scene'],row=row))
    summary=dict(gallery=gallery,max_rgb_pose_difference=max(x['delta']['xy'] for x in r['records']),max_rgb_logit_difference=max(x['delta']['logits'] for x in r['records']),max_composite_cache_difference=max(g['tests'].get('cached_composite_xy',0) for g in r['groups']),unique_RGB=r['unique_images'],encoder_calls=r['encoder_calls'],sum_forward_seconds=sum(x['seconds'] for x in r['records']),wall_seconds=r['elapsed_seconds'],cache_metrics=cache['results'],limits='Engineering replay only.14uniqueimages and15independentencoderpasses; subset/order/duplicate fusion tests reuse these freshly encoded outputs, not separately encoded joint-view contexts. Dualtestpair only2views, bothpredict2 andabstain, notgeneral identityguarantee. Physicalframe GTannotations displayed onlyafterforward; allconfidence>=.5queries shown.')
    (OUT/'review.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    page=['<!doctype html><meta charset="utf-8"><title>组合候选真实RGB复现</title><style>body{font:17px system-ui;margin:30px}img{max-width:100%}td,th{padding:8px;border:1px solid #aaa}</style><h1>真实RGB路径与候选缓存一致</h1><p>14张唯一图片（两组真实五视角、合成单鼠两视角、双鼠两视角），15次独立单视角encoder前向含一次重复检查。不是一次联合多视角encoder的算力公平对照。下图为真正送入模型的预处理后RGB。</p><p>局部选择在本图只有一个可信query时启用；几何融合需要另外两个不同物理相机的唯一可信query。支持不足时保留局部结果。默认全部关闭。几何选择原始候选0时，会有意覆盖局部修正返回原坐标，而非忽略该选择。</p><table><tr><th>开发缓存方案</th><th>真实可靠点</th><th>底视足点</th><th>合成单鼠</th></tr>']
    for row in cache['results']:
        m=row['metrics'];page.append(f'<tr><td>{row["mode"]}</td>'+''.join(f'<td>{m[k]["mean_error"]:.2f}</td>' for k in ['real/reliable','real/view0/feet','synthetic/reliable'])+'</tr>')
    page.append('</table><p>上述准确率表来自原80张开发缓存，不是14张RGB重采样的全新泛化成绩。RGB仅核对工程一致性。双鼠两视角均预测两个query，组合完全回退；这不能证明多鼠身份匹配正确，若某视角漏掉一只鼠，单query假设仍可能失效。不得据此部署多鼠融合。</p><h2>数值与成本</h2><pre>'+json.dumps({k:v for k,v in summary.items() if k not in ['gallery','cache_metrics']},indent=2)+'</pre><p>重新排列、重复视角、单/双视角局部回退、全无效视角均通过；这些操作是在新鲜独立V1特征上重算融合，未宣称每种组合都重新编码。标注、预测框和实例数未进入推理；框和全部非鼻足点不变。无训练、无checkpoint晋升。</p><p><a href="results.json">逐图来源/误差/耗时</a> | <a href="review.json">汇总</a></p>')
    for g in gallery:page.append(f'<h2>{g["path"]}</h2><p>{g["scene"]}</p><img src="{g["path"]}">')
    (OUT/'index.html').write_text('\n'.join(page),encoding='utf-8');DEST.mkdir(exist_ok=True)
    for name in ['index.html','results.json','review.json','protocol.json']+[g['path'] for g in gallery]:shutil.copy2(OUT/name,DEST/name)
    print(json.dumps({k:v for k,v in summary.items() if k!='cache_metrics'}))


if __name__=='__main__':main()
