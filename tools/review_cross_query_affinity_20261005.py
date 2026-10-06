"""Affinity matrices and coverage-aware summary; no selected-model claim."""
import json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';OUT=CYCLE/'cross_query_affinity_diagnostic_v1'
DEST=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005')/OUT.name


def main():
    r=json.loads((OUT/'results.json').read_text());items=torch.load(OUT/'evidence.pt',map_location='cpu',weights_only=False);gallery=[]
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13)
    for role in ['train','validation']:
        g=next(x for x in items if x['role']==role and x['group']==0);n=len(g['nodes']);cell=48;pw=520;ph=500
        canvas=Image.new('RGB',(2*pw,2*ph),'white');draw=ImageDraw.Draw(canvas)
        for panel,(channel,m) in enumerate(g['evidence'].items()):
            ox=(panel%2)*pw;oy=(panel//2)*ph;draw.text((ox+8,oy+10),role+' scene0 | '+channel,fill='black',font=font)
            draw.text((ox+8,oy+32),'green edge = same posthoc identity; gray = unknown',fill='black',font=font)
            for a,node in enumerate(g['nodes']):
                label=f'v{node["view"]}q{node["query"]}'
                draw.text((ox+6,oy+89+a*cell),label,fill='black',font=font)
                draw.text((ox+65+a*cell,oy+65),label,fill='black',font=font)
                for b in range(n):
                    x=ox+60+b*cell;y=oy+85+a*cell;val=m[a,b]
                    if np.isfinite(val):
                        strength=np.clip(1-val/80,0,1) if channel.startswith('epipolar') else np.clip((val+.0),0,1)
                        color=(int(245-160*strength),int(245-80*strength),250)
                        draw.rectangle((x,y,x+cell-1,y+cell-1),fill=color)
                        ia,ib=g['identity_diagnostic'][a],g['identity_diagnostic'][b]
                        outline='#777777' if ia is None or ib is None else '#00aa55' if ia==ib else '#dd5555'
                        draw.rectangle((x+1,y+1,x+cell-2,y+cell-2),outline=outline,width=2)
                        draw.text((x+4,y+16),f'{val:.1f}' if channel.startswith('epipolar') else f'{val:.2f}',fill='black',font=font)
                    else:draw.rectangle((x,y,x+cell-1,y+cell-1),fill='#eeeeee')
        name=role+'_scene00_affinity.jpg';canvas.save(OUT/name,quality=95);gallery.append(name)
    page=['<!doctype html><meta charset="utf-8"><title>跨query关联证据</title><style>body{font:17px system-ui;margin:30px}img{max-width:100%}td,th{padding:8px;border:1px solid #aaa}</style><h1>跨视角身份证据：当前语义 token 更接近 query 编号</h1><p>21个训练场景84张图、16个开发场景64张图；所有图独立预测，无新训练。只使用预测点、预测框与特征计算证据，完成后才以IoU≥0.5的框关联诊断身份。未知身份检测保留并单列，不被偷偷排除出候选匹配。</p><table><tr><th>划分</th><th>证据</th><th>身份AUC</th><th>同query AUC</th><th>互选正确/已知</th><th>互选未知</th></tr>']
    for role,value in r['results'].items():
        for channel,s in value['channels'].items():
            m=s['mutual'];page.append(f'<tr><td>{role}</td><td>{channel}</td><td>{s["same_identity_auc"]:.3f}</td><td>{s["same_query_auc"]:.3f}</td><td>{m["correct"]}/{m["known"]}</td><td>{m["unknown"]}/{m["proposed"]}</td></tr>')
    page.append('</table><p>语义余弦互选在开发集185条边中全部选择相同query编号，但106条可判断身份的边仅43条属于同鼠；不能拿它直接当re-ID特征。关键点原坐标处的局部特征也未改善。几何均值互选有93/98条已知边正确，仍有64条未知、5条已知错误，且互选无法拒绝只有一个错误候选的情形，尚不能安全融合。</p><p>AUC仅在身份可判断的边上评估；分帧划分不等于独立视频，黑/棕毛色数据不代表同色真实鼠。几何数值越低越接近，余弦越高越接近。下图绿色边框=事后同鼠，红色=异鼠，灰色=未知；颜色仅用于报告，未进入特征计算。</p><p><a href="results.json">完整结果及来源哈希</a></p>')
    for name in gallery:page.append(f'<h2>{name}</h2><img src="{name}">')
    (OUT/'index.html').write_text('\n'.join(page),encoding='utf-8');DEST.mkdir(exist_ok=True)
    for name in gallery+['index.html','results.json']:shutil.copy2(OUT/name,DEST/name)
    print(DEST/'index.html')


if __name__=='__main__':main()
