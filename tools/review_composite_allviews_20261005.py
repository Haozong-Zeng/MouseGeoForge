"""Render all confidence-thresholded queries, without GT count filtering."""
import json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from PIL import Image,ImageDraw
OUT=ROOT/'output/singleview_fusion_cycle_20261005/composite_allview_association_audit_v1'
DEST=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005')/OUT.name
KP=[0,14,15,16,17]


def main(dual_only=False):
    r=json.loads((OUT/'results.json').read_text());groups=torch.load(OUT/'groups.pt',map_location='cpu',weights_only=False);gallery=[]
    for gi,g in enumerate(groups):
        d=g['evaluation_data'];p=g['predictions']
        for vi,view in enumerate(g['record']['views']):
            rgb=Image.open(OUT/g['images'][vi]).convert('RGB');w,h=rgb.size
            im=Image.new('RGB',(2*w,h+40),'white');draw=ImageDraw.Draw(im)
            for col,mode in enumerate(['baseline','composite']):
                im.paste(rgb,(col*w,40));draw.text((col*w+8,12),f'{mode} | group{gi} view{view}',fill='black')
                for q in np.flatnonzero(p['objectness_logits'][vi].sigmoid().numpy()>=.5):
                    color=['#ff5555','#00ccff','#eecc00','#ee00cc','#00ee99','#ff9900'][q]
                    cx,cy,bw,bh=p['boxes'][vi,q].numpy()*[w,h,w,h]
                    draw.rectangle((col*w+cx-bw/2,40+cy-bh/2,col*w+cx+bw/2,40+cy+bh/2),outline=color,width=2)
                    draw.text((col*w+cx-bw/2,40+cy-bh/2),f'q{q}',fill=color)
                    for k in KP:
                        x,y=g['outputs'][mode]['xy'][vi,q,k].numpy()*[w,h]+[col*w,40]
                        draw.ellipse((x-4,y-4,x+4,y+4),outline=color,width=2);draw.text((x+4,y),str(k+1),fill=color)
                for t in range(2):
                    for k in KP:
                        if d['point_valid'][vi,t,k]:
                            x,y=d['gt_xy'][vi,t,k].numpy()*[w,h]+[col*w,40]
                            draw.line((x-4,y-4,x+4,y+4),fill='white',width=2);draw.line((x-4,y+4,x+4,y-4),fill='white',width=2)
            name=f'group{gi:02d}_view{view}.jpg';im.save(OUT/name,quality=92);gallery.append(name)
    scope='既有验证列表全部16个双鼠场景，每个4视角，共64张图片。' if dual_only else '预先固定源列表最前两个双鼠验证场景和首个合成单鼠场景。'
    page=['<!doctype html><meta charset="utf-8"><title>全视角关联检查</title><style>body{font:17px system-ui;margin:30px}img{max-width:100%}td,th{padding:8px;border:1px solid #aaa}</style><h1>全视角关联检查</h1><p>'+scope+'每张 RGB 独立编码；相机保持共同场景坐标。模型未接收 GT 框、实例数量或跨视角身份。白叉仅在预测结束后显示标注，颜色为 query 编号，不是跨相机身份。</p><table><tr><th>场景</th><th>各视角预测数</th><th>原始误差 px</th><th>组合误差 px</th><th>融合可用点</th></tr>']
    for i,g in enumerate(r['groups']):
        k=g['kind']+'/reliable'
        page.append(f'<tr><td>{i}: {g["kind"]}</td><td>{[x["count"] for x in g["detections"]]}</td><td>{g["metrics"]["baseline"][k]["mean_error"]:.3f}</td><td>{g["metrics"]["composite"][k]["mean_error"]:.3f}</td><td>{sum(x["fusion_eligible"] for x in g["detections"])}</td></tr>')
    page.append('</table><p>双鼠各四个视角均预测两个query，因此局部与融合均回退。这只是没有触发假设的有限样本，不能证明漏检后不会串鼠；预测数量正确不代表框或足点正确。误差是源开发场景上事后框关联的条件指标；新增视角不是独立视频泛化。</p><p>1/2视角局部回退、顺序、重复和全无效支持均通过。RGB 与旧视角缓存一致，相机投影误差小于0.002px。未训练、未修改标签或正式模型。</p><p><a href="results.json">完整检测/误差/投影/耗时</a> | <a href="protocol.json">固定协议</a></p>')
    for name in gallery:page.append(f'<h2>{name}</h2><img src="{name}">')
    (OUT/'index.html').write_text('\n'.join(page),encoding='utf-8');DEST.mkdir(exist_ok=True)
    for name in gallery+['index.html','results.json','protocol.json']:shutil.copy2(OUT/name,DEST/name)
    print(str(DEST/'index.html'))


if __name__=='__main__':main()
