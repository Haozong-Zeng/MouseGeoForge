"""Visual diagnostic; GT is display-only, all confident queries retained."""
import json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from PIL import Image,ImageDraw
from training.finetune_blendedmvs import load_scene
from extract_singleview_probability_diagnostic_20261005 import nms_peaks
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';BASE=CYCLE/'singleview_probability_diagnostic_v1';SOURCE=CYCLE/'singleview_fusion_inputs_seed17_v1'
OUT=CYCLE/'review_singleview_probabilities_v1';DEST=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005')
KP=[0,14,15,16,17]


def main():
    assert not OUT.exists();OUT.mkdir();pages=[]
    for role in ['train','validation']:
        data=torch.load(SOURCE/f'{role}.pt',map_location='cpu',weights_only=False)
        with np.load(BASE/f'{role}_probabilities.npz') as a:prob=a['probability'];ids=a['row_indices'];local={int(i):j for j,i in enumerate(ids)}
        scene=next(data['metadata'][i]['record']['scene'] for i in ids if data['metadata'][i]['record']['kind']=='real')
        selected=[int(i) for i in ids if data['metadata'][i]['record']['scene']==scene and data['metadata'][i]['view'] in [0,3,5]]
        for i in selected:
            m=data['metadata'][i];physical=int(m['view']);camera='cam'+chr(65+physical);frame=load_scene(ROOT/scene).frames[physical]
            w,h=map(int,data['pixels'][i]);rgb=Image.open(frame.image_path).convert('RGB').resize((w,h));score=data['objectness_logits'][i].sigmoid().numpy()
            for q in np.flatnonzero(score>=.5):
                canvas=Image.new('RGB',(3*w,2*(h+48)),(240,240,240));draw=ImageDraw.Draw(canvas)
                for ki,kp in enumerate(KP):
                    p=prob[local[i],q,ki];heat=np.array(Image.fromarray((p/max(float(p.max()),1e-12)).astype(np.float32)).resize((w,h),Image.Resampling.BILINEAR))
                    # Fixed linear intensity normalized per map; neither mask
                    # nor GT is used to crop or rescale the spatial support.
                    alpha=(.6*heat)[...,None];array=np.asarray(rgb).astype(float);overlay=np.uint8(array*(1-alpha)+np.array([255,35,0])[None,None]*alpha)
                    x0=(ki%3)*w;y0=(ki//3)*(h+48);canvas.paste(Image.fromarray(overlay),(x0,y0+48));draw.text((x0+10,y0+10),f'{role} {camera} q{q} ID{kp+1} score{score[q]:.2f}',fill='black')
                    mean=data['xy'][i,q,kp].numpy()*[w,h];mx,my=mean+[x0,y0+48]
                    draw.ellipse((mx-7,my-7,mx+7,my+7),outline='#00ffff',width=3)
                    peaks,values=nms_peaks(p)
                    for j,(peak,val) in enumerate(zip(peaks,values)):
                        x,y=peak*[w,h]+[x0,y0+48];draw.rectangle((x-4,y-4,x+4,y+4),outline='#ff00ff',width=2);draw.text((x+6,y),f'{j+1}:{val:.2f}',fill='#ff00ff')
                    for t in range(2):
                        if data['point_valid'][i,t,kp]:
                            x,y=data['gt_xy'][i,t,kp].numpy()*[w,h]+[x0,y0+48]
                            draw.line((x-8,y-8,x+8,y+8),fill='white',width=3);draw.line((x-8,y+8,x+8,y-8),fill='white',width=3)
                x0=2*w;y0=h+48;draw.text((x0+20,y0+50),'Cyan circle: predicted expectation\nMagenta squares: top 5 NMS peaks\nWhite cross: annotation (display only)\nRed: probability / own maximum\n\nFixed first scene per split; no GT crop\nIDs: nose 1, paws 15-18\nQuery colors/indices are NOT identity',fill='black')
                name=f'{role}_{camera}_q{q}.jpg';canvas.save(OUT/name,quality=92);pages.append(dict(image=name,role=role,scene=scene,view=physical,query=int(q)))
    result=json.loads((BASE/'results.json').read_text());html=['<!doctype html><meta charset="utf-8"><title>空间概率图诊断</title><style>body{font:17px system-ui;margin:30px}img{max-width:100%}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #aaa}</style><h1>均值、峰值与局部证据</h1><p>青色圆圈是原预测期望位置，紫方框是只由预测图选择的前五个候选，白叉是事后显示的标注。热度按每张图自身最大值显示。候选里存在正确位置不等于模型能正确选择它；oracle不是可部署成绩。</p><table><tr><th>数据/点集</th><th>数目</th><th>均值预测误差</th><th>最大峰误差</th><th>前5候选oracle误差</th></tr>']
    for r in result['summaries']:
        if r['kind']=='real':html.append(f'<tr><td>{r["role"]}/{r["part"]}</td><td>{r["points"]}</td><td>{r["mean_error"]:.2f}</td><td>{r["peak_error"]:.2f}</td><td>{r["oracle_top5_error"]:.2f} (不可当预测)</td></tr>')
    html.append('</table><p>单位为模型输入像素，GT框在前向后辅助关联；不是端到端AP，验证集为反复使用的开发集。足点误差大于32px定义困难子集，仅用于事后分析。</p>')
    for p in pages:html.append(f'<h2>{p["image"]}</h2><a href="{p["image"]}"><img src="{p["image"]}"></a>')
    (OUT/'index.html').write_text('\n'.join(html),encoding='utf-8');(OUT/'manifest.json').write_text(json.dumps(pages,indent=2),encoding='utf-8');shutil.copytree(OUT,DEST/OUT.name);print(json.dumps(pages))


if __name__=='__main__':main()
