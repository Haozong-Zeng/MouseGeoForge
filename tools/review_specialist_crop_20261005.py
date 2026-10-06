"""Post-inference same-cohort crop evaluation and all-query visual review."""
import json,sys,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np,torch
from PIL import Image,ImageDraw
from train_pixel_adapter_matched_20261005 import prepare_rows,metrics,KP
from review_pixel_adapter_20261005 import font
from run_depth_pose_ablation import write_json
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';SRC=CYCLE/'specialist_crop_frozen_v1';OUT=CYCLE/'specialist_crop_review_v1';DEST=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005')/OUT.name

def main():
    torch.set_num_threads(2);OUT.mkdir(exist_ok=True);DEST.mkdir(exist_ok=True);rows=prepare_rows();raw=np.load(SRC/'predictions.npz');ids=raw['indices'].tolist();base=torch.stack([r['baseline']['xy'] for r in rows]);pred={'baseline':base}
    for name,folder in [('hard_roi','specialist_roi_frozen_v1'),('crop','specialist_crop_frozen_v1')]:
        data=np.load(CYCLE/folder/'predictions.npz');assert np.array_equal(data['indices'],raw['indices']);p=base.clone();p[ids]=torch.tensor(data['xy']);pred[name]=p
    summary={a:metrics(rows,p,ids) for a,p in pred.items()};cross={a:dict(known=0,closer_other_by5=0) for a in pred};per=[]
    for i in ids:
        r=rows[i];v=r['valid'];err={a:((p[i,:,KP]-r['gt'])*r['pixels']).norm(dim=-1) for a,p in pred.items()}
        z=dict(index=i,metadata=r['metadata'],known=int(v.sum()),errors={a:float(e[v].mean()) if v.any() else None for a,e in err.items()});z['delta']=z['errors']['crop']-z['errors']['baseline'] if v.any() else None;per.append(z)
        if r['metadata']['kind']=='dual':
            for q,k in v.nonzero().tolist():
                others=[o for o in range(6) if o!=q and v[o,k]]
                if k==0 or not others:continue
                for a,p in pred.items():
                    own=float(err[a][q,k]);other=min(float(((p[i,q,KP[k]]-r['gt'][o,k])*r['pixels']).norm()) for o in others);cross[a]['known']+=1;cross[a]['closer_other_by5']+=int(other+5<own)
    result=dict(summary=summary,dual_identity_diagnostic=cross,inference=json.loads((SRC/'results.json').read_text()),limits='Repeateddevelopment, sameoriginalbox-qualifiedknownpoints notAP; six extra HRNetforwards/image, unequalpretraining/training, no labels modified. Fixed1.25context combines scale/context changes, not isolated causal ablation.')
    write_json(OUT/'results.json',result);write_json(OUT/'per_image.json',per)
    manifest={r['index']:r for r in json.loads((CYCLE/'specialist_roi_inputs_v1/inference_manifest.json').read_text())};selections=[];colors=['#ff7744','#00bbff','#ff77dd','#aaee44','#ffaa00','#7777ff']
    for kind,fn in [('real_bottom',lambda m:m['kind']=='real' and m['physical_bottom']),('single',lambda m:m['kind']=='single'),('dual',lambda m:m['kind']=='dual')]:
        group=[z for z in per if z['known'] and fn(z['metadata'])]
        for mode,z in [('best_change',min(group,key=lambda z:z['delta'])),('worst_change',max(group,key=lambda z:z['delta']))]:
            z=dict(z,label=kind,mode=mode);i=z['index'];r=rows[i];im=Image.open(ROOT/manifest[i]['image']).convert('RGB');w,h=im.size;tile=Image.new('RGB',(2960,960),'white');draw=ImageDraw.Draw(tile)
            for col,name in enumerate(['baseline','hard_roi','crop','GT_evaluation_only']):
                panel=im.copy();di=ImageDraw.Draw(panel)
                for q,b in enumerate(r['baseline']['boxes']):
                    score=float(r['baseline']['objectness_logits'][q].sigmoid());cx,cy,bw,bh=(b*torch.tensor([w,h,w,h])).tolist();color=colors[q] if score>=.5 else '#888888';di.rectangle((cx-bw/2,cy-bh/2,cx+bw/2,cy+bh/2),outline=color,width=2 if score>=.5 else 1);di.text((cx-bw/2,cy-bh/2),f'q{q} {score:.2f}',font=font(12),fill=color)
                    for k,kp in enumerate(KP):
                        if name=='GT_evaluation_only':
                            if not r['valid'][q,k]:continue
                            x,y=(r['gt'][q,k]*r['pixels']).tolist();di.line((x-6,y,x+6,y),fill='#00ff66',width=3);di.line((x,y-6,x,y+6),fill='#00ff66',width=3)
                        else:
                            x,y=(pred[name][i,q,kp]*r['pixels']).tolist();di.ellipse((x-4,y-4,x+4,y+4),outline=color,width=2 if score>=.5 else 1)
                        if score>=.5 or name=='GT_evaluation_only':di.text((x+5,y+3),str(kp+1),font=font(13),fill='#00ff66' if name=='GT_evaluation_only' else color)
                panel.thumbnail((720,795));tile.paste(panel,(740*col+(740-panel.width)//2,100));draw.text((740*col+10,10),name,font=font(25),fill='black');draw.text((740*col+10,50),f'{kind} {mode} view{r["metadata"]["view"]}',font=font(18),fill='black')
            draw.text((15,915),'All6 queries shown, gray score<0.5; GT for evaluation only; no GT crops.',font=font(23),fill='black');z['file']=f'{kind}_{mode}_{i}.jpg';tile.save(OUT/z['file'],quality=95);selections.append(z)
    write_json(OUT/'visual_examples.json',selections)
    page=['<!doctype html><meta charset="utf-8"><title>固定预测框裁剪对照</title><style>body{font:17px system-ui;margin:30px}img{max-width:100%}td,th{border:1px solid #999;padding:8px}table{border-collapse:collapse}</style><h1>冻结四足定位模型：固定预测框裁剪</h1><p>同一144张开发输入、所有6个模型预测框，一律扩大1.25倍，Bicubic等比缩放并补黑至704×768。排除图像外和letterbox补边，坐标回投到原输入；不更改检测框/分数或其余18点。无GT框、GT数量、类别路由、训练或置信度筛选。8项单测通过。</p><p>比较的是上下文与尺度共同变化，不能解释成单一尺度因果实验。每图额外6次专门模型前向，不是公平计算预算比较。已反复使用的开发集，不是外部测试；指标只评价原框匹配的同一批已知点，不是AP。</p><table><tr><th>范围</th><th>方法</th><th>点数</th><th>mean↓</th><th>PCK32↑</th></tr>']
    for kind in ['all','real','single','dual','real_bottom_feet']:
        for a,s in summary.items():
            t=s[kind];page.append(f'<tr><td>{kind}</td><td>{a}</td><td>{t["known"]}</td><td>{t["mean"]:.3f}</td><td>{t["pck32"]:.3f}</td></tr>')
    page+=['</table><h2>结论：停止冻结裁剪组合</h2><p>真实底视仍较准，但合成单鼠和双鼠继续退步，跨鼠错误增加。固定扩大目标不能解决现有教师的合成域失配；不继续调整边距、不按类别选择输出、不晋升为默认模型。下一步只做共享高分辨率特征与实例条件化读出的训练可行性检查，不把教师坐标变成真值。</p><h2>跨鼠诊断</h2><pre>'+json.dumps(cross,indent=2)+'</pre><h2>固定规则结果极值</h2><p>变化最小不一定改善，正数仍为退步；极值图不表示发生频率。</p>']
    for z in selections:page.append(f'<h3>{z["label"]} {z["mode"]}: {z["delta"]:+.2f}px</h3><img src="{z["file"]}">')
    page.append('<p><a href="results.json">指标与计算量</a> | <a href="per_image.json">逐图指标</a></p>');(OUT/'index.html').write_text('\n'.join(page),encoding='utf-8')
    for p in OUT.iterdir():
        if p.is_file():shutil.copy2(p,DEST/p.name)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
