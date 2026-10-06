"""Evaluate frozen specialist ROI readout only after all-query inference."""
import json,sys,shutil,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np,torch
from PIL import Image,ImageDraw
from train_pixel_adapter_matched_20261005 import prepare_rows,metrics,KP
from review_pixel_adapter_20261005 import font
from run_depth_pose_ablation import write_json
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';SRC=CYCLE/'specialist_roi_frozen_v1';INPUT=CYCLE/'specialist_roi_inputs_v1';OUT=CYCLE/'specialist_roi_review_v1';DEST=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005')/OUT.name
def main():
    OUT.mkdir(exist_ok=True);DEST.mkdir(exist_ok=True);torch.set_num_threads(2);rows=prepare_rows();raw=np.load(SRC/'predictions.npz');ids=raw['indices'].tolist();assert len(ids)==144
    baseline=torch.stack([r['baseline']['xy'] for r in rows]);updated=baseline.clone();updated[ids]=torch.tensor(raw['xy']);pred={'baseline':baseline,'specialist_roi':updated};summary={a:metrics(rows,p,ids) for a,p in pred.items()};inference=json.loads((SRC/'results.json').read_text());manifest={r['index']:r for r in json.loads((INPUT/'inference_manifest.json').read_text())}
    changed=[];pairs=[];cross={a:dict(known_points=0,closer_other_by5=0) for a in pred};per_image=[];coverage=[]
    for j,i in enumerate(ids):
        r=rows[i];v=r['valid'];e={a:((p[i,:,KP]-r['gt'])*r['pixels']).norm(dim=-1) for a,p in pred.items()};m=r['metadata'];conf=r['baseline']['objectness_logits'].sigmoid()>=.5
        per_image.append(dict(index=i,metadata=m,known=int(v.sum()),errors={a:float(z[v].mean()) if v.any() else None for a,z in e.items()},delta=float((e['specialist_roi'][v]-e['baseline'][v]).mean()) if v.any() else None))
        for q,k in v.nonzero().tolist():
            box=r['baseline']['boxes'][q];lo=box[:2]-box[2:]/2;hi=box[:2]+box[2:]/2
            target=r['gt'][q,k];global_xy=torch.tensor(raw['global_valid_xy'][j,k]);roi_xy=updated[i,q,KP[k]]
            coverage.append(dict(index=i,kind=m['kind'],query=q,point=KP[k]+1,
                gt_inside_box=bool(((target>=lo)&(target<=hi)).all()),
                global_peak_inside_box=bool(((global_xy>=lo)&(global_xy<=hi)).all()),
                distance_to_box_edge_px=float(torch.minimum((roi_xy-lo)*r['pixels'],(hi-roi_xy)*r['pixels']).min()),
                raw_heatmap_score=float(raw['raw_scores'][j,q,k]),roi_error=float(e['specialist_roi'][q,k])))
        if m['kind']=='real':
            global_px=torch.tensor(raw['global_valid_xy'][j])*r['pixels']
            for q,k in v.nonzero().tolist():
                shift=float(((updated[i,q,KP[k]]-torch.tensor(raw['global_valid_xy'][j,k]))*r['pixels']).norm())
                if shift>1e-3:changed.append(dict(index=i,query=q,point=KP[k]+1,shift_from_global=shift,global_error=float((global_px[k]-r['gt'][q,k]*r['pixels']).norm()),roi_error=float(e['specialist_roi'][q,k])))
        if m['kind']=='dual':
            qq=conf.nonzero().flatten().tolist()
            for ix,q in enumerate(qq):
                for o in qq[ix+1:]:
                    for k in range(1,5):
                        pairs.append(dict(index=i,q=q,other=o,point=KP[k]+1,distances={a:float(((p[i,q,KP[k]]-p[i,o,KP[k]])*r['pixels']).norm()) for a,p in pred.items()}))
            for q,k in v.nonzero().tolist():
                if k==0:continue
                others=[o for o in range(6) if o!=q and v[o,k]]
                if not others:continue
                for a,p in pred.items():
                    own=float(e[a][q,k]);other=min(float(((p[i,q,KP[k]]-r['gt'][o,k])*r['pixels']).norm()) for o in others);cross[a]['known_points']+=1;cross[a]['closer_other_by5']+=int(other+5<own)
    pair_summary={a:dict(pairs=len(pairs),same_point_within4=sum(r['distances'][a]<=4 for r in pairs)) for a in pred};write_json(OUT/'per_image.json',per_image);write_json(OUT/'roi_changes_from_global_real.json',changed);write_json(OUT/'dual_pair_distances.json',pairs)
    coverage_summary={}
    for kind in ['real','single','dual']:
        cc=[z for z in coverage if z['kind']==kind];inside=[z['roi_error'] for z in cc if z['gt_inside_box']]
        coverage_summary[kind]=dict(known=len(cc),gt_outside_box=sum(not z['gt_inside_box'] for z in cc),global_peak_outside_box=sum(not z['global_peak_inside_box'] for z in cc),roi_within4px_box_edge=sum(z['distance_to_box_edge_px']<=4 for z in cc),raw_score_nonpositive=sum(z['raw_heatmap_score']<=0 for z in cc),raw_score_median=float(np.median([z['raw_heatmap_score'] for z in cc])),gt_inside_roi_mean_error=float(np.mean(inside)))
    write_json(OUT/'box_coverage_points.json',coverage)
    result=dict(summary=summary,box_coverage=coverage_summary,dual_other_identity_diagnostic=cross,dual_confident_query_pairs=pair_summary,real_known_points_changed_by_roi=len(changed),inference=inference,limitations='Fixed originalboxes/counts, all6queriesread; GTmatchingafterforwardonly. ConditionalknownpointsnotAP. Singlemouse-trainedspecialist, unequaltraining/compute. No datasetkindgate orpromotion.')
    write_json(OUT/'results.json',result)
    selections=[]
    for label,predicate in [('real_bottom',lambda m:m['physical_bottom']),('single',lambda m:m['kind']=='single'),('dual',lambda m:m['kind']=='dual')]:
        candidates=[r for r in per_image if r['known'] and predicate(r['metadata'])]
        for mode,row in [('gain',min(candidates,key=lambda r:r['delta'])),('regression',max(candidates,key=lambda r:r['delta']))]:selections.append(dict(label=label,mode=mode,**row))
    colors=['#ff7744','#00bbff','#ff77dd','#aaee44','#ffaa00','#7777ff']
    for z in selections:
        i=z['index'];r=rows[i];rec=manifest[i];im=Image.open(ROOT/rec['image']).convert('RGB');w,h=im.size;tile=Image.new('RGB',(2220,960),'white');d=ImageDraw.Draw(tile)
        for col,name in enumerate(['baseline','specialist_roi','evaluation_GT']):
            panel=im.copy();di=ImageDraw.Draw(panel)
            for q,box in enumerate(r['baseline']['boxes']):
                score=float(r['baseline']['objectness_logits'][q].sigmoid());cx,cy,bw,bh=(box*torch.tensor([w,h,w,h])).tolist();color=colors[q] if score>=.5 else '#888888';di.rectangle((cx-bw/2,cy-bh/2,cx+bw/2,cy+bh/2),outline=color,width=2 if score>=.5 else 1);di.text((cx-bw/2,cy-bh/2),f'q{q} {score:.2f}',font=font(12),fill=color)
                for k,kp in enumerate(KP):
                    if name=='evaluation_GT':
                        if not r['valid'][q,k]:continue
                        x,y=(r['gt'][q,k]*r['pixels']).tolist();di.line((x-6,y,x+6,y),fill='#00ff66',width=3);di.line((x,y-6,x,y+6),fill='#00ff66',width=3)
                    else:
                        x,y=(pred[name][i,q,kp]*r['pixels']).tolist();di.ellipse((x-4,y-4,x+4,y+4),outline=color,width=2 if score>=.5 else 1)
                    if score>=.5 or name=='evaluation_GT':di.text((x+5,y+3),str(kp+1),font=font(13),fill='#00ff66' if name=='evaluation_GT' else color)
            panel.thumbnail((720,795));tile.paste(panel,(col*740+(740-panel.width)//2,100));d.text((col*740+10,10),name,font=font(25),fill='black');d.text((col*740+10,50),f'{z["label"]} {z["mode"]} view{r["metadata"]["view"]}',font=font(21),fill='black')
        d.text((15,915),'All6 boxes/query outputs shown; gray score<0.5; GT only in right evaluation panel.',font=font(23),fill='black');file=f'{z["label"]}_{z["mode"]}_{i}.jpg';tile.save(OUT/file,quality=95);z['file']=file
    write_json(OUT/'visual_examples.json',selections)
    page=['<!doctype html><meta charset="utf-8"><title>预测框约束的冻结定位教师</title><style>body{font:17px system-ui;max-width:2150px;margin:30px auto}img{max-width:100%}td,th{border:1px solid #aaa;padding:8px}table{border-collapse:collapse}</style><h1>冻结定位教师 + 预测框：单双鼠跨域诊断</h1><p>144张固定开发输入（48真实、32合成单鼠、64双鼠），逐像素验证从实际loader张量到PNG往返误差为零。GT只在独立评价文件中，推理读取所有6个模型预测框，框内读取同一张HRNet热图的鼻尖/四脚峰值；无数据集类别开关、置信度筛点、GT裁剪或实例数量输入。</p><p>右/下补黑到32倍数、不缩放，热图读取排除补边；stride4单元中心与原生解码核验一致。7项单测通过，144图默认关闭精确不变、其余18点精确不变、冻结权重哈希不变。无训练、不晋升。额外HRNet前向/解码合计8.15秒，不是与原1B模型公平计算预算的架构消融。</p><table><tr><th>范围</th><th>方法</th><th>known</th><th>mean↓</th><th>P95↓</th><th>PCK32↑</th></tr>']
    for kind in ['all','real','single','dual','real_bottom_feet']:
        for a,s in summary.items():
            t=s[kind];page.append(f'<tr><td>{kind}</td><td>{a}</td><td>{t["known"]}</td><td>{t["mean"]:.3f}</td><td>{t["p95"]:.3f}</td><td>{t["pck32"]:.3f}</td></tr>')
    page+=['</table><p>条件误差只计算原框匹配合格的相同已知点，不是检测AP；框数量、分数完全没改，单鼠教师不等同多目标模型。开发集非外部测试。</p><h2>跨鼠与重复点诊断</h2><p>更接近另一鼠同名GT点是评价诊断，不用于修改预测。不同query相同点距离≤4px也可能来自接触/重叠，不自动视为错误。</p><pre>'+json.dumps(dict(cross=cross,pairs=pair_summary,real_ROI_changes=len(changed)),indent=2)+'</pre><h2>原始六查询可视化</h2><p>左原模型，中预测框约束专门热图，右相同已知GT；展示改进和退步极值，非频率估计。</p>']
    page+=['<h2>结论：拒绝直接组合</h2><p>真实底部足点改善，但硬框限制会截掉鼻尖/肢端；IoU合格并不保证框覆盖全部关键点。合成单鼠和双鼠显著退步。双鼠图例的gain仅表示本组变化最小，仍退步16.55px，不能称为改善。不同query没有输出同一坐标也不等于实例身份正确：更接近另一鼠的足点从12/304升至55/304。不按数据集类别偷偷开关，不晋升模型。</p><h3>框覆盖与热图诊断（仅评价）</h3><p>原始热图响应值不是准确率概率；贴框边也不是错误的充分证据。即使GT在框内仍定位差，说明扩大框不能单独解释/解决域与尺度问题。禁止用这些GT统计选择每图预测。</p><pre>'+json.dumps(coverage_summary,indent=2)+'</pre>']
    for z in selections:page.append(f'<h3>{z["label"]} {"变化最小" if z["mode"]=="gain" else "退步最大"}: {z["delta"]:+.2f}px</h3><img src="{z["file"]}">')
    page.append('<p><a href="results.json">完整结果</a> | <a href="per_image.json">逐图指标</a> | <a href="visual_examples.json">图例索引</a> | <a href="roi_changes_from_global_real.json">ROI改变的真实已知点</a></p>');(OUT/'index.html').write_text('\n'.join(page),encoding='utf-8')
    for p in OUT.iterdir():
        if p.is_file():shutil.copy2(p,DEST/p.name)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
