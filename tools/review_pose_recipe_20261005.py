"""Recipe factorial review and synthetic non-target accuracy audit (not relabeling)."""
import json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np,torch
from PIL import Image,ImageDraw
import review_pixel_adapter_four_arm_20261005 as gallery
from train_pixel_adapter_matched_20261005 import prepare_rows,KP
from mousegeoforge.detection_readout_loss import match_detection
from mousegeoforge.pilot_metrics import box_iou
from run_surface_curve_pilot_20261003 import load_item
from run_depth_pose_ablation import write_json
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';BASE=CYCLE/'pixel_adapter_matched_pilot_v1';NEW=CYCLE/'pose_recipe_factorial_v1';OUT=CYCLE/'pose_recipe_review_v1'
DEST=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005')/OUT.name
def read(p):return torch.load(p,map_location='cpu',weights_only=False)
def full_targets(rows):
    lookup={}
    for role in ['train','validation']:
        raw=read(CYCLE/'singleview_fusion_inputs_seed17_v1'/f'{role}.pt')
        for i,m in enumerate(raw['metadata']):lookup[role,m['record']['scene'],m['view']]={k:raw[k][i] for k in ['gt_xy','gt_boxes','point_valid','box_valid','instance_valid']}
        for g in read(CYCLE/('dual_train_allview_affinity_context_v1' if role=='train' else 'dual_allscene_association_audit_v1')/'groups.pt'):
            for vi,v in enumerate(g['record']['views']):lookup[role,g['record']['scene'],v]={k:g['evaluation_data'][k][vi] for k in ['gt_xy','gt_boxes','point_valid','box_valid','instance_valid']}
    gt=torch.zeros(len(rows),6,23,2);valid=torch.zeros(len(rows),6,23,dtype=torch.bool)
    for i,r in enumerate(rows):
        m=r['metadata'];d=lookup[m['role'],m['scene'],m['view']];b=r['baseline']
        qs,ts=match_detection(dict(instance_boxes_cxcywh=b['boxes'][None],instance_objectness_logits=b['objectness_logits'][None]),d['gt_boxes'][None],(d['box_valid']&d['instance_valid'])[None])[0]
        for q,t in zip(qs.tolist(),ts.tolist()):
            if b['objectness_logits'][q].sigmoid()<.5 or box_iou(b['boxes'][q:q+1],d['gt_boxes'][t:t+1])[0,0]<.5:continue
            gt[i,q]=d['gt_xy'][t];valid[i,q]=d['point_valid'][t]
        if m['kind']=='dual':valid[i,:,7:14]=False
        assert torch.equal(valid[i,:,KP],r['valid'])
    return gt,valid
def main():
    OUT.mkdir(exist_ok=True);DEST.mkdir(exist_ok=True);torch.set_num_threads(2)
    results=json.loads((NEW/'results.json').read_text());summary=results['summary'];data=read(BASE/'predictions.pt');pred=read(NEW/'predictions.pt')['predictions'];rows=prepare_rows();arms=list(summary)
    gallery.OUT=OUT;gallery.ARMS=arms;gallery.COLORS=dict(zip(arms,['#222222','#bb5533','#266abf','#bb3388','#249966']))
    curves=[dict(r,arm='mse_high') for r in json.loads((BASE/'curves.json').read_text()) if r['arm']=='head_only']+json.loads((NEW/'curves.json').read_text());gallery.chart(curves,summary)
    gt,valid=full_targets(rows);pixels=data['pixels'];extra={};drift={}
    for arm,p in pred.items():
        e=((p-gt)*pixels[:,None,None]).norm(dim=-1);extra[arm]={}
        for stage in ['calibration','validation']:
            extra[arm][stage]={}
            for kind in ['single','dual']:
                ids=[i for i,r in enumerate(rows) if r['metadata']['stage']==stage and r['metadata']['kind']==kind];extra[arm][stage][kind]={}
                for label,ks in [('other18',[k for k in range(23) if k not in KP]),('tail',list(range(7,14))),('abdomen',[22]),('surface_back',[5,6])]:
                    z=e[ids][:,:,ks][valid[ids][:,:,ks]];extra[arm][stage][kind][label]=dict(n=z.numel(),mean=float(z.mean()) if z.numel() else None,p95=float(z.quantile(.95)) if z.numel() else None)
        if arm!='baseline':
            ids=data['evaluated_ids'];conf=torch.stack([rows[i]['baseline']['objectness_logits'].sigmoid()>=.5 for i in ids]);z=((p[ids]-pred['baseline'][ids])*pixels[ids,None,None]).norm(dim=-1)
            drift[arm]={label:float(z[:,:,ks][conf].mean()) for label,ks in [('tail',list(range(7,14))),('abdomen',[22]),('all23',list(range(23)))]}
    write_json(OUT/'synthetic_non_target_accuracy.json',extra);write_json(OUT/'confident_prediction_drift.json',drift)
    error={a:((p[:,:,KP]-data['gt'])*pixels[:,None,None]).norm(dim=-1) for a,p in pred.items()};examples=[]
    for label,predicate in [('real_bottom',lambda m:m['physical_bottom']),('real_side',lambda m:m['kind']=='real' and not m['physical_bottom']),('dual',lambda m:m['kind']=='dual')]:
        candidates=[(float(error['huber_low'][i,q,k]-error['baseline'][i,q,k]),i,q,k) for i,m in enumerate(data['metadata']) if m['stage']=='validation' and predicate(m) for q in range(6) for k in range(1,5) if data['valid'][i,q,k]]
        for mode,entry in [('best',min(candidates)),('worst',max(candidates))]:
            delta,i,q,k=entry;r=rows[i];m=r['metadata'];item=load_item(r['record']);vi=r['record']['views'].index(m['view']);rgb=item['batch']['images'][0,vi]
            image=Image.fromarray((rgb.permute(1,2,0).numpy().clip(0,1)*255).round().astype(np.uint8));w,h=image.size
            tile=Image.new('RGB',(2500,710),'white');td=ImageDraw.Draw(tile)
            for ci,(name,p) in enumerate(pred.items()):
                panel=image.copy();d=ImageDraw.Draw(panel);x,y=(p[i,q,KP[k]]*torch.tensor([w,h])).tolist();gx,gy=(data['gt'][i,q,k]*torch.tensor([w,h])).tolist()
                d.line((x,y,gx,gy),fill='yellow',width=2);d.ellipse((x-8,y-8,x+8,y+8),outline='red',width=4);d.line((gx-10,gy,gx+10,gy),fill='#00ff66',width=4);d.line((gx,gy-10,gx,gy+10),fill='#00ff66',width=4)
                panel.thumbnail((490,575));tile.paste(panel,(ci*500+(500-panel.width)//2,125));td.text((ci*500+10,10),name,font=gallery.font(24),fill=gallery.COLORS[name]);td.text((ci*500+10,46),f'error {error[name][i,q,k]:.2f}px',font=gallery.font(22),fill='black');td.text((ci*500+10,80),f'{label} view{m["view"]} point{KP[k]+1}',font=gallery.font(18),fill='black')
            file=f'{label}_{mode}.jpg';tile.save(OUT/file,quality=95);examples.append(dict(file=file,category=label,selection=mode,delta=delta,index=i,query=q,point=KP[k]+1,metadata=m,errors={a:float(e[i,q,k]) for a,e in error.items()}))
    write_json(OUT/'visual_examples.json',examples);write_json(OUT/'results.json',results)
    page=['<!doctype html><meta charset="utf-8"><title>损失配方与学习率对照</title><style>body{font:17px system-ui;max-width:2200px;margin:30px auto;padding:15px}img{max-width:100%}td,th{border:1px solid #aaa;padding:7px}table{border-collapse:collapse}</style>',
      '<h1>学习率降低能缓解回退，尚未解决底视脚点</h1><p>同初始姿态头、同400步和原图采样；MSE高学习率复用已完成的不可变对照，其余3组新训练。Huber配方还包含像素长宽缩放和坐标平均变化，并非单独检验鲁棒性。低学习率：空间头5e-5→1e-5，解码器1e-5→1e-6；其余权重衰减、五点监督、冻结预测框及实际RGB路径固定。四项损失单测通过，950425761冻结权重哈希不变。</p>',
      '<p>内部校验底脚原31.130，高LR MSE42.287，低LR MSE34.170，高LR Huber39.874，低LR Huber32.803：降低学习率可减轻两种配方的回退，但均未优于原模型。开发底脚43.465/45.853/41.494/45.027/43.465，只有24个已知点；低LR Huber底脚PCK32从0.500降到0.458，不能只看均值相等。</p>',
      '<p>低LR Huber在开发真实/单鼠/双鼠均值小幅下降到31.789/18.414/19.430，但真实PCK32仍略低。单种子、重复使用开发集、条件框匹配指标，不是外部测试或部署成功。暂不增加步数或替换正式模型。</p>']
    for phase in ['calibration','validation']:
        page.append(f'<h2>{phase} / 输入像素</h2><table><tr><th>范围</th><th>方法</th><th>known</th><th>mean↓</th><th>P95↓</th><th>PCK32↑</th></tr>')
        for kind in ['all','real','single','dual','real_bottom_feet']:
            for arm,s in summary.items():
                v=s[phase][kind];page.append(f'<tr><td>{kind}</td><td>{arm}</td><td>{v["known"]}</td><td>{v["mean"]:.3f}</td><td>{v["p95"]:.3f}</td><td>{v["pck32"]:.3f}</td></tr>')
        page.append('</table>')
    page += ['<h2>曲线</h2><img src="calibration_curves.png"><img src="fit_curves.png">','<h2>已有合成GT的非目标点误差</h2><p>不是坐标漂移，而是相对原有合成GT的条件误差。双鼠尾巴仍禁用；真实背部不作为精确GT。other18指五个当前监督点之外的索引集合，实际仅计算有效点；双鼠不一定有18个。</p><table><tr><th>范围</th><th>方法</th><th>有效点数</th><th>其他点均值</th></tr>']
    for stage in ['calibration','validation']:
        for kind in ['single','dual']:
            for arm in arms:
                z=extra[arm][stage][kind]['other18'];page.append(f'<tr><td>{stage}/{kind}</td><td>{arm}</td><td>{z["n"]}</td><td>{z["mean"]:.3f}</td></tr>')
    page += ['</table><h2>低LR Huber相对原模型的极值脚点</h2><p>红圈预测、绿十字评价GT，事后极值不代表频率；所有五组同图。没有GT裁剪进入模型。</p>']
    for e in examples:page.append(f'<h3>{e["category"]} {e["selection"]}: {e["delta"]:+.2f}px</h3><img src="{e["file"]}">')
    page.append('<p><a href="results.json">完整指标与成本</a> | <a href="synthetic_non_target_accuracy.json">其他点精度</a> | <a href="confident_prediction_drift.json">坐标漂移（不是精度）</a> | <a href="visual_examples.json">图例索引</a></p>')
    (OUT/'index.html').write_text('\n'.join(page),encoding='utf-8')
    for p in OUT.iterdir():
        if p.is_file():shutil.copy2(p,DEST/p.name)
    print(json.dumps(dict(non_target=extra,drift=drift)));print(DEST/'index.html')
if __name__=='__main__':main()
