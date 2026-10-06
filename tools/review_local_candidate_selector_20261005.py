"""Learning curves, unchanged detection counts, and raw candidate failures."""
import collections,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from PIL import Image,ImageDraw
from training.finetune_blendedmvs import load_scene
from mousegeoforge.detection_readout_loss import match_detection
from mousegeoforge.pilot_metrics import box_iou
from run_depth_pose_ablation import write_json
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';BASE=CYCLE/'local_candidate_selector_pilot_v1';OUT=CYCLE/'local_candidate_selector_review_v1';DEST=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005');KP=[0,14,15,16,17]


def main():
    assert not OUT.exists();OUT.mkdir();torch.set_num_threads(2);results=json.loads((BASE/'results.json').read_text());data=torch.load(BASE/'validation_inputs.pt',map_location='cpu',weights_only=False)
    with np.load(BASE/'predictions.npz') as a:pred={arm:a['validation_'+arm].copy() for arm in ['baseline','unary','geometry']}
    points={arm:json.loads((BASE/f'{arm}_validation_points.json').read_text()) for arm in ['unary','geometry']};worst=sorted([p for p in points['geometry'] if data['metadata'][p['row']]['kind']=='real' and data['metadata'][p['row']]['view']==0 and p['point']!=1],key=lambda p:p['error']-p['baseline_error'],reverse=True)
    ids=[]
    for v in [0,5]:ids.append(next(i for i,m in enumerate(data['metadata']) if m['kind']=='real' and m['view']==v))
    for p in worst:
        if p['row'] not in ids:ids.append(p['row'])
        if len(ids)>=5:break
    images=[]
    for i in ids:
        m=data['metadata'][i];frame=load_scene(ROOT/m['scene']).frames[m['view']];w,h=map(int,data['pixels'][i]);rgb=Image.open(frame.image_path).convert('RGB').resize((w,h));canvas=Image.new('RGB',(3*w,h+44),'white');draw=ImageDraw.Draw(canvas);scores=data['logits'][i].sigmoid().numpy()
        for col,arm in enumerate(pred):
            canvas.paste(rgb,(col*w,44));draw.text((col*w+10,12),arm,fill='black')
            for q in np.flatnonzero(scores>=.5):
                color=['#ff5555','#00ccff','#eecc00','#ee00cc','#00ee99','#ff9900'][q]
                for kp in KP:
                    x,y=pred[arm][i,q,kp]*[w,h]+[col*w,44];draw.ellipse((x-5,y-5,x+5,y+5),outline=color,width=3);draw.text((x+5,y),str(kp+1),fill=color)
            for t in range(2):
                for kp in KP:
                    if data['point_valid'][i,t,kp]:
                        x,y=data['gt_xy'][i,t,kp].numpy()*[w,h]+[col*w,44];draw.line((x-5,y-5,x+5,y+5),fill='white',width=2);draw.line((x-5,y+5,x+5,y-5),fill='white',width=2)
        name=f'row{i:03d}_view{m["view"]}.jpg';canvas.save(OUT/name,quality=92);images.append(dict(path=name,**m,row=i))
    curves={}
    for arm in ['unary','geometry']:
        values=[r['loss'] for r in json.loads((BASE/f'{arm}_history.json').read_text())];curves[arm]=dict(first100=float(np.mean(values[:100])),last100=float(np.mean(values[-100:])),previous100=float(np.mean(values[-200:-100])),blocks=[float(np.mean(values[i:i+50])) for i in range(0,len(values),50)])
    harms={}
    for arm,rows in points.items():
        for group in ['real_reliable','real_bottom_feet']:
            subset=[p for p in rows if data['metadata'][p['row']]['kind']=='real' and (group=='real_reliable' or data['metadata'][p['row']]['view']==0 and p['point']!=1)]
            harms[arm+'/'+group]=dict(points=len(subset),changed=sum(p['changed'] for p in subset),improved_gt16=sum(p['baseline_error']-p['error']>16 for p in subset),harmed_gt16=sum(p['error']-p['baseline_error']>16 for p in subset),worst_increase=max(p['error']-p['baseline_error'] for p in subset))
    floors={};counts={}
    for role in ['train','validation']:
        d=torch.load(BASE/f'{role}_inputs.pt',map_location='cpu',weights_only=False);prob=d['target_prob'];mask=d['train_valid'];ent=-(prob*prob.clamp_min(1e-12).log()).sum(-1);floors[role]=float(((ent*mask).sum((-1,-2))/mask.sum((-1,-2)).clamp_min(1)).mean())
        matched=match_detection(dict(instance_boxes_cxcywh=d['boxes'],instance_objectness_logits=d['logits']),d['gt_boxes'],d['box_valid']&d['instance_valid']);acc=collections.defaultdict(lambda:dict(images=0,gt=0,pred=0,tp=0))
        for i,m in enumerate(d['metadata']):
            group=m['kind'];a=acc[group];a['images']+=1;a['gt']+=int((d['box_valid'][i]&d['instance_valid'][i]).sum());a['pred']+=int((d['logits'][i].sigmoid()>=.5).sum())
            for q,t in zip(*matched[i]):a['tp']+=int(d['logits'][i,q].sigmoid()>=.5 and box_iou(d['boxes'][i,q:q+1],d['gt_boxes'][i,t:t+1])[0,0]>=.5)
        counts[role]=dict(acc)
    summary=dict(curves=curves,target_entropy_floor=floors,validation_harms=harms,unchanged_detector_counts=counts,worst_bottom_points=worst[:12],gallery=images,
        limitation='GTbox-conditioned errors on reused development data. Log loss approaching target entropy does not guarantee low mean hard-choice pixel error; near/catastrophic miss tradeoffs differ. No checkpoint selection or production promotion.')
    write_json(OUT/'summary.json',summary)
    page=['<!doctype html><meta charset="utf-8"><title>学习候选选择首轮</title><style>body{font:17px system-ui;margin:30px}img{max-width:100%}td,th{padding:8px;border:1px solid #aaa}</style><h1>800步候选选择：有收益，也有严重底视退步</h1><p>同初始化、相同图像调度、22657参数、原模型完全冻结。unary只用本视角局部信息；geometry增加其他视角几何特征。两者支持可用条件相同。GT只用于损失/评价，不生成推理候选。</p><table><tr><th>方案/数据</th><th>真实可靠点</th><th>底视足点</th><th>合成可靠点</th></tr>']
    for r in results['results']:
        m=r['metrics'];page.append(f'<tr><td>{r["arm"]}/{r["role"]}</td><td>{m["real/reliable"]["mean_error"]:.2f}</td><td>{m["real/view0/feet"]["mean_error"]:.2f}</td><td>{m["synthetic/reliable"]["mean_error"]:.2f}</td></tr>')
    page.append('</table><p>geometry开发验证底视平均误差升至61.16px，即使PCK32从0.375升至0.531，仍不可采用。unary底视降至36.12px，但当前还依赖匹配实验的支持门槛；未证明单视角或多鼠可用。合成开发验证仅两视角全部回退，不能当作融合泛化成绩。</p><h2>损失与错误尾部</h2><pre>'+json.dumps(dict(curves=curves,entropy_floor=floors,harms=harms),indent=2,ensure_ascii=False)+'</pre><p>以下包括固定首个底视/侧视样本，以及按geometry新增误差选择的诊断样本；后者并非代表性随机抽样。白叉为已有标注，彩色点为全部可信预测。</p>')
    for im in images:page.append(f'<h2>{im["path"]}</h2><p>{im["scene"]}</p><img src="{im["path"]}">')
    (OUT/'index.html').write_text('\n'.join(page),encoding='utf-8');shutil.copytree(OUT,DEST/OUT.name);print(json.dumps(summary))


if __name__=='__main__':main()
