"""Mixed-dual trial gallery, box-escape diagnosis and next architecture evidence."""
import json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from PIL import Image,ImageDraw
from training.finetune_blendedmvs import load_scene
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';OUT=CYCLE/'dual_train_inclusion_control_v1';DEST=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005')/OUT.name;KP=[0,14,15,16,17]


def main():
    r=json.loads((OUT/'results.json').read_text());pred=np.load(OUT/'predictions.npz');d=torch.load(CYCLE/'dual_unary_diagnostic_v1/inputs.pt',map_location='cpu',weights_only=False)
    ps=json.loads((OUT/'dual_mixed_validation_dual_points.json').read_text());worst=sorted(ps,key=lambda p:p['error']-p['baseline_error'],reverse=True);ids=list(dict.fromkeys([13,worst[0]['row'],0]));gallery=[]
    for i in ids:
        m=d['metadata'][i];w,h=map(int,d['pixels'][i]);f=load_scene(ROOT/m['scene']).frames[m['view']];rgb=Image.open(f.image_path).convert('RGB').resize((w,h));canvas=Image.new('RGB',(3*w,h+44),'white');draw=ImageDraw.Draw(canvas)
        for col,arm in enumerate(['baseline','single_only','dual_mixed']):
            canvas.paste(rgb,(col*w,44));draw.text((col*w+10,12),arm,fill='black')
            for q in np.flatnonzero(d['logits'][i].sigmoid().numpy()>=.5):
                color=['#ff5555','#00ccff','#eecc00','#ee00cc','#00ee99','#ff9900'][q];cx,cy,bw,bh=d['boxes'][i,q].numpy()*[w,h,w,h];draw.rectangle((col*w+cx-bw/2,44+cy-bh/2,col*w+cx+bw/2,44+cy+bh/2),outline=color,width=2);draw.text((col*w+cx-bw/2,44+cy-bh/2),f'q{q}',fill=color)
                for k in KP:
                    x,y=pred[arm+'_validation_dual'][i,q,k]*[w,h]+[col*w,44];draw.ellipse((x-5,y-5,x+5,y+5),outline=color,width=3);draw.text((x+5,y),str(k+1),fill=color)
            for t in range(2):
                for k in KP:
                    if d['point_valid'][i,t,k]:
                        x,y=d['gt_xy'][i,t,k].numpy()*[w,h]+[col*w,44];draw.line((x-4,y-4,x+4,y+4),fill='white',width=2);draw.line((x-4,y+4,x+4,y-4),fill='white',width=2)
        name=f'row{i:03d}_view{m["view"]}.jpg';canvas.save(OUT/name,quality=92);gallery.append(dict(row=i,path=name,**m))
    box_audit={}
    for role,path in [('train',CYCLE/'dual_train_candidate_audit_v1/train_inputs.pt'),('validation',CYCLE/'dual_unary_diagnostic_v1/inputs.pt')]:
        data=torch.load(path,map_location='cpu',weights_only=False)
        for arm in ['baseline','single_only','dual_mixed']:
            rows=json.loads((OUT/f'{arm}_{role}_dual_points.json').read_text());groups={}
            for group,points in [('all',rows),('detector_hit',[p for p in rows if p['hit']]),('detector_miss',[p for p in rows if not p['hit']]),('harm_gt16',[p for p in rows if p['error']-p['baseline_error']>16]),('new_other_nearer',[p for p in rows if p.get('new_other_closer')])]:
                inside=gtinside=0
                for p in points:
                    i,q,t,k=p['row'],p['query'],p['target'],p['point']-1;box=data['boxes'][i,q].numpy();xy=pred[arm+'_'+role+'_dual'][i,q,k];gt=data['gt_xy'][i,t,k].numpy();inside+=int((np.abs(xy-box[:2])<=.625*box[2:]).all());gtinside+=int((np.abs(gt-box[:2])<=.625*box[2:]).all())
                groups[group]=dict(points=len(points),mean_error=float(np.mean([p['error'] for p in points])) if points else None,pred_inside_125percent_box=inside,gt_inside_125percent_box=gtinside)
            box_audit[role+'/'+arm]=groups
    curves={}
    for arm in ['single_only','dual_mixed']:
        v=[p['loss'] for p in json.loads((OUT/f'{arm}_history.json').read_text())];curves[arm]=dict(first100=float(np.mean(v[:100])),previous100=float(np.mean(v[-200:-100])),last100=float(np.mean(v[-100:])))
    summary=dict(gallery=gallery,curves=curves,box_escape_diagnostic=box_audit,worst_validation_dual=worst[:8],decision='Mixed dual TRAIN improves training much more than validation; still worse than unchanged baseline on dual, and synthetic single validation regresses. No promotion or blind step extension. Investigate instance-structured decoder/localization, not another candidate MLP.',research=[dict(name='GroupPose',paper='https://arxiv.org/abs/2308.07313',code='https://raw.githubusercontent.com/Atten4Vis/GroupPose/main/models/grouppose/transformer_deformable.py',relevance='Official decoder uses within-instance then same-type across-instance attention. Current project flattens all6x23 into one unrestricted self-attention; structured interaction is a testable next hypothesis, not proven cause.'),dict(name='ED-Pose',paper='https://arxiv.org/abs/2302.01593',code='https://github.com/IDEA-Research/ED-Pose',relevance='Explicit instance box and keypoint context is a second route. Current spatial readout already has soft predicted-box suppression, so do not present adding any box prior as novel or absent. No downloaded weights/full reproduction.')])
    (OUT/'review.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    page=['<!doctype html><meta charset="utf-8"><title>双鼠TRAIN加入对照</title><style>body{font:17px system-ui;margin:30px}img{max-width:100%}td,th{padding:8px;border:1px solid #aaa}</style><h1>加入双鼠训练有帮助，但未解决泛化与实例归属</h1><p>同初始权重、800步、固定有界unary结构；两组均使用本图预测score≥0.5的逐query启用。混合组每批16图仅替换最后2张为42张既有双鼠TRAIN（21场景），前14张完全相同。无GT数量/框/匹配进入推理。</p><table><tr><th>数据/方案</th><th>真实可靠点</th><th>真实底视足点</th><th>合成可靠点</th><th>另一实例同名点更近</th></tr>']
    for row in r['results']:
        m=row['metrics'];vals=[f'{m[k]["mean_error"]:.2f}' if k in m else '-' for k in ['real/reliable','real/view0/feet','synthetic/reliable']];page.append(f'<tr><td>{row["dataset"]}/{row["arm"]}</td>'+''.join(f'<td>{v}</td>' for v in vals)+f'<td>{row["other_nearer"]}</td></tr>')
    page.append('</table><p>双鼠训练25.61→21.67，开发验证仅26.36→25.56，仍差于原始23.30；合成单鼠验证17.92→19.88。存在分布适应/泛化差距，不能凭训练下降就继续加步数。检测器完全不变；条件误差不是AP，另一实例更近只用于诊断，不是可靠ID标签。没有覆盖原模型。</p><h2>下一步结构假设</h2><p>当前代码将6×23个关键点query展平后进行无约束自注意力。<a href="https://github.com/Atten4Vis/GroupPose">GroupPose</a>采用实例内、再跨实例同类关键点的注意力，值得做严格同参数/同数据对照。<a href="https://github.com/IDEA-Research/ED-Pose">ED-Pose</a>提供显式实例框与关键点上下文的另一方向。现有读出层已经有预测框软抑制，不应把“加框”当作全新能力。尚未证明自注意力就是错误原因，也未复现这些论文。</p><p><a href="review.json">损失、检测框逃逸及研究链接</a> | <a href="results.json">完整结果</a></p><p>下图为已知跨鼠错误、当前最大新增误差和首图；显示全部可信query，颜色不是跨相机身份。白叉为GT。</p>')
    for g in gallery:page.append(f'<h2>{g["path"]}</h2><img src="{g["path"]}">')
    (OUT/'index.html').write_text('\n'.join(page),encoding='utf-8');DEST.mkdir(exist_ok=True)
    for name in ['index.html','results.json','review.json','protocol.json']+[g['path'] for g in gallery]:shutil.copy2(OUT/name,DEST/name)
    print(json.dumps(dict(gallery=gallery,curves=curves,box_audit=box_audit['validation/dual_mixed'])))


if __name__=='__main__':main()
