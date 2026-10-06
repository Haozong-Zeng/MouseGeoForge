"""Three selector seeds, actual subset stress and diagnostic visual gallery."""
import json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from PIL import Image,ImageDraw
from training.finetune_blendedmvs import load_scene
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';OUT=CYCLE/'bounded_prior_repeat_review_v1';DEST=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005')/OUT.name
KP=[0,14,15,16,17]

def main():
    assert not OUT.exists();OUT.mkdir()
    first=json.loads((CYCLE/'bounded_prior_candidate_pilot_v1/results.json').read_text());repeat=json.loads((CYCLE/'bounded_prior_selector_seed_repeats_v1/results.json').read_text());stress=json.loads((CYCLE/'bounded_prior_viewstress_v1/results.json').read_text())
    rows=[dict(r,seed=17,tag='seed17_'+r['tag']) for r in first['results']]+[dict(r,seed=int(r['tag'].split('_')[0][4:])) for r in repeat['results']]
    selected=[r for r in rows if 'validation_matched' in r['tag']]
    agg={}
    for arm in ['unary','geometry']:
        m=[r for r in selected if '_'+arm+'_' in r['tag']]
        agg[arm]={k:dict(mean=float(np.mean([r['metrics'][k]['mean_error'] for r in m])),std=float(np.std([r['metrics'][k]['mean_error'] for r in m],ddof=1))) for k in ['real/reliable','real/view0/feet']}
    d=torch.load(CYCLE/'local_candidate_selector_pilot_v1/validation_inputs.pt',map_location='cpu',weights_only=False)
    firstpred=np.load(CYCLE/'bounded_prior_candidate_pilot_v1/predictions.npz');repeatpred=np.load(CYCLE/'bounded_prior_selector_seed_repeats_v1/predictions.npz')
    pred={17:firstpred['geometry_validation_matched'],29:repeatpred['seed29_geometry_validation_matched'],43:repeatpred['seed43_geometry_validation_matched']}
    ps=json.loads((CYCLE/'bounded_prior_selector_seed_repeats_v1/seed43_geometry_validation_matched_points.json').read_text());bottom=[p for p in ps if d['metadata'][p['row']]['kind']=='real' and d['metadata'][p['row']]['view']==0 and p['point']!=1]
    worst=sorted(bottom,key=lambda p:p['error']-p['baseline_error'],reverse=True);ids=list(dict.fromkeys([0,15,worst[0]['row']]));gallery=[]
    for i in ids:
        m=d['metadata'][i];w,h=map(int,d['pixels'][i]);f=load_scene(ROOT/m['scene']).frames[m['view']];rgb=Image.open(f.image_path).convert('RGB').resize((w,h));canvas=Image.new('RGB',(3*w,h+44),'white');draw=ImageDraw.Draw(canvas)
        for col,seed in enumerate([17,29,43]):
            canvas.paste(rgb,(col*w,44));draw.text((col*w+10,12),f'geometry selector seed{seed}, frozen baseline17',fill='black')
            for q in np.flatnonzero(d['logits'][i].sigmoid().numpy()>=.5):
                color=['#ff5555','#00ccff','#eecc00','#ee00cc','#00ee99','#ff9900'][q]
                for k in KP:
                    x,y=pred[seed][i,q,k]*[w,h]+[col*w,44];draw.ellipse((x-5,y-5,x+5,y+5),outline=color,width=3);draw.text((x+5,y),str(k+1),fill=color)
            for t in range(2):
                for k in KP:
                    if d['point_valid'][i,t,k]:
                        x,y=d['gt_xy'][i,t,k].numpy()*[w,h]+[col*w,44];draw.line((x-4,y-4,x+4,y+4),fill='white',width=2);draw.line((x-4,y+4,x+4,y-4),fill='white',width=2)
        name=f'row{i:03d}_view{m["view"]}.jpg';canvas.save(OUT/name,quality=92);gallery.append(dict(row=i,path=name,**m))
    summary=dict(seed_rows=rows,aggregate=agg,stress=stress,seed43_worst_bottom=worst[:6],gallery=gallery,interpretation='Overall real improvement geometry vs matched unary in all3 selector seeds. Bottom improvement onlyseed17, worse seed29/43. All3 dual perquery unary worse than originalbaseline. Do not promote or select per-camera models. Only selector initialization/schedule varied, frozen baseline17; not external/source/encoder-seed replication.')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    page=['<!doctype html><meta charset="utf-8"><title>视角压力与三种子对照</title><style>body{font:17px system-ui;margin:30px}img{max-width:100%}td,th{padding:8px;border:1px solid #aaa}</style><h1>整体改善可重复，底视优势不稳定</h1><p>仅改变新选择头初始化与采样顺序，冻结原始编码器/检测/姿态基线seed17。不能当成不同视频、划分或全模型种子验证。每组800步，unary与geometry同初始化和样本顺序；最终步评价，无验证集选择。</p><table><tr><th>方案</th><th>真实可靠点</th><th>真实底视足点</th></tr>']
    for r in selected:page.append(f'<tr><td>{r["tag"]}</td><td>{r["metrics"]["real/reliable"]["mean_error"]:.2f}</td><td>{r["metrics"]["real/view0/feet"]["mean_error"]:.2f}</td></tr>')
    page.append('</table><p>总体geometry均优于同种子unary，但底视在29/43种子反而退步。均值±种子样本标准差（不是置信区间）：</p><pre>'+json.dumps(agg,indent=2)+'</pre><h2>重新计算的实际视角压力测试</h2><p>不是简单切分已有融合特征。顺序/重复完全一致；V1/V2/全无效回退精确。三视角固定取目标与编号最小的另两个物理相机，不按标签选视角。</p><table><tr><th>条件</th><th>观察图数</th><th>启用点数</th><th>真实误差</th><th>同一观察集基线</th><th>底视误差</th></tr>')
    for r in stress['results']:
        if not r['tag'].endswith('geometry'):continue
        m=r['metrics'];b=r['baseline_same_observed'];bottom=m.get('real/view0/feet',{}).get('mean_error');bv='-' if bottom is None else f'{bottom:.2f}'
        page.append(f'<tr><td>{r["tag"]}</td><td>{r["observed_images"]}</td><td>{r["eligible_points"]}</td><td>{m["real/reliable"]["mean_error"]:.2f}</td><td>{b["real/reliable"]["mean_error"]:.2f}</td><td>{bv}</td></tr>')
    page.append('</table><p>移除相机也移除该目标视角，必须和同一观察子集基线比较，不可据此挑选“最佳相机”。双鼠三个种子仍差于原始基线，不部署。下一步准备既有双鼠TRAIN上下文，先审计候选可达精度与框质量，再判断是否值得增加多实例训练，而非继续叠加类似小头。</p><p><a href="summary.json">完整结果与检查</a></p><p>下图包含已知错误与seed43最大新增底视错误；白叉GT、彩点所有可信query。诊断抽样，不是代表性随机图。</p>')
    for g in gallery:page.append(f'<h2>{g["path"]}</h2><img src="{g["path"]}">')
    (OUT/'index.html').write_text('\n'.join(page),encoding='utf-8');shutil.copytree(OUT,DEST);print(json.dumps(dict(aggregate=agg,gallery=gallery,worst=worst[:2])))

if __name__=='__main__':main()
