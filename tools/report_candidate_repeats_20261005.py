"""Summarize fixed-rule repeats and inventory existing synthetic TRAIN views."""
import collections,datetime as dt,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from PIL import Image,ImageDraw
from training.finetune_blendedmvs import load_scene
from run_depth_pose_ablation import write_json,fingerprint
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';SOURCE=CYCLE/'singleview_fusion_inputs_seed17_v1';REPEAT=CYCLE/'epipolar_candidates_seed_repeats_v1'
OUT=CYCLE/'candidate_repeats_review_v1';DEST=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005');KP=[0,14,15,16,17]


def main():
    assert not OUT.exists();OUT.mkdir();torch.set_num_threads(2)
    result=json.loads((REPEAT/'results.json').read_text());seed17=json.loads((CYCLE/'epipolar_candidates_pilot_v1/results.json').read_text());rgb=json.loads((CYCLE/'epipolar_candidate_rgb_audit_v1/results.json').read_text());stress=json.loads((CYCLE/'epipolar_candidate_stress_v1/results.json').read_text())
    rows=[dict(r,seed=17) for r in seed17['results']]+result['results'];data=torch.load(SOURCE/'validation.pt',map_location='cpu',weights_only=False);pred={};logits={}
    for seed in [17,29,43]:
        path=CYCLE/'epipolar_candidates_pilot_v1/validation_predictions.npz' if seed==17 else REPEAT/f'seed{seed}_validation_predictions.npz'
        with np.load(path) as a:
            pred[seed]={k:a[k].copy() for k in ['baseline_xy','epipolar_local_xy','cohort']};logits[seed]=data['objectness_logits'].numpy() if seed==17 else a['logits'].copy()
    scene=next(data['metadata'][i]['record']['scene'] for i in np.flatnonzero(pred[17]['cohort']) if data['metadata'][i]['record']['kind']=='real');images=[]
    for i,m in enumerate(data['metadata']):
        view=int(m['view'])
        if m['record']['scene']!=scene or view not in [0,2,5]:continue
        frame=load_scene(ROOT/scene).frames[view];w,h=map(int,data['pixels'][i]);source=Image.open(frame.image_path).convert('RGB').resize((w,h));canvas=Image.new('RGB',(3*w,2*(h+42)),'white');draw=ImageDraw.Draw(canvas)
        for col,seed in enumerate([17,29,43]):
            score=1/(1+np.exp(-logits[seed][i]))
            for row,arm in enumerate(['baseline','epipolar_local']):
                ox=col*w;oy=row*(h+42);canvas.paste(source,(ox,oy+42));draw.text((ox+10,oy+12),f'seed{seed} {arm}',fill='black')
                for q in np.flatnonzero(score>=.5):
                    color=['#ff5555','#00ccff','#eecc00','#ee00cc','#00ee99','#ff9900'][q]
                    for kp in KP:
                        x,y=pred[seed][arm+'_xy'][i,q,kp]*[w,h]+[ox,oy+42];draw.ellipse((x-5,y-5,x+5,y+5),outline=color,width=3);draw.text((x+5,y),str(kp+1),fill=color)
                for t in range(2):
                    for kp in KP:
                        if data['point_valid'][i,t,kp]:
                            x,y=data['gt_xy'][i,t,kp].numpy()*[w,h]+[ox,oy+42];draw.line((x-5,y-5,x+5,y+5),fill='white',width=2);draw.line((x-5,y+5,x+5,y-5),fill='white',width=2)
        name=f'cam{chr(65+view)}_three_seeds.jpg';canvas.save(OUT/name,quality=92);images.append(name)
    # Read-only inventory, using actual parsed camera frames, not file counts.
    inventory=[]
    for g in torch.load(CYCLE/'camera_geometry_audit_v2/train_cameras.pt',map_location='cpu',weights_only=False):
        if g['kind']!='synthetic':continue
        sceneobj=load_scene(ROOT/g['scene']);available=sorted(sceneobj.frames);inventory.append(dict(scene=g['scene'],role='train',existing_views=g['physical_views'],available_views=available,
            additional_views=[v for v in available if v not in g['physical_views']],image_sizes={str(v):list(Image.open(sceneobj.frames[v].image_path).size) for v in available},
            camera_missing=[v for v in available if sceneobj.frames[v].cam_path is None]))
    write_json(OUT/'synthetic_train_view_inventory.json',inventory)
    doc=['<!doctype html><meta charset="utf-8"><title>候选路线三种子与RGB复验</title><style>body{font:17px system-ui;margin:35px}img{max-width:100%}td,th{padding:8px;border:1px solid #aaa}</style><h1>固定规则三种子复验</h1><p>未增加训练、未搜索开发验证参数；使用既有seed17/29/43检测头和姿态头。误差单位为模型输入像素，GT框仅用于前向后评价关联，不是端到端AP或外部测试。</p><table><tr><th>种子/数据</th><th>可靠点 基线→极线候选</th><th>底视足点 基线→极线候选</th><th>可靠点gated PCK32</th></tr>']
    for r in sorted(rows,key=lambda r:(r['seed'],r['role'])):
        m=r['metrics'];a=m['baseline/real/reliable'];b=m['epipolar_local/real/reliable'];c=m['baseline/real/camA/feet'];d=m['epipolar_local/real/camA/feet'];doc.append(f'<tr><td>{r["seed"]}/{r["role"]}</td><td>{a["error_mean"]:.2f} → {b["error_mean"]:.2f}</td><td>{c["error_mean"]:.2f} → {d["error_mean"]:.2f}</td><td>{a["gated_pck32"]:.3f} → {b["gated_pck32"]:.3f}</td></tr>')
    doc.append(f'</table><p>三个种子在开发验证都有改善，但训练集底视足点均值均退步；不能宣称已解决或直接推广。</p><h2>工程验证</h2><p>单视角/双视角精确回退，倒序及重复视角输出一致。7张固定RGB样本（真实5、合成2）重跑，原生空间特征和实例token与缓存完全一致；最终候选归一化坐标最大差{rgb["max_candidate_delta"]:.3g}。这是工程一致性检查，不是独立精度测试。</p><p><a href="../epipolar_candidate_stress_v1/index.html">视角压力测试</a> | <a href="../epipolar_candidate_rgb_audit_v1.json">RGB重放明细</a></p><p>增加支持视角并非总有利：固定三视角整体误差27.47，而全部视角30.56；底视则全部视角更好。不要据此在开发验证上挑最佳相机组合。</p><h2>下一步</h2><p>先补齐既有合成TRAIN场景的多视角上下文，保持相同场景与动作划分；现有两视角合成验证只能回退，无法检验融合的合成域代价。之后做有监督的候选选择/可靠性控制对照，不继续硬三角化替换坐标。原标注及正式模型不变。</p>')
    for name in images:doc.append(f'<h2>{name}</h2><img src="{name}">')
    (OUT/'index.html').write_text('\n'.join(doc),encoding='utf-8');write_json(OUT/'summary.json',dict(results=rows,rgb=rgb,stress=stress['stress'],inventory_scenes=len(inventory),available_view_counts=dict(collections.Counter(len(r['available_views']) for r in inventory)),new_views=sum(len(r['additional_views']) for r in inventory),script_sha256=fingerprint(Path(__file__))));shutil.copytree(OUT,DEST/OUT.name)
    print(json.dumps(dict(report=str(DEST/OUT.name/'index.html'),inventory_scenes=len(inventory),available_view_counts=dict(collections.Counter(len(r['available_views']) for r in inventory)),new_views=sum(len(r['additional_views']) for r in inventory),max_rgb_delta=rgb['max_candidate_delta'])))


if __name__=='__main__':main()
