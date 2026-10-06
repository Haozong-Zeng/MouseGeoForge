"""Visual failure analysis; exhaustive small-group geometry is diagnostic only."""
import itertools,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from mousegeoforge.multiview_camera_geometry import weighted_dlt,project_points
from mousegeoforge.candidate_selector_features import KP,build_selector_inputs
from audit_association_faults_20261005 import match
from run_depth_pose_ablation import write_json
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';SOURCE=CYCLE/'association_gated_fusion_audit_v1';CONTEXT=CYCLE/'dual_allscene_association_audit_v1'
OUT=CYCLE/'association_gated_fusion_review_v1';DEST=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005')/OUT.name
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def font(n):return ImageFont.truetype(FONT,n)


def geometry_diagnostic(groups,records):
    results=[]
    for row in records:
        if row['role']!='validation' or row['mode']!='natural':continue
        g=groups[row['group_index']];p=g['predictions'];d=g['evaluation_data'];views=g['record']['views'];matches=match(p,d)
        mapping=[dict(zip(q.tolist(),t.tolist())) for q,t in matches]
        for assoc in row['associations']:
            ids=[(views.index(v),q) for v,q in assoc]
            assert len(ids)==3,'This bounded diagnostic expects current frozen seed17 three-view groups'
            proj=np.stack([g['projections'][i] for i,q in ids]);pixel=np.array(g['pixels'])
            for ki,kp in enumerate(KP):
                candidates=np.stack([p['candidate_xy'][i,q,ki].numpy()*pixel for i,q in ids])
                evaluated=[]
                for comb in itertools.product(range(6),repeat=len(ids)):
                    uv=np.stack([candidates[j,k] for j,k in enumerate(comb)])
                    xyz=weighted_dlt(proj,uv)
                    if xyz is None:continue
                    projections=[project_points(pm,xyz) for pm in proj]
                    if any(float(z)<=0 for uvp,z in projections):continue
                    errors=np.array([np.linalg.norm(uvp-uv[j]) for j,(uvp,z) in enumerate(projections)])
                    evaluated.append((float(np.sqrt(np.mean(errors**2))),comb,xyz.tolist()))
                if not evaluated:continue
                evaluated.sort(key=lambda x:(x[0],x[1]));best=evaluated[0];node_details=[]
                for ni,(vi,q) in enumerate(ids):
                    t=mapping[vi].get(q)
                    if t is None or not d['point_valid'][vi,t,kp]:continue
                    gt=d['gt_xy'][vi,t,kp].numpy()*pixel
                    errors=np.linalg.norm(candidates[ni]-gt,axis=-1)
                    learned=row['outputs']['grouped_geometry'][vi,q,kp].numpy()*pixel
                    selected=int(np.linalg.norm(candidates[ni]-learned,axis=-1).argmin())
                    node_details.append(dict(view=views[vi],query=q,target=t,point=kp+1,candidate_error_px=errors.tolist(),
                        mass=p['candidate_mass'][vi,q,ki].tolist(),learned_candidate=selected,
                        baseline_error=float(errors[0]),learned_error=float(np.linalg.norm(learned-gt)),
                        joint_min_reprojection_candidate=best[1][ni],joint_error=float(errors[best[1][ni]]),
                        oracle_candidate=int(errors.argmin()),oracle_error=float(errors.min())))
                results.append(dict(group_index=row['group_index'],association=assoc,point=kp+1,valid_combinations=len(evaluated),
                    min_joint_reprojection_rms=best[0],min_joint_candidate_indices=best[1],nodes=node_details))
    nodes=[n for r in results for n in r['nodes']]
    summary=dict(groups_points=len(results),nodes=len(nodes),baseline_mean=float(np.mean([n['baseline_error'] for n in nodes])),
        learned_mean=float(np.mean([n['learned_error'] for n in nodes])),joint_min_reprojection_mean=float(np.mean([n['joint_error'] for n in nodes])),
        optimistic_per_view_oracle_mean=float(np.mean([n['oracle_error'] for n in nodes])),
        min_joint_rms_mean=float(np.mean([r['min_joint_reprojection_rms'] for r in results])),
        caveat='Post-hoc diagnostic on all five accepted frozen seed17 development groups; exhaustive6^3 candidate combinations, positive depth, fixed lowest reprojection RMS selection. No RGB model training or pose output changed. GT used ONLY to score and define explicitly optimistic per-view oracle; oracle is not an inference method.')
    return dict(summary=summary,details=results)


def panel(point,g,saved,name):
    im=Image.new('RGB',(1536,1020),'white');draw=ImageDraw.Draw(im);kp=point['point']-1;ki=KP.index(kp);views=g['record']['views'];group=next(a for a in saved['associations'] if (views[point['view_index']],point['query']) in [tuple(x) for x in a])
    qs={view:q for view,q in group};mapping=[dict(zip(q.tolist(),t.tolist())) for q,t in match(g['predictions'],g['evaluation_data'])]
    draw.text((15,10),f'scene {point["group_index"]} / {point["mode"]} / point {kp+1}: group-conditioned failure',font=font(22),fill='black')
    draw.text((15,43),'Blue circle=original; red X=geometry; green +=GT evaluation only; yellow dots=other candidates',font=font(18),fill='black')
    for vi,view in enumerate(views):
        x=vi%2*768;y=83+vi//2*465;image=Image.open(CONTEXT/g['images'][vi]).convert('RGB');im.paste(image,(x,y+26))
        if view not in qs:
            draw.text((x+8,y),f'view{view}: NOT in accepted group',font=font(17),fill='black');continue
        q=qs[view];p=g['predictions'];t=mapping[vi].get(q)
        draw.text((x+8,y),f'view{view} q{q}: accepted group',font=font(17),fill='black')
        cx,cy,bw,bh=p['boxes'][vi,q].tolist();draw.rectangle((x+(cx-bw/2)*768,y+26+(cy-bh/2)*432,x+(cx+bw/2)*768,y+26+(cy+bh/2)*432),outline='#fff',width=2)
        def xy(uv):return (x+float(uv[0])*768,y+26+float(uv[1])*432)
        for ci,uv in enumerate(p['candidate_xy'][vi,q,ki]):
            a,b=xy(uv);draw.ellipse((a-3,b-3,a+3,b+3),fill='#ffff00');draw.text((a+4,b+1),str(ci),font=font(13),fill='#ffff00',stroke_width=1,stroke_fill='black')
        a,b=xy(p['xy'][vi,q,kp]);draw.ellipse((a-8,b-8,a+8,b+8),outline='#00bfff',width=3)
        a,b=xy(saved['outputs']['grouped_geometry'][vi,q,kp]);draw.line((a-7,b-7,a+7,b+7),fill='#ff3333',width=3);draw.line((a-7,b+7,a+7,b-7),fill='#ff3333',width=3)
        if t is not None:
            a,b=xy(g['evaluation_data']['gt_xy'][vi,t,kp]);draw.line((a-9,b,a+9,b),fill='#00ff88',width=3);draw.line((a,b-9,a,b+9),fill='#00ff88',width=3)
    im.save(OUT/name,quality=95)


def main():
    OUT.mkdir(exist_ok=True);DEST.mkdir(exist_ok=True);r=json.loads((SOURCE/'results.json').read_text());points=json.loads((SOURCE/'points.json').read_text());saved=torch.load(SOURCE/'predictions.pt',map_location='cpu',weights_only=False);groups=torch.load(CONTEXT/'groups.pt',map_location='cpu',weights_only=False)
    diagnostic=geometry_diagnostic(groups,saved);write_json(OUT/'candidate_geometry_diagnostic.json',diagnostic)
    natural=points['validation/natural/fixed_original_assignment'];worst=sorted([p for p in natural if p['associated']],key=lambda p:p['errors']['grouped_geometry']-p['errors']['baseline'],reverse=True)
    examples=[]
    for i,point in enumerate(worst[:3]):
        row=next(x for x in saved if x['role']=='validation' and x['mode']=='natural' and x['group_index']==point['group_index']);name=f'worst{i+1}_scene{point["group_index"]}_point{point["point"]}.jpg'
        panel(point,groups[point['group_index']],row,name);examples.append(dict(file=name,point=point))
    write_json(OUT/'visual_examples.json',examples)
    page=['<!doctype html><meta charset="utf-8"><title>关联后融合：负结果</title><style>body{font:17px system-ui;margin:28px auto;max-width:1550px;padding:12px}img{max-width:100%}td,th{border:1px solid #bbb;padding:7px}table{border-collapse:collapse}</style>',
        '<h1>关联改善没有转化为定位改善：暂不启用融合</h1><p>固定历史seed17的800步关联头与已冻结校准阈值；重新由预测裁剪特征计算关联，不使用标注身份过滤组。主模型/阈值未调整，CPU离线测试。默认关闭；未关联节点及18个非目标关键点逐元素保持原样。7个新适配单测通过，并检查实际场景的相机顺序、1/2视角回退及旧分组重放。</p>',
        '<h2>自然双鼠开发集</h2><p>16场景64图，640个可评估可靠点；仅75点在接受组中。下表均为768×432输入像素，不是原图像素或毫米。条件GT框关联只在前向后评价，不等于检测AP。</p><table><tr><th>范围</th><th>方法</th><th>点数</th><th>均值px</th><th>P95px</th><th>比原始多错&gt;16px</th><th>最差新增误差px</th></tr>']
    for subset in ['all_matched','associated','feet_associated']:
        for arm,m in r['summary']['validation/natural/fixed_original_assignment'][subset].items():
            if arm in ['local','unsafe_composite','grouped_composite']:continue
            page.append(f'<tr><td>{subset}</td><td>{arm}</td><td>{m["points"]}</td><td>{m["mean"]:.3f}</td><td>{m["p95"]:.3f}</td><td>{m["harms_vs_original_over16"]}</td><td>{m["worst_harm_vs_original"]:.3f}</td></tr>')
    page+=['</table><p>75个已关联点：原始18.864px，组内局部修正21.502px，组内几何修正21.123px；几何略好于错误的局部修正，但仍差于原始。最差额外误差57.17px。此处组内已知身份一致，因此不能再把错误全部归结为跨鼠关联。</p><h2>故障注入</h2><p>人工只保留单个预测，不是真实遮挡实验。保守分组减少以前的大幅跨鼠跳点，但未胜过不修正的原始结果。</p><table><tr><th>保留规则</th><th>方法</th><th>点数</th><th>平均px</th><th>新增误差&gt;16px</th><th>最差新增px</th></tr>']
    for mode in ['highest_confidence','leftmost_box','alternating_confidence_rank']:
        for arm in ['baseline','unsafe_composite','grouped_geometry']:
            m=r['summary'][f'validation/{mode}/fixed_original_assignment']['retained'][arm]
            page.append(f'<tr><td>{mode}</td><td>{arm}</td><td>{m["points"]}</td><td>{m["mean"]:.3f}</td><td>{m["harms_vs_original_over16"]}</td><td>{m["worst_harm_vs_original"]:.3f}</td></tr>')
    s=diagnostic['summary'];page+=['</table><h2>候选集合与几何诊断</h2><p>对固定接受的5组三视角、5个可靠点逐个穷举6³候选组合，要求正深度，并按最小联合重投影RMS选取。仅作失败诊断，未修改模型输出；另给出利用GT挑每视角最优候选的乐观上限，不能用于推理。</p>',
        f'<p>同75点：原始{s["baseline_mean"]:.3f}px，现有学习选择{s["learned_mean"]:.3f}px，单纯最小联合重投影{s["joint_min_reprojection_mean"]:.3f}px，GT候选乐观上限{s["optimistic_per_view_oracle_mean"]:.3f}px。联合解自身重投影RMS平均{s["min_joint_rms_mean"]:.3f}px；低自身重投影误差不等于定位正确。</p>',
        '<h2>最差三例（事后诊断，不是随机抽样）</h2><p>红X=几何选择，蓝圈=原始预测，绿+=GT仅用于评价，黄点=候选。白框由检测器预测。关键点标签与原始标注不改。</p>']
    for e in examples:page.append(f'<h3>{e["file"]}</h3><img src="{e["file"]}">')
    page+=['<p>下一步根据候选覆盖与几何诊断决定是否改目标：将“支持视角存在”与“支持观测可靠”分开，避免仅凭沿极线找到一个峰就认为支持成立；不再直接增加现有头的训练步数，不在此开发集上反复挑阈值。</p><p><a href="results.json">完整对照</a> | <a href="candidate_geometry_diagnostic.json">候选诊断明细</a> | <a href="visual_examples.json">图例来源</a></p>']
    (OUT/'index.html').write_text('\n'.join(page),encoding='utf-8');shutil.copy2(SOURCE/'results.json',OUT/'results.json')
    for path in OUT.iterdir():
        if path.is_file():shutil.copy2(path,DEST/path.name)
    print(json.dumps(s,indent=2));print(DEST/'index.html')


if __name__=='__main__':main()

