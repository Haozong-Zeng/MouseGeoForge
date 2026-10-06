"""Show zero-coverage rejection and failed frozen predicted-ROI appearance."""
import json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'vggt-omega')]
import torch
from PIL import Image,ImageDraw
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';OUT=CYCLE/'partial_and_prequery_review_v1';DEST=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005')/OUT.name


def main():
    OUT.mkdir(exist_ok=True);partial=json.loads((CYCLE/'partial_association_control_v1/results.json').read_text());roi=json.loads((CYCLE/'prequery_roi_affinity_diagnostic_v1/results.json').read_text());runtime=json.loads((CYCLE/'prequery_roi_descriptors_v1/results.json').read_text())
    g=torch.load(CYCLE/'dual_allscene_association_audit_v1/groups.pt',map_location='cpu',weights_only=False)[0]
    provenance=json.loads((CYCLE/'dual_allscene_association_audit_v1/results.json').read_text())['groups'][0]
    canvas=Image.new('RGB',(4*280,2*250),'white');draw=ImageDraw.Draw(canvas)
    for vi,view in enumerate(g['record']['views']):
        rgb=Image.open(CYCLE/'dual_allscene_association_audit_v1'/g['images'][vi]).convert('RGB');w,h=rgb.size
        for qi,q in enumerate(torch.where(g['predictions']['objectness_logits'][vi].sigmoid()>=.5)[0].tolist()):
            box=g['predictions']['boxes'][vi,q].numpy();cx,cy,bw,bh=box*[w,h,w,h]
            bounds=(max(0,int(cx-bw/2)),max(0,int(cy-bh/2)),min(w,int(cx+bw/2)+1),min(h,int(cy+bh/2)+1))
            crop=rgb.crop(bounds);crop.thumbnail((270,205));x=vi*280;y=qi*250
            canvas.paste(crop,(x+(280-crop.width)//2,y+40));draw.text((x+6,y+8),f'view{view} q{q}: predicted crop',fill='black')
            det=next(x for x in provenance['detections'][vi]['predictions'] if x['query']==q)
            draw.text((x+6,y+23),f'posthoc target={det["evaluation_target"]} IoU={det["iou"]:.2f}',fill='black')
    canvas.save(OUT/'first_scene_predicted_crops.jpg',quality=95)
    page=['<!doctype html><meta charset="utf-8"><title>保守关联与外观检查</title><style>body{font:17px system-ui;margin:30px}img{max-width:100%}td,th{padding:8px;border:1px solid #aaa}</style><h1>两个负结果：零覆盖率不等于成功，ROI平均也尚未成为身份特征</h1><h2>保守几何关联</h2><p>门限仅由训练集确定：平均极线距离严格小于2.36447px，使164条已知异鼠训练配对仅1条误接收。但162条已知同鼠配对也只保留2条。互选后要求至少3个相机、组内每对都相连、相机不能重复、歧义组全部拒绝。</p><p>结果：训练21场景、开发16场景和三种故障注入均产生0个三视角组。这不是安全融合成功，而是完全失去覆盖率。没有按开发结果放宽门限。7项分组单测通过，未修改姿态。</p><h2>query之前的图像特征</h2><p>从冻结VGGT-Omega图像骨干的patch输出（实例query之前）和投影后的图像memory，在每个新鲜预测框内5×5取样平均。不是人工框裁剪；也不是把裁剪图重新送入编码器。148张RGB重新独立编码，旧姿态/框结果一致，5项ROI单测通过。这里并非额外下载的DINO或训练好的re-ID模型。</p><table><tr><th>划分</th><th>特征</th><th>同鼠AUC</th><th>同query AUC</th><th>互选同鼠/已知</th><th>未知/全部互选</th></tr>']
    for role,rows in roi['results'].items():
        for name,r in rows.items():
            m=r['mutual'];page.append(f'<tr><td>{role}</td><td>{name}</td><td>{r["same_identity_auc"]:.3f}</td><td>{r["same_query_auc"]:.3f}</td><td>{m["correct"]}/{m["known"]}</td><td>{m["unknown"]}/{m["proposed"]}</td></tr>')
    page.append('</table><p>原始余弦仍接近随机，不能据此开启身份融合。可能因素包括整图编码的上下文、框内混入另一只鼠、视角变化及分辨率；本轮并未证明哪一个是根因。下面按预测框展示首个开发场景，部分侧视框确实含有两只鼠。图像仅用于检查，身份说明为事后评价。</p><img src="first_scene_predicted_crops.jpg"><p>下一项不同假设可检查：先按预测框裁剪RGB，再独立编码，另设灰度控制以识别毛色捷径；原始标签/姿态模型不变。不把黑棕鼠关联等同于同色或真实双鼠成功。</p>')
    summary=dict(partial=partial['summary'],calibration=partial['calibration'],roi=roi['results'],encoder_calls=runtime['images'],wall_seconds=runtime['elapsed_seconds'],max_replay_delta=max(x['delta'] for x in runtime['records']))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');page.append('<p><a href="summary.json">指标及计算量</a></p>')
    (OUT/'index.html').write_text('\n'.join(page),encoding='utf-8');DEST.mkdir(exist_ok=True)
    for name in ['index.html','summary.json','first_scene_predicted_crops.jpg']:shutil.copy2(OUT/name,DEST/name)
    print(json.dumps({k:v for k,v in summary.items() if k in ['encoder_calls','wall_seconds','max_replay_delta']}))


if __name__=='__main__':main()
