"""Read-only small-cohort observability audit; no inference/training or label edits."""
import sys,json,hashlib,shutil,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import torch,numpy as np
from PIL import Image,ImageDraw
from train_pixel_adapter_matched_20261005 import prepare_rows,KP
from run_surface_curve_pilot_20261003 import load_item
from training.instance_pose_losses import instance_targets_from_batch
from mousegeoforge.detection_readout_loss import match_detection
from mousegeoforge.pilot_metrics import box_iou
from review_pixel_adapter_20261005 import font
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';INPUT=CYCLE/'highres_preflight_inputs_v1';OUT=CYCLE/'highres_visibility_audit_v1';DEST=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005')/OUT.name
def read(p):return torch.load(p,weights_only=False,map_location='cpu')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def main():
    OUT.mkdir(exist_ok=False);DEST.mkdir(exist_ok=True);torch.set_num_threads(2)
    rows=prepare_rows();ev={r['index']:r for r in read(INPUT/'evaluation_only.pt')};manifest={r['index']:r for r in json.loads((INPUT/'inference_manifest.json').read_text())};oldhash=json.loads((INPUT/'source_hashes.json').read_text());hashes={};items={};aligned={};details=[]
    runs={'finetuned':('highres_preflight_pilot_v1','appearance'),'zero_appearance':('highres_preflight_pilot_v1','zero_appearance'),'pre_relu':('highres_pre_relu_pilot_v1','appearance'),'adapt_unit':('highres_unit_control_v1','adapt_unit'),'official':('official_quadruped_readout_pilot_v1','appearance')}
    pred={'baseline':read(CYCLE/'highres_preflight_pilot_v1/predictions.pt')['baseline']}
    for arm,(run,key) in runs.items():pred[arm]=read(CYCLE/run/'predictions.pt')[key]
    for i,e in ev.items():
        r=rows[i];assert r['metadata']==e['metadata'] and r['metadata']['stage']!='validation'
        assert torch.equal(r['gt'],e['gt']) and torch.equal(r['valid'],e['valid'])
        cachekey=(r['record']['scene'],tuple(r['record']['views']))
        if cachekey not in items:items[cachekey]=load_item(r['record'])
        item=items[cachekey];batch=item['batch'];t=instance_targets_from_batch(batch);v=r['record']['views'].index(r['metadata']['view']);b=r['baseline']
        for path,h in item['source_hashes'].items():
            assert oldhash[path]==h,('changed source',path);hashes[path]=h
        occ=ROOT/r['record']['scene']/'keypoint_occlusion.npz'
        if occ.exists():hashes[str(occ.relative_to(ROOT))]=sha(occ)
        qs,ts=match_detection(dict(instance_boxes_cxcywh=b['boxes'][None],instance_objectness_logits=b['objectness_logits'][None]),t['boxes_cxcywh'][:,v],(t['box_valid']&t['instance_valid'])[:,v])[0]
        gt=torch.zeros(6,5,2);valid=torch.zeros(6,5,dtype=torch.bool);vv=valid.clone();vis=valid.clone();cause=torch.full((6,5),-1,dtype=torch.long);pairs=[]
        for q,m in zip(qs.tolist(),ts.tolist()):
            iou=float(box_iou(b['boxes'][q:q+1],t['boxes_cxcywh'][0,v,m:m+1])[0,0])
            if b['objectness_logits'][q].sigmoid()<.5 or iou<.5:continue
            gt[q]=t['keypoint_xy'][0,v,m,KP];valid[q]=t['keypoint_valid'][0,v,m,KP];vv[q]=t['keypoint_visibility_valid'][0,v,m,KP];vis[q]=t['keypoint_visible'][0,v,m,KP];cause[q]=t['keypoint_occlusion_cause'][0,v,m,KP];pairs.append(dict(query=q,instance=m,iou=iou))
        assert torch.equal(valid,e['valid']),('valid mismatch',i)
        assert torch.allclose(gt[valid],e['gt'][valid],atol=1e-6,rtol=0),('GT mismatch',i)
        masks={'visible':valid&vv&vis,'occluded':valid&vv&~vis,'visibility_unknown':valid&~vv}
        assert torch.equal(sum(m.int() for m in masks.values()),valid.int())
        if e['metadata']['kind']=='real':assert not vv.any()
        aligned[i]=dict(index=i,visibility_valid=vv,visible=vis,occlusion_cause=cause,masks=masks,pairs=pairs)
        errors={a:((p[i][:,KP]-e['gt'])*e['pixels']).norm(dim=-1) for a,p in pred.items()}
        details.append(dict(index=i,metadata=e['metadata'],counts={g:int(m.sum()) for g,m in masks.items()},pairs=pairs,errors={a:{g:float(x[m].mean()) if m.any() else None for g,m in masks.items()} for a,x in errors.items()}))
        print('aligned',i,details[-1]['counts'],flush=True)
    summary=[]
    for stage in ['fit','calibration']:
        for kind in ['real','single','dual']:
            ids=[i for i,e in ev.items() if e['metadata']['stage']==stage and e['metadata']['kind']==kind]
            for group in ['visible','occluded','visibility_unknown']:
                row=dict(stage=stage,kind=kind,visibility=group,metrics={},causes={})
                causes=[]
                for i in ids:causes.extend(aligned[i]['occlusion_cause'][aligned[i]['masks'][group]].tolist())
                row['causes']=dict(collections.Counter(causes))
                for arm,p in pred.items():
                    errors=[]
                    for i in ids:
                        e=ev[i];x=((p[i][:,KP]-e['gt'])*e['pixels']).norm(dim=-1);errors.extend(x[aligned[i]['masks'][group]].tolist())
                    a=np.asarray(errors);row['metrics'][arm]=dict(n=len(a),mean=float(a.mean()) if len(a) else None,p95=float(np.quantile(a,.95)) if len(a) else None,pck32=float((a<=32).mean()) if len(a) else None)
                summary.append(row)
    write(OUT/'summary.json',summary);write(OUT/'details.json',details);write(OUT/'source_hashes.json',hashes);torch.save(aligned,OUT/'evaluation_visibility.pt')
    write(OUT/'protocol.json',dict(images=42,fit=24,internal_cal=18,development=0,qualification='Original box-only Hungarian association, score>=.5 and IoU>=.5 after forward; GT/valid parity checked against cached targets.',visibility='Explicit loader visibility_valid AND visible. Coordinate-valid but visibility unavailable remains unknown, not occluded. Native cause values recorded without reinterpretation.',limits='Read-only posthoc diagnostic. Current annotations/source hashes verified; occlusion sidecar hash newly recorded because original export did not fingerprint it. Small enriched conditional subset; no AP or external generalization claim. Mixed runtime for adapt_unit is descriptive, not isolated feature-control comparison.'))
    selections=[]
    for stage in ['fit','calibration']:
        for kind in ['single','dual']:
            entries=[z for z in details if z['metadata']['stage']==stage and z['metadata']['kind']==kind and z['counts']['visible']]
            if not entries:continue
            z=max(entries,key=lambda x:x['errors']['official']['visible']-x['errors']['baseline']['visible']);i=z['index'];e=ev[i];im=Image.open(ROOT/manifest[i]['image']).convert('RGB');w,h=im.size;tile=Image.new('RGB',(2960,960),'white');d=ImageDraw.Draw(tile);colors=['#ff7744','#00bbff','#ff77dd','#aaee44','#ffaa00','#7777ff']
            for col,arm in enumerate(['baseline','finetuned','official','GT_visibility']):
                panel=im.copy();di=ImageDraw.Draw(panel)
                for q in range(6):
                    for k,kp in enumerate(KP):
                        if arm=='GT_visibility':
                            if not e['valid'][q,k]:continue
                            x,y=(e['gt'][q,k]*e['pixels']).tolist();color='#00ff66' if aligned[i]['masks']['visible'][q,k] else '#ff44dd' if aligned[i]['masks']['occluded'][q,k] else '#ffff00';di.line((x-5,y,x+5,y),fill=color,width=3);di.line((x,y-5,x,y+5),fill=color,width=3)
                        else:
                            x,y=(pred[arm][i][q,kp]*e['pixels']).tolist();color=colors[q];di.ellipse((x-3,y-3,x+3,y+3),outline=color,width=2)
                        if e['valid'][q,k]:di.text((x+5,y),str(kp+1),font=font(13),fill=color)
                panel.thumbnail((720,795));tile.paste(panel,(col*740+(740-panel.width)//2,100));d.text((col*740+10,10),arm,font=font(24),fill='black');d.text((col*740+10,50),f'{stage} {kind} visible worst change',font=font(18),fill='black')
            d.text((15,915),'GT green=explicit visible, magenta=occluded, yellow=unknown. All6 queries forwarded.',font=font(22),fill='black');name=f'{stage}_{kind}_visible_worst_{i}.jpg';tile.save(OUT/name,quality=95);selections.append(dict(z,file=name))
    write(OUT/'visual_examples.json',selections)
    page=['<!doctype html><meta charset="utf-8"><title>可见性分层审计</title><style>body{font:17px system-ui;margin:30px}img{max-width:100%}td,th{padding:7px;border:1px solid #aaa}table{border-collapse:collapse}</style><h1>坐标有效 ≠ 该视角可见</h1><p>仅复查同42张TRAIN/内部校准图。重新执行原预测框的评价匹配并验证坐标/有效性一致，读取明确的可见性字段；未知保持未知。无训练、无标签改写、无新开发验证。</p><table><tr><th>阶段</th><th>组</th><th>可见性</th><th>N</th>']+['<th>'+a+'</th>' for a in pred]+['</tr>']
    for z in summary:
        page.append('<tr>'+''.join('<td>'+str(z[k])+'</td>' for k in ['stage','kind','visibility'])+f'<td>{z["metrics"]["baseline"]["n"]}</td>'+''.join('<td>'+('—' if z['metrics'][a]['mean'] is None else f'{z["metrics"][a]["mean"]:.2f}')+'</td>' for a in pred)+'</tr>')
    page.append('</table><p>表中为输入像素mean↓。真实组没有可信可见性标记，因此全部记为unknown，不推断为不可见。各方法来自已有固定端点；adapt_unit运行环境不同，仅作描述性分层。</p><h2>明确可见点退步最大的合成案例</h2><p>极值不是发生率；与遮挡点不同，可见点可用于审计局部外观定位失败。</p>')
    for z in selections:page.append('<h3>'+str(z['index'])+'</h3><img src="'+z['file']+'">')
    page.append('<a href="summary.json">完整统计</a> | <a href="protocol.json">协议与限制</a>');(OUT/'index.html').write_text('\n'.join(page),encoding='utf-8')
    for p in OUT.iterdir():
        if p.is_file() and p.suffix!='.pt':shutil.copy2(p,DEST/p.name)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
