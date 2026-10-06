"""Read-only full TRAIN synthetic candidate coverage and fixed-rule audit."""
import collections,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from PIL import Image,ImageDraw
from mousegeoforge.epipolar_candidate_readout import rerank_candidates
from mousegeoforge.detection_readout_loss import match_detection
from mousegeoforge.pilot_metrics import box_iou
from training.finetune_blendedmvs import load_scene
from run_depth_pose_ablation import fingerprint,write_json
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';SOURCE=CYCLE/'synthetic_full_candidate_context_v1';OUT=CYCLE/'synthetic_full_candidate_audit_v1';DEST=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005');KP=[0,14,15,16,17]


def main():
    assert not OUT.exists();OUT.mkdir();torch.set_num_threads(2);manifest=json.loads((SOURCE/'manifest.json').read_text());metrics=collections.defaultdict(lambda:dict(errors=[],hits=[],active=0));points=[];counts=collections.Counter();samples=[]
    for si,record in enumerate(manifest['records']):
        assert fingerprint(ROOT/record['path'])==record['sha256'];scene=torch.load(ROOT/record['path'],map_location='cpu',weights_only=False);p=scene['predictions'];t=scene['targets'];pixel=scene['pixels'];views=scene['physical_views']
        assert scene['role']=='train' and p['candidate_memory'].shape==(len(views),6,5,6,256) and p['tokens'].shape==(len(views),6,5,256)
        torch.testing.assert_close(p['candidate_xy'][:,:,:,0],p['xy'][:,:,KP],atol=0,rtol=0);np.testing.assert_allclose(p['probability'].sum((-1,-2)).numpy(),1,atol=2e-6)
        assert all(torch.isfinite(v).all() for v in p.values());score=p['objectness_logits'].sigmoid().numpy();available=(score>=.5).sum(-1)==1;q=score.argmax(-1);counts['images']+=len(views);counts['unique_query_images']+=int(available.sum());counts['extra_images']+=sum(i is None for i in scene['source_rows'])
        positions=p['candidate_xy'].numpy()[np.arange(len(views)),q,:,1:]*pixel;masses=p['candidate_mass'].numpy()[np.arange(len(views)),q,:,1:]
        base=p['xy'].numpy();pred={a:base.copy() for a in ['baseline','unary','epipolar']};active=np.zeros((len(views),6,23),bool)
        for v in range(len(views)):
            for ki,kp in enumerate(KP):
                for arm,weight in [('unary',0),('epipolar',1)]:
                    r=rerank_candidates(scene['cameras']['projections'],positions[:,ki],masses[:,ki],available,views,v,base[v,q[v],kp]*pixel,enabled=True,geometry_weight=weight)
                    if r['active']:pred[arm][v,q[v],kp]=r['xy']/pixel;active[v,q[v],kp]=True
        det=dict(instance_boxes_cxcywh=p['boxes'],instance_objectness_logits=p['objectness_logits']);matches=match_detection(det,t['boxes_cxcywh'],t['box_valid']&t['instance_valid'])
        for v,(qs,ts) in enumerate(matches):
            for qq,tt in zip(qs,ts):
                qi,ti=int(qq),int(tt);hit=bool(score[v,qi]>=.5 and box_iou(p['boxes'][v,qi:qi+1],t['boxes_cxcywh'][v,ti:ti+1])[0,0]>=.5)
                for kp in KP:
                    if not t['keypoint_valid'][v,ti,kp]:continue
                    for arm in pred:
                        error=float(np.linalg.norm((pred[arm][v,qi,kp]-t['keypoint_xy'][v,ti,kp].numpy())*pixel));key=f'{arm}/all';a=metrics[key];a['errors'].append(error);a['hits'].append(hit and error<=32);a['active']+=int(active[v,qi,kp] and arm!='baseline')
                        subset='old_views' if scene['source_rows'][v] is not None else 'extra_views';a=metrics[f'{arm}/{subset}'];a['errors'].append(error);a['hits'].append(hit and error<=32);a['active']+=int(active[v,qi,kp] and arm!='baseline')
                        points.append(dict(scene=scene['scene'],view=views[v],point=kp+1,query=qi,arm=arm,error=error,subset=subset,active=bool(active[v,qi,kp])))
        if si in [0,4,12]:
            frames=load_scene(ROOT/scene['scene']).frames;w,h=map(int,pixel)
            for v in range(min(len(views),4)):
                image=Image.open(frames[views[v]].image_path).convert('RGB').resize((w,h));canvas=Image.new('RGB',(3*w,h+40),'white');draw=ImageDraw.Draw(canvas)
                for col,arm in enumerate(pred):
                    canvas.paste(image,(col*w,40));draw.text((col*w+8,10),arm,fill='black')
                    for qi in np.flatnonzero(score[v]>=.5):
                        color=['#ff5555','#00ccff','#eecc00','#ee00cc','#00ee99','#ff9900'][qi]
                        for kp in KP:
                            x,y=pred[arm][v,qi,kp]*pixel+[col*w,40];draw.ellipse((x-4,y-4,x+4,y+4),outline=color,width=2);draw.text((x+5,y),str(kp+1),fill=color)
                    for ti in range(2):
                        for kp in KP:
                            if t['keypoint_valid'][v,ti,kp]:
                                x,y=t['keypoint_xy'][v,ti,kp].numpy()*pixel+[col*w,40];draw.line((x-4,y-4,x+4,y+4),fill='white',width=2);draw.line((x-4,y+4,x+4,y-4),fill='white',width=2)
                name=f'scene{si:03d}_view{views[v]}.jpg';canvas.save(OUT/name,quality=90);samples.append(name)
    report=dict(counts=dict(counts),metrics={k:dict(points=len(a['errors']),mean_error=float(np.mean(a['errors'])),gated_pck32=float(np.mean(a['hits'])),active_points=a['active']) for k,a in metrics.items()},source_manifest_sha256=fingerprint(SOURCE/'manifest.json'),max_projection_px=manifest['max_synthetic_projection_px'],
        warning='TRAIN-only diagnostic, not generalization. Same original153 views now have additional context; comparisons across expanded versus old 2-view sets require matching subsets. New candidate0 remains originalmean. No label or production edits.')
    write_json(OUT/'results.json',report);write_json(OUT/'points.json',points)
    html=['<!doctype html><meta charset="utf-8"><title>合成TRAIN全视角检查</title><style>body{font:17px system-ui;margin:30px}img{max-width:100%}td{padding:8px;border:1px solid #aaa}</style><h1>既有合成TRAIN全视角输入检查</h1><p>69个原训练场景，共461图：复用153、新编码308。训练/验证划分及原始标注不变。所有评价只是TRAIN诊断，不是泛化测试。白叉标注、彩色预测；不按GT选择预测查询。</p><table>']
    for k,a in report['metrics'].items():html.append(f'<tr><td>{k}</td><td>{a["mean_error"]:.2f}px</td><td>PCK32 {a["gated_pck32"]:.3f}</td><td>active {a["active_points"]}/{a["points"]}</td></tr>')
    html.append('</table>')
    for name in samples:html.append(f'<h2>{name}</h2><img src="{name}">')
    (OUT/'index.html').write_text('\n'.join(html),encoding='utf-8');shutil.copytree(OUT,DEST/OUT.name);print(json.dumps(report))


if __name__=='__main__':main()
