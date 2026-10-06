"""Box-matched visual review of completed balanced arms, without loading the encoder."""
import argparse,collections,datetime as dt,fcntl,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'vggt-omega'),str(ROOT/'tools')]
import numpy as np
from PIL import Image,ImageDraw
import torch
from torch import nn
from vggt_omega.models.heads.dense_head import DenseHead
from vggt_omega.models.heads.instance_pose_head import MultiInstanceMouseHead
from mousegeoforge.models.surface_curve_pose import SurfaceCurveDepthPoseModel
from training.instance_pose_losses import instance_targets_from_batch,hungarian_match_instances
from run_depth_pose_ablation import move_batch,write_json,fingerprint
BASE=ROOT/'output/surface_curve_cycle_20261003/training_v1/pilot'
ARMS=['strict_points','weak_points','weak_points_curve6']

def common_metrics():
    steps=sorted(set.intersection(*[{int(p.stem.split('_')[-1]) for p in (BASE/a).glob('eval_*.json')} for a in ARMS]))
    result={}
    for arm in ARMS:
        result[arm]={str(s):json.loads((BASE/arm/f'eval_{s:06d}.json').read_text())['metrics'] for s in steps}
    write_json(BASE/'balanced_metrics.json',dict(common_steps=steps,arms=result,interpretation='Development diagnostics, not independent external test; new curve metrics are training-set only'))
    return steps,result

def pick(rows):
    selected=[]
    requests=[('real','FR2',584,[0,1]),('real','FR2',0,[4,5]),('real','AZR2L2',1559,[0,2]),('real','AZL3',0,[0,1])]
    for kind,session,frame,views in requests:
        r=next(x for x in rows if x['kind']==kind and x.get('session')==session and x.get('frame')==frame and x['views']==views);selected.append(r)
    for frame in ['000001','000002','000003']:
        selected.extend(x for x in rows if x['kind']=='curve' and x['scene'].endswith(frame) and x['views'] in ([0,1],[4,5]))
    selected.append(next(x for x in rows if x['kind']=='synthetic' and x['role']=='validation' and 'BlackBrownFur' in x['scene']))
    return selected

def panel(item,pred,title):
    b=item['batch'];t=instance_targets_from_batch(b);h,w=b['images'].shape[-2:];ratio=352/w;ph=round(h*ratio)
    canvas=Image.new('RGB',(352,2*(ph+32)),(24,24,28));draw=ImageDraw.Draw(canvas)
    if pred is not None:
        mt=dict(t);mt['keypoint_valid']=torch.zeros_like(t['keypoint_valid']);matches=hungarian_match_instances(pred,mt)[0]
    for v in range(2):
        off=v*(ph+32);rgb=np.uint8(b['images'][0,v].permute(1,2,0).numpy().clip(0,1)*255)
        canvas.paste(Image.fromarray(rgb).resize((352,ph)),(0,off+32));draw.text((5,off+5),f"{title} cam{chr(65+item['record']['views'][v])}",fill='white')
        for m in torch.where(t['instance_valid'][0,v])[0].tolist():
            good=t['keypoint_valid'][0,v,m];gt=t['keypoint_xy'][0,v,m].numpy()*[352,ph]+[0,off+32]
            for k in torch.where(good)[0].tolist():
                x,y=gt[k];draw.ellipse((x-2,y-2,x+2,y+2),fill='#4aff70')
                if k in [0,5,6,14,15,16,17]:draw.text((x+2,y),str(k+1),fill='#4aff70')
            if t['box_valid'][0,v,m]:
                box=t['boxes_cxcywh'][0,v,m].numpy();lo=(box[:2]-box[2:]/2)*[352,ph]+[0,off+32];hi=(box[:2]+box[2:]/2)*[352,ph]+[0,off+32]
                draw.rectangle((*lo,*hi),outline='#4aff70',width=1)
        if pred is None and 'curve_target' in item['aux']:
            points=item['aux']['curve_target'][0,v,0].numpy()*ratio+[0,off+32]
            draw.line([tuple(p) for p in points],fill='#ffd740',width=2)
            draw.text((5,off+ph+16),'GT surface line (may be occluded)',fill='#ffd740')
        if pred is not None:
            for q,m in zip(*[x.tolist() for x in matches[v]]):
                good=t['keypoint_valid'][0,v,m];xy=pred['instance_keypoint_xy'][0,v,q].numpy()*[352,ph]+[0,off+32]
                for k in torch.where(good)[0].tolist():
                    x,y=xy[k];draw.ellipse((x-2,y-2,x+2,y+2),fill='#ff5062')
                    if k in [0,5,6,14,15,16,17]:draw.text((x+2,y),str(k+1),fill='#ff5062')
                box=pred['instance_boxes_cxcywh'][0,v,q].numpy();lo=(box[:2]-box[2:]/2)*[352,ph]+[0,off+32];hi=(box[:2]+box[2:]/2)*[352,ph]+[0,off+32]
                draw.rectangle((*lo,*hi),outline='#ff5062',width=2)
                if 'instance_surface_curve_xy' in pred:
                    line=pred['instance_surface_curve_xy'][0,v,q].numpy()*[352,ph]+[0,off+32];draw.line([tuple(p) for p in line],fill='#42deff',width=2)
    return canvas

def main():
    p=argparse.ArgumentParser();p.add_argument('--metrics-only',action='store_true');p.add_argument('--gt-only',action='store_true');args=p.parse_args()
    steps,metrics=common_metrics();print('balanced steps',steps,flush=True)
    if args.metrics_only:
        keys=['real_validation/reliable_px','real_validation/back_normal_px','synthetic_validation/reliable_px','curve_train/curve_to_gt_px']
        for arm in ARMS:
            print(arm,{s:{k:metrics[arm][s].get(k,{}).get('mean') for k in keys} for s in metrics[arm]},flush=True)
        return
    if args.gt_only:
        folder=BASE/'gt_review';folder.mkdir(exist_ok=True)
        for r in pick(json.loads((BASE/'records.json').read_text())):
            item=torch.load(BASE/'cache'/f"{r['key']}.pt",map_location='cpu',weights_only=False)
            panel(item,None,'GT').save(folder/f"{r['key']}.png")
        return
    assert (BASE/'complete.json').exists(),'Wait for running balanced pilot; do not race changing checkpoints'
    lock=(ROOT/'output/teacher_temporal_cycle_20261001/gpu_job.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out=BASE/'visual_review';assert not out.exists();out.mkdir()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rows=pick(json.loads((BASE/'records.json').read_text()));items={r['key']:torch.load(BASE/'cache'/f"{r['key']}.pt",map_location='cpu',weights_only=False) for r in rows}
    # Cached encoder features only. Same trained decoder, no encoder allocation or fake images.
    model=SurfaceCurveDepthPoseModel(nn.Identity(),DenseHead(),MultiInstanceMouseHead(num_keypoints=23)).cuda().eval()
    outputs=collections.defaultdict(dict);provenance=[];ceilings=[]
    for arm in ARMS:
        path=BASE/arm/'latest.pt';cp=torch.load(path,map_location='cpu',weights_only=False);assert cp['step']==steps[-1]
        model.load_state_dict(cp['model'],strict=True);model.surface_curve_enabled=arm=='weak_points_curve6';del cp
        provenance.append(dict(arm=arm,checkpoint_sha256=fingerprint(path),step=steps[-1]))
        for key,item in items.items():
            with torch.no_grad():
                pred=model.forward_cached(item['batch']['images'].cuda(),[None if f is None else f.cuda() for f in item['cached']],item['patch_start'],coupling='off')
            keep={k:v.cpu() for k,v in pred.items() if k.startswith('instance_') and 'mask_logits' not in k and 'heatmap' not in k}
            outputs[key][arm]=keep
            np.savez_compressed(out/f'{key}_{arm}.npz',**{k:v.numpy() for k,v in keep.items()})
            if arm=='weak_points_curve6' and 'curve_target' in item['aux']:
                target=instance_targets_from_batch(item['batch']);mt=dict(target);mt['keypoint_valid']=torch.zeros_like(target['keypoint_valid'])
                match=hungarian_match_instances(keep,mt)[0];h,w=item['batch']['images'].shape[-2:]
                for v,(qs,ms) in enumerate(match):
                    for q,m in zip(qs.tolist(),ms.tolist()):
                        gt=item['aux']['curve_target'][0,v,m];box=keep['instance_boxes_cxcywh'][0,v,q]*torch.tensor([w,h,w,h]);lo=box[:2]-box[2:]/2;hi=box[:2]+box[2:]/2
                        delta=(lo-gt).clamp_min(0)+(gt-hi).clamp_min(0);dist=delta.norm(dim=-1)
                        ceilings.append(dict(key=key,view=item['record']['views'][v],curve_gt_outside_predicted_box_fraction=float((dist>0).float().mean()),gt_to_box_minimum_possible_distance_px=float(dist.mean()),interpretation='Geometric lower bound for GT-to-prediction distance if all predicted control points are constrained inside the predicted box; not a GT-assisted model result'))
    html=['<!doctype html><meta charset="utf-8"><title>Surface curve pilot review</title><style>body{background:#202027;color:#eee;font:16px sans-serif}img{max-width:100%}</style>',f'<h1>Three-arm review: {steps[-1]} equal steps</h1><p>Green: original GT points/boxes. Red: box-matched predictions. Cyan: predicted dorsal surface curve. Yellow: synthetic curve GT, including occluded portions. No full real curve GT exists. Each mouse instance matched by boxes/objectness, not keypoints. No external-generalization claim.</p>']
    for r in rows:
        key=r['key'];item=items[key];panels=[panel(item,None,'GT')]+[panel(item,outputs[key][a],a.replace('weak_points','weak')) for a in ARMS]
        canvas=Image.new('RGB',(sum(p.width for p in panels),max(p.height for p in panels)),(24,24,28));x=0
        for im in panels:canvas.paste(im,(x,0));x+=im.width
        canvas.save(out/f'{key}.jpg',quality=95)
        html.append(f'<h2>{key}: {r["scene"]}, views {r["views"]}</h2><img src="{key}.jpg">');print('VISUAL',key,flush=True)
    (out/'index.html').write_text('\n'.join(html),encoding='utf-8')
    write_json(out/'manifest.json',dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),rows=rows,checkpoints=provenance,inspection_pending=True,curve_box_limits=ceilings))
if __name__=='__main__':main()
