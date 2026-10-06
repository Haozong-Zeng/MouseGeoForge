"""Frozen synthetic curve heads on real DEVELOPMENT only; no labels updated."""
import collections,datetime as dt,fcntl,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'vggt-omega'),str(ROOT/'tools')]
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image,ImageDraw
from vggt_omega.models.heads.dense_head import DenseHead
from vggt_omega.models.heads.instance_pose_head import MultiInstanceMouseHead
from mousegeoforge.models.surface_curve_pose import SurfaceCurveDepthPoseModel
from mousegeoforge.models.surface_curve_refinement import SurfaceCurveSpatialRefiner
from mousegeoforge.surface_curve_supervision import cubic_bspline,points_to_polyline
from mousegeoforge.pilot_metrics import box_iou
from training.instance_pose_losses import instance_targets_from_batch,hungarian_match_instances
from probe_diverse_curve_features_20261003 import FeatureProbe
from run_depth_pose_ablation import write_json,fingerprint,move_batch
CYCLE=ROOT/'output/surface_curve_cycle_20261003'
PILOT=CYCLE/'training_v1/pilot'
OUT=CYCLE/'real_curve_transfer_v1'
ORACLE_DIAGNOSTIC=False


def main():
    assert not OUT.exists() and (CYCLE/'curve_spatial_refinement_v1/results.json').exists()
    lock=(ROOT/'output/teacher_temporal_cycle_20261001/gpu_job.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert torch.cuda.is_available();OUT.mkdir();(OUT/'visuals').mkdir();torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    deadline=dt.datetime(2026,10,4,8,5,16,tzinfo=dt.timezone.utc)
    def status(stage,**kw):write_json(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),**kw))
    status('loading_frozen_heads')
    model=SurfaceCurveDepthPoseModel(nn.Identity(),DenseHead(),MultiInstanceMouseHead(num_keypoints=23)).cuda().eval()
    initial=PILOT/'initial_heads.pt';model.load_state_dict(torch.load(initial,map_location='cpu',weights_only=True),strict=True);model.requires_grad_(False)
    assert torch.count_nonzero(model.surface_curve_head.net[-1].weight)==0
    bases={};refiners={};hashes={'initial_heads':fingerprint(initial)}
    for seed in [17,29,43]:
        path=CYCLE/f'curve_data_scaling_v1/expanded_seed{seed}.pt';p=FeatureProbe(False).cuda().eval()
        p.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['state']);p.requires_grad_(False);bases[seed]=p;hashes[f'base{seed}']=fingerprint(path)
        for mode in ['global','local']:
            path=CYCLE/f'curve_spatial_refinement_v1/{mode}_seed{seed}.pt';p=SurfaceCurveSpatialRefiner(mode).cuda().eval()
            p.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['state']);p.requires_grad_(False);refiners[(seed,mode)]=p;hashes[f'{mode}{seed}']=fingerprint(path)
    records=[r for r in json.loads((PILOT/'records.json').read_text()) if r['kind']=='real']
    assert all(r['session'] not in ('GR2','UR2','UR3') for r in records)
    write_json(OUT/'protocol.json',dict(checkpoint_hashes=hashes,real_groups=36,
        split='Existing real train/development validation only; lockedGR2UR2UR3neverloaded. Appearance development used these sources, not external benchmark.',
        inference='Frozen original model, synthetic-only curve base/refinement. No real optimization. Oracle variants, when explicitly enabled, substitute GT mask-derived box center/size into curve coordinates ONLY; these are not deployable model scores.',
        oracle_box_diagnostic=ORACLE_DIAGNOSTIC,
        association='Original predicted query selected with target box/objectness Hungarian, keypoint_valid zeroed. Association-assisted diagnostic, not end-to-end detection score.',
        metrics='Observed06/07point-to-predicted-polyline distance only, not fullcurveGT. Mask membership and predicted boxIoU are localization sanity checks.',
        deduplication='One entry per source image, deterministic first paired-view context. No mean over duplicate appearances.',
        visibility='Unknown. Missing points not treated invisible. No full real curve, virtual endpoints or3DGT invented.',
        deadline=deadline.isoformat()))
    rows=[];arrays=collections.defaultdict(list);seen=set();visuals=[];shown=set()
    for n,r in enumerate(records):
        assert dt.datetime.now(dt.timezone.utc)<deadline
        item=torch.load(PILOT/'cache'/f"{r['key']}.pt",map_location='cpu',weights_only=False)
        assert all(fingerprint(ROOT/p)==sha for p,sha in item['source_hashes'].items())
        batch=move_batch(item['batch'],'cuda');memory=[];pose=[]
        a=model.instance_pose_head.instance_decoder.register_forward_pre_hook(lambda m,args:memory.append(args[1].detach()))
        b=model.instance_pose_head.keypoint_decoder.register_forward_hook(lambda m,args,out:pose.append(out.detach()))
        try:
            with torch.no_grad():pred=model.forward_cached(batch['images'],[None if x is None else x.cuda() for x in item['cached']],item['patch_start'],coupling='off')
        finally:a.remove();b.remove()
        target=instance_targets_from_batch(batch);matching=dict(target);matching['keypoint_valid']=torch.zeros_like(target['keypoint_valid'])
        matches=hungarian_match_instances(pred,matching)[0];h,w=batch['images'].shape[-2:];pixels=torch.tensor([w,h],device='cuda')
        maps=memory[0].transpose(1,2).reshape(2,256,h//16,w//16).half().float()
        maps=F.interpolate(maps,(48,48),mode='bilinear',align_corners=False);pt=pose[0].reshape(2,6,23,256)
        for v,(qs,ms) in enumerate(matches):
            identity=(r['session'],r['frame'],r['views'][v])
            if identity in seen:continue
            assert len(qs)==1 and ms.tolist()==[0];seen.add(identity);q=int(qs[0]);m=0
            token=pt[v,q,[4,5,6,7]].flatten()[None];box=pred['instance_boxes_cxcywh'][0,v,q:q+1]
            # Original curve net has zero final weights and fixed prior bias.
            logits=model.surface_curve_head.net.bias if hasattr(model.surface_curve_head.net,'bias') else model.surface_curve_head.net[-1].bias
            logits=logits.reshape(1,6,2)
            curves={}
            bv=bool(target['box_valid'][0,v,m])
            with torch.no_grad():
                for seed,base in bases.items():
                    controls=box[:,:2,None].transpose(-1,-2)+(logits.sigmoid()-.5)*box[:,2:,None].transpose(-1,-2)+.5*base.delta(token).reshape(1,6,2).tanh()
                    curves[f'base_seed{seed}']=cubic_bspline(controls,33)[0]
                    for mode in ['global','local']:curves[f'{mode}_seed{seed}']=refiners[(seed,mode)](token,maps[v:v+1],controls)[0]
                    if ORACLE_DIAGNOSTIC:
                        gt_box=target['boxes_cxcywh'][0,v,m:m+1] if bv else box
                        center_delta=(gt_box[:,:2]-box[:,:2])[:,None]
                        size_delta=(gt_box[:,2:]-box[:,2:])[:,None]
                        curves[f'oracle_center_seed{seed}']=cubic_bspline(controls+center_delta,33)[0]
                        curves[f'oracle_box_seed{seed}']=cubic_bspline(controls+center_delta+(logits.sigmoid()-.5)*size_delta,33)[0]
            valid=target['keypoint_valid'][0,v,m,5:7];observed=target['keypoint_xy'][0,v,m,5:7]*pixels
            bv=bool(target['box_valid'][0,v,m]);iou=float(box_iou(box,target['boxes_cxcywh'][0,v,m:m+1])[0,0]) if bv else None
            mv=bool(target['mask_valid'][0,v,m]);mask=target['masks'][0,v,m]
            metrics={}
            for name,curve in curves.items():
                p=curve*pixels;errors=points_to_polyline(observed[valid],p,torch.ones(33,dtype=torch.bool,device='cuda'))[0].sqrt().tolist() if valid.any() else []
                xy=p.round().long();inside=(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
                membership=mask[xy[:,1].clamp(0,h-1),xy[:,0].clamp(0,w-1)]&inside
                length=(p[1:]-p[:-1]).norm(dim=-1).sum();chord=(p[-1]-p[0]).norm()
                metrics[name]=dict(observed_point_to_curve_px=errors,mask_membership_fraction=float(membership.float().mean()) if mv else None,
                    in_image_fraction=float(inside.float().mean()),length_to_chord=float(length/chord.clamp_min(1e-6)))
                arrays[name].append(curve.cpu().numpy())
            rows.append(dict(session=r['session'],frame=r['frame'],view=r['views'][v],role=r['role'],query=q,cache_key=r['key'],box_iou=iou,mask_valid=mv,oracle_box_valid=bv,
                             predicted_box=box[0].cpu().tolist(),target_box=target['boxes_cxcywh'][0,v,m].cpu().tolist() if bv else None,
                             observed_back=observed.cpu().numpy().tolist(),observed_valid=valid.cpu().tolist(),metrics=metrics))
            selection=(r['session'],r['views'][v]);is_r12=r['session']=='FR2' and r['frame']==584 and r['views'][v]==1
            if (r['views'][v] in [1,2,5] and selection not in shown) or is_r12:
                shown.add(selection);names=['base_seed17','global_seed17','local_seed17'];rgb=batch['images'][0,v].permute(1,2,0).cpu().numpy()
                if ORACLE_DIAGNOSTIC:names=['base_seed17','oracle_center_seed17','oracle_box_seed17']
                im=Image.fromarray(np.uint8(rgb.clip(0,1)*255)).resize((352,384));canvas=Image.new('RGB',(352*3,412),(22,22,28));draw=ImageDraw.Draw(canvas)
                for col,name in enumerate(names):
                    canvas.paste(im,(352*col,28));draw.text((col*352+4,5),name,fill='white')
                    p=curves[name].cpu().numpy()*[352,384]+[352*col,28];draw.line([tuple(x) for x in p],fill='#39dfff',width=2)
                    for k in range(2):
                        if bool(valid[k]):
                            x,y=(observed[k].cpu().numpy()/[w,h]*[352,384]+[352*col,28]);draw.ellipse((x-3,y-3,x+3,y+3),fill='#ffdd33');draw.text((x+4,y-8),str(k+6),fill='#ffdd33')
                filename=f'{r["session"]}_{r["frame"]:06d}_cam{chr(65+r["views"][v])}.jpg';canvas.save(OUT/'visuals'/filename,quality=94);visuals.append(dict(file='visuals/'+filename,**{k:rows[-1][k] for k in ['session','frame','view','role','box_iou']}))
        status('evaluating',completed=n+1,total=len(records));write_json(OUT/'partial_rows.json',rows)
    summary=[]
    for role in ['train','validation']:
        for arm in arrays:
            selected=[r for r in rows if r['role']==role and (not arm.startswith('oracle_') or r['oracle_box_valid'])];errors=[e for r in selected for e in r['metrics'][arm]['observed_point_to_curve_px']]
            masks=[r['metrics'][arm]['mask_membership_fraction'] for r in selected if r['mask_valid']]
            summary.append(dict(role=role,arm=arm,n_images=len(selected),n_back_points=len(errors),point_to_curve_mean_px=float(np.mean(errors)),point_to_curve_median_px=float(np.median(errors)),mask_membership_mean=float(np.mean(masks))))
    np.savez_compressed(OUT/'predictions.npz',**{k:np.stack(v) for k,v in arrays.items()})
    write_json(OUT/'results.json',dict(summary=summary,rows=rows,visuals=visuals,inspection_pending=True,no_pseudolabels_created=True))
    html=['<!doctype html><meta charset="utf-8"><title>Real development curve transfer</title><style>body{background:#202027;color:white;font:16px sans-serif}img{max-width:100%}</style><h1>Frozen synthetic curve heads on real development images</h1><p>Yellow=manual06/07observations only, NOT fullcurveGT. Cyan=predicted curve. No real training. Paired view contexts; seed17 visuals, all3seed metrics. Masks/boxes only for diagnostic association, never input to curve forward.</p>']
    if ORACLE_DIAGNOSTIC:html=['<!doctype html><meta charset="utf-8"><title>GT-box assisted diagnostic ONLY</title><style>body{background:#202027;color:white;font:16px sans-serif}img{max-width:100%}</style><h1>Oracle box substitution: NOT deployable model performance</h1><p>Second/third columns use GT mask-derived box center/full box; pose tokens and learned curve residuals unchanged. Yellow=manual06/07, not fullcurveGT. No real training or label updates. This diagnoses box-coordinate contribution, not all localization information encoded in pose tokens.</p>']
    for r in visuals:html.append(f'<h2>{r["session"]} frame{r["frame"]} view{r["view"]}, {r["role"]}, boxIoU={r["box_iou"]}</h2><img src="{r["file"]}">')
    (OUT/'index.html').write_text('\n'.join(html));status('complete');print(json.dumps(summary,indent=2))


if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(exist_ok=True);write_json(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()));raise
