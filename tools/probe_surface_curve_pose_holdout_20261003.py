"""Fixed-budget leave-one-pose-out DEVELOPMENT diagnostic, not external validation."""
import collections,datetime as dt,fcntl,json,random,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'vggt-omega'),str(ROOT/'tools')]
import numpy as np
import torch
from torch import nn
from PIL import Image,ImageDraw
from vggt_omega.models.heads.dense_head import DenseHead
from vggt_omega.models.heads.instance_pose_head import MultiInstanceMouseHead
from mousegeoforge.models.surface_curve_pose import SurfaceCurveDepthPoseModel,SurfaceCurveHead
from mousegeoforge.surface_curve_supervision import curve_loss
from training.instance_pose_losses import instance_targets_from_batch,hungarian_match_instances
from run_depth_pose_ablation import write_json,fingerprint,move_batch
from probe_surface_curve_capacity_20261003 import ResidualProbe,measure
BASE=ROOT/'output/surface_curve_cycle_20261003/training_v1/pilot'
OUT=ROOT/'output/surface_curve_cycle_20261003/curve_pose_holdout_v1'

def main():
    assert not OUT.exists()
    lock=(ROOT/'output/teacher_temporal_cycle_20261001/gpu_job.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    OUT.mkdir();torch.set_num_threads(2);torch.manual_seed(17);torch.backends.cuda.matmul.allow_tf32=False
    deadline=dt.datetime(2026,10,4,8,5,16,tzinfo=dt.timezone.utc)
    def status(stage,**kw):write_json(OUT/'status.json',dict(stage=stage,pid=__import__('os').getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),**kw))
    status('extracting_original_initial_tokens')
    path=BASE/'initial_heads.pt';cp=torch.load(path,map_location='cpu',weights_only=True)
    model=SurfaceCurveDepthPoseModel(nn.Identity(),DenseHead(),MultiInstanceMouseHead(num_keypoints=23)).cuda().eval();model.load_state_dict(cp,strict=True);model.requires_grad_(False)
    base_state={k:v.cpu().clone() for k,v in model.surface_curve_head.state_dict().items()};del cp
    rows=[r for r in json.loads((BASE/'records.json').read_text()) if r['kind']=='curve'];features=[];boxes=[];truth=[];meta=[];images=[]
    for r in rows:
        item=torch.load(BASE/'cache'/f"{r['key']}.pt",map_location='cpu',weights_only=False);capture=[]
        hook=model.instance_pose_head.instance_decoder.register_forward_hook(lambda m,a,o:capture.append(o.detach()))
        try:
            with torch.no_grad():pred=model.forward_cached(item['batch']['images'].cuda(),[None if f is None else f.cuda() for f in item['cached']],item['patch_start'],coupling='off')
        finally:hook.remove()
        target=instance_targets_from_batch(move_batch(item['batch'],'cuda'));mt=dict(target);mt['keypoint_valid']=torch.zeros_like(mt['keypoint_valid']);match=hungarian_match_instances(pred,mt)[0]
        for v,(qs,ms) in enumerate(match):
            for q,m in zip(qs.tolist(),ms.tolist()):
                features.append(capture[0][v,q]);boxes.append(pred['instance_boxes_cxcywh'][0,v,q]);truth.append(item['aux']['curve_target'][0,v,m].cuda()/torch.tensor([704,768],device='cuda'))
                meta.append(dict(key=r['key'],pose=int(r['scene'].split('_')[-1]),view=r['views'][v],query=q));images.append(item['batch']['images'][0,v].clone())
    tokens=torch.stack(features);boxes=torch.stack(boxes);gt=torch.stack(truth);del model,features,truth;torch.cuda.empty_cache()
    torch.save(dict(tokens=tokens.cpu(),boxes=boxes.cpu(),gt_normalized=gt.cpu(),metadata=meta),OUT/'frozen_inputs.pt')
    write_json(OUT/'protocol.json',dict(source_checkpoint_sha256=fingerprint(path),source='original_common_initial_heads_NOT600step',steps=1200,batch_size=4,lr=.001,seed=17,
        folds=[1,2,3],modes=['box_bounded','free_residual'],main_model_frozen=True,
        split='ALLpaired-context observations of held pose excluded from loss and minibatch sampling; fixed1200updates, no best-checkpoint selection',
        limitations=['All3poses have already been inspected for architecture/hyperparameter design. These are development folds, NOT untouched tests.','Same animal asset, same cameras,3poses only','Frozen original pose backbone may have seen related source motions, not a new-animal benchmark','No real complete curve labels; cannot claim real curve accuracy','Different mappings give different effective coordinate step sizes'],
        inputs=meta))
    pixel_scale=gt.new_tensor([704.,768.]);ratio=pixel_scale/768.;results=[];visuals=[]
    for fold in [1,2,3]:
        train=[i for i,m in enumerate(meta) if m['pose']!=fold];held=[i for i,m in enumerate(meta) if m['pose']==fold]
        assert len(train)==16 and len(held)==8 and not set(train)&set(held)
        rng=random.Random(17);schedule=[rng.sample(train,4) for _ in range(1200)]
        assert all(meta[i]['pose']!=fold for ids in schedule for i in ids)
        predictions={}
        for mode in ['box_bounded','free_residual']:
            torch.manual_seed(17);base=SurfaceCurveHead(256).cuda();base.load_state_dict(base_state);probe=ResidualProbe(base,mode).cuda()
            with torch.no_grad():torch.testing.assert_close(probe(tokens,boxes),base(tokens,boxes)[1],rtol=0,atol=0)
            optimizer=torch.optim.AdamW(probe.delta.parameters(),lr=.001,weight_decay=.001);history=[]
            def metrics(ids):
                class Scaled(nn.Module):
                    def forward(self,t,b):return probe(t,b)*ratio
                return measure(Scaled(),tokens[ids],boxes[ids],gt[ids]*ratio)
            initial=dict(train=metrics(train),held=metrics(held));status('training',fold=fold,mode=mode,step=0)
            for step,ids in enumerate(schedule,1):
                assert dt.datetime.now(dt.timezone.utc)<deadline
                optimizer.zero_grad(set_to_none=True);pred=probe(tokens[ids],boxes[ids])*pixel_scale
                loss=torch.stack([curve_loss(p,g*pixel_scale,torch.ones(len(g),device='cuda',dtype=torch.bool),full_curve=True,scale=768) for p,g in zip(pred,gt[ids])]).mean()
                assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(probe.delta.parameters(),1.);assert torch.isfinite(norm);optimizer.step()
                if step%200==0:
                    # Training diagnostics only during optimization, no held-fold selection.
                    row=dict(step=step,loss=float(loss.detach()),train=metrics(train));history.append(row);write_json(OUT/f'fold{fold}_{mode}_history.json',history);status('training',fold=fold,mode=mode,step=step);print(fold,mode,step,row['train']['pred_to_gt_px']['mean'],flush=True)
            final=dict(fold=fold,mode=mode,step=1200,initial=initial,train=metrics(train),held=metrics(held),train_indices=train,held_indices=held)
            results.append(final);write_json(OUT/'partial_results.json',results)
            torch.save(dict(state=probe.state_dict(),fold=fold,mode=mode,step=1200),OUT/f'fold{fold}_{mode}.pt')
            predictions[mode]=probe(tokens,boxes).detach().cpu().numpy()
        np.savez_compressed(OUT/f'fold{fold}_predictions.npz',gt=gt.cpu().numpy(),**predictions)
        for i in held:
            rgb=Image.fromarray(np.uint8(images[i].permute(1,2,0).numpy()*255)).resize((352,384));canvas=Image.new('RGB',(1056,412),(25,25,30));draw=ImageDraw.Draw(canvas)
            for col,label in enumerate(['Held pose GT','box_bounded','free_residual']):
                canvas.paste(rgb,(col*352,28));draw.text((col*352+6,6),label,fill='white')
                g=gt[i].cpu().numpy()*[352,384]+[col*352,28];draw.line([tuple(x) for x in g],fill='#ffe454',width=2)
                if col:
                    p=predictions[label][i]*[352,384]+[col*352,28];draw.line([tuple(x) for x in p],fill='#39dfff',width=2)
            name=f'fold{fold}_{i:03d}.jpg';canvas.save(OUT/name,quality=95);visuals.append(dict(file=name,**meta[i]))
    write_json(OUT/'results.json',dict(results=results,visuals=visuals,inspection_pending=True,not_external_generalization=True))
    html=['<!doctype html><meta charset="utf-8"><title>Curve pose holdout</title><style>body{background:#202027;color:#eee;font:16px sans-serif}img{max-width:100%}</style><h1>Leave-one-pose-out development diagnostics</h1><p>Yellow=GT. Cyan=predicted curve. Each fold trained on the OTHER two poses; all cameras of held pose excluded from supervision. Original frozen initial heads, not600step adapted heads. Fixed1200steps. These poses were previously inspected for design: NOT a clean external test. Bottom GT is an occluded material line, not visible back annotation.</p>']
    for row in visuals:html.append(f'<h2>Pose{row["pose"]}, view{row["view"]}, context{row["key"]}</h2><img src="{row["file"]}">')
    (OUT/'index.html').write_text('\n'.join(html));status('complete');print('POSE_HOLDOUT_COMPLETE',flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(exist_ok=True);write_json(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()));raise
