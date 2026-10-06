"""Training-set capacity probe only; frozen main model, no real curve GT fabricated."""
import collections,datetime as dt,fcntl,json,random,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'vggt-omega'),str(ROOT/'tools')]
import numpy as np
import torch
from torch import nn
from PIL import Image,ImageDraw
from vggt_omega.models.heads.dense_head import DenseHead
from vggt_omega.models.heads.instance_pose_head import MultiInstanceMouseHead
from mousegeoforge.models.surface_curve_pose import SurfaceCurveDepthPoseModel,SurfaceCurveHead
from mousegeoforge.surface_curve_supervision import cubic_bspline,curve_loss,points_to_polyline
from training.instance_pose_losses import instance_targets_from_batch,hungarian_match_instances
from run_depth_pose_ablation import write_json,fingerprint,move_batch
BASE=ROOT/'output/surface_curve_cycle_20261003/training_v1/pilot'
OUT=ROOT/'output/surface_curve_cycle_20261003/curve_capacity_v1'

class ResidualProbe(nn.Module):
    def __init__(self,base,mode):
        super().__init__();self.base=base;self.base.requires_grad_(False);self.mode=mode
        self.delta=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,256),nn.GELU(),nn.Linear(256,12))
        nn.init.zeros_(self.delta[-1].weight);nn.init.zeros_(self.delta[-1].bias)
    def forward(self,tokens,boxes):
        raw=self.base.net(tokens).reshape(-1,6,2);delta=self.delta(tokens).reshape(-1,6,2)
        center=boxes[:,:2,None].transpose(-1,-2).detach();size=boxes[:,2:,None].transpose(-1,-2).detach()
        initial=center+(raw.sigmoid()-.5)*size
        control=center+((raw+delta).sigmoid()-.5)*size if self.mode=='box_bounded' else initial+.5*delta.tanh()
        return cubic_bspline(control,33)

@torch.no_grad()
def measure(model,tokens,boxes,gt):
    pred=model(tokens,boxes);a=[];b=[];ends=[]
    for p,g in zip(pred,gt):
        a.extend(points_to_polyline(p,g,torch.ones(len(g),device=g.device,dtype=torch.bool))[0].sqrt().mul(768).tolist())
        b.extend(points_to_polyline(g,p,torch.ones(len(p),device=p.device,dtype=torch.bool))[0].sqrt().mul(768).tolist())
        ends.extend((p[[0,-1]]-g[[0,-1]]).norm(dim=-1).mul(768).tolist())
    def stats(x):return dict(mean=float(np.mean(x)),median=float(np.median(x)),p90=float(np.quantile(x,.9)))
    return dict(pred_to_gt_px=stats(a),gt_to_pred_px=stats(b),endpoints_px=stats(ends))

def main():
    assert (BASE/'complete.json').exists() and not OUT.exists()
    lock=(ROOT/'output/teacher_temporal_cycle_20261001/gpu_job.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    OUT.mkdir();torch.set_num_threads(2);torch.manual_seed(17);torch.backends.cuda.matmul.allow_tf32=False
    deadline=dt.datetime(2026,10,4,8,5,16,tzinfo=dt.timezone.utc)
    def status(stage,**more):write_json(OUT/'status.json',dict(stage=stage,pid=__import__('os').getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),**more))
    status('extracting_frozen_instance_tokens')
    path=BASE/'weak_points_curve6/latest.pt';cp=torch.load(path,map_location='cpu',weights_only=False);assert cp['step']==600
    model=SurfaceCurveDepthPoseModel(nn.Identity(),DenseHead(),MultiInstanceMouseHead(num_keypoints=23)).cuda().eval();model.load_state_dict(cp['model'],strict=True);model.requires_grad_(False)
    base_state={k:v.cpu().clone() for k,v in model.surface_curve_head.state_dict().items()};del cp
    rows=[r for r in json.loads((BASE/'records.json').read_text()) if r['kind']=='curve'];features=[];boxes=[];truth=[];meta=[];images=[]
    for r in rows:
        item=torch.load(BASE/'cache'/f"{r['key']}.pt",map_location='cpu',weights_only=False);capture=[]
        hook=model.instance_pose_head.instance_decoder.register_forward_hook(lambda m,a,o:capture.append(o.detach()))
        try:
            with torch.no_grad():out=model.forward_cached(item['batch']['images'].cuda(),[None if x is None else x.cuda() for x in item['cached']],item['patch_start'],coupling='off')
        finally:hook.remove()
        target=instance_targets_from_batch(move_batch(item['batch'],'cuda'));mt=dict(target);mt['keypoint_valid']=torch.zeros_like(mt['keypoint_valid']);match=hungarian_match_instances(out,mt)[0]
        h,w=item['batch']['images'].shape[-2:];assert (h,w)==(768,704)
        for v,(qs,ms) in enumerate(match):
            for q,m in zip(qs.tolist(),ms.tolist()):
                features.append(capture[0][v,q]);boxes.append(out['instance_boxes_cxcywh'][0,v,q]);truth.append(item['aux']['curve_target'][0,v,m].cuda()/torch.tensor([w,h],device='cuda'))
                meta.append(dict(key=r['key'],scene=r['scene'],view=r['views'][v],query=q,instance=m));images.append(item['batch']['images'][0,v].clone())
    tokens=torch.stack(features);boxes=torch.stack(boxes);gt=torch.stack(truth);del model,features,truth;torch.cuda.empty_cache()
    # Coordinate loss and diagnostics use native704x768 pixels, not a square-image fiction.
    pixel_scale=gt.new_tensor([704.,768.]);scale_ratio=pixel_scale/768.
    samples=len(gt);assert samples==24
    torch.save(dict(tokens=tokens.cpu(),boxes=boxes.cpu(),gt_normalized=gt.cpu(),metadata=meta),OUT/'frozen_inputs.pt')
    write_json(OUT/'protocol.json',dict(source_checkpoint_sha256=fingerprint(path),source_step=600,steps=1200,batch_size=4,seed=17,lr=.001,
        modes=['box_bounded','free_residual'],main_model_frozen=True,supervision='Only same3synthetic training poses,24 paired-context observations of21unique pose/cameras',
        purpose='Can the optional head fit known training curves? Not generalization, not model-promotion evidence',
        initialization='Both modes exactly reproduce same600-step curve; same learned residual parameterization and identical sampled order. Different mappings imply different effective coordinate step sizes; this is a capacity diagnostic, not an isolated causal ablation.',
        units='768long-side pixels;704x768 aspect ratio accounted',source_meta=meta))
    rng=random.Random(17);schedule=[rng.sample(range(samples),4) for _ in range(1200)];results={};predictions={}
    for mode in ['box_bounded','free_residual']:
        torch.manual_seed(17);base=SurfaceCurveHead(256).cuda();base.load_state_dict(base_state);probe=ResidualProbe(base,mode).cuda()
        with torch.no_grad():torch.testing.assert_close(probe(tokens,boxes),base(tokens,boxes)[1],rtol=0,atol=0)
        optimizer=torch.optim.AdamW(probe.delta.parameters(),lr=.001,weight_decay=.001);history=[]
        # measure uses long-side-normalized coordinates for Euclidean pixel distances.
        def metrics():
            class Scaled(nn.Module):
                def forward(self,t,b):return probe(t,b)*scale_ratio
            return measure(Scaled(),tokens,boxes,gt*scale_ratio)
        history.append(dict(step=0,**metrics()));status('training',mode=mode,step=0)
        for step,indices in enumerate(schedule,1):
            assert dt.datetime.now(dt.timezone.utc)<deadline,'Deadline reached'
            optimizer.zero_grad(set_to_none=True);pred=probe(tokens[indices],boxes[indices])*pixel_scale
            loss=torch.stack([curve_loss(p,g*pixel_scale,torch.ones(len(g),device='cuda',dtype=torch.bool),full_curve=True,scale=768) for p,g in zip(pred,gt[indices])]).mean()
            assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(probe.delta.parameters(),1.);assert torch.isfinite(norm);optimizer.step()
            if step%100==0:
                row=dict(step=step,loss=float(loss.detach()),**metrics());history.append(row);write_json(OUT/f'{mode}_history.json',history);status('training',mode=mode,step=step);print(mode,row,flush=True)
        results[mode]=history[-1];torch.save(dict(state=probe.state_dict(),mode=mode,step=1200),OUT/f'{mode}.pt');predictions[mode]=probe(tokens,boxes).detach().cpu().numpy()
    gt_cpu=gt.cpu().numpy();np.savez_compressed(OUT/'predictions.npz',gt=gt_cpu,**predictions)
    mosaics=[]
    for i in [0,1,8,9,16,17,20,21]:
        rgb=Image.fromarray(np.uint8(images[i].permute(1,2,0).numpy()*255)).resize((352,384));canvas=Image.new('RGB',(1056,412),(25,25,30));draw=ImageDraw.Draw(canvas)
        for col,label in enumerate(['GT','box_bounded','free_residual']):
            canvas.paste(rgb,(col*352,28));draw.text((col*352+6,6),label,fill='white')
            g=gt_cpu[i]*[352,384]+[col*352,28];draw.line([tuple(p) for p in g],fill='#ffe454',width=2)
            if label!='GT':
                p=predictions[label][i]*[352,384]+[col*352,28];draw.line([tuple(x) for x in p],fill='#39dfff',width=2)
        name=f'{i:03d}.jpg';canvas.save(OUT/name,quality=95);mosaics.append(dict(file=name,**meta[i]))
    write_json(OUT/'results.json',dict(results=results,mosaics=mosaics,inspection_pending=True,main_weights_unchanged=True,not_generalization=True))
    status('complete');print('CAPACITY_COMPLETE',flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(exist_ok=True);write_json(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()));raise
