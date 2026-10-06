"""Frozen-backbone2x2 feature/parameterization probe with source-separated curve GT."""
import collections,datetime as dt,fcntl,json,random,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'vggt-omega'),str(ROOT/'tools')]
import numpy as np
import torch
from torch import nn
from PIL import Image
from vggt_omega.models.heads.dense_head import DenseHead
from vggt_omega.models.heads.instance_pose_head import MultiInstanceMouseHead
from mousegeoforge.models.surface_curve_pose import SurfaceCurveDepthPoseModel
from mousegeoforge.surface_curve_supervision import cubic_bspline,curve_loss,points_to_polyline
from training.instance_pose_losses import instance_targets_from_batch,hungarian_match_instances
from run_depth_pose_ablation import write_json,fingerprint,move_batch
BASE=ROOT/'output/surface_curve_cycle_20261003/training_v1/pilot'
GT=ROOT/'output/surface_curve_cycle_20261003/diverse_curve_gt_v1'
OUT=ROOT/'output/surface_curve_cycle_20261003/diverse_curve_features_v1'

class FeatureProbe(nn.Module):
    def __init__(self,bounded):
        super().__init__();self.bounded=bounded
        self.delta=nn.Sequential(nn.LayerNorm(1024),nn.Linear(1024,256),nn.GELU(),nn.Linear(256,12))
        nn.init.zeros_(self.delta[-1].weight);nn.init.zeros_(self.delta[-1].bias)
    def forward(self,conditioning,boxes,base_logits):
        d=self.delta(conditioning).reshape(-1,6,2);c=boxes[:,:2,None].transpose(-1,-2);s=boxes[:,2:,None].transpose(-1,-2)
        controls=c+((base_logits+d).sigmoid()-.5)*s if self.bounded else c+(base_logits.sigmoid()-.5)*s+.5*d.tanh()
        return cubic_bspline(controls,33)

def now():return dt.datetime.now(dt.timezone.utc)
def main():
    assert not OUT.exists() and json.loads((GT/'qa.json').read_text())['numeric_passed']
    lock=(ROOT/'output/teacher_temporal_cycle_20261001/gpu_job.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    OUT.mkdir();torch.set_num_threads(2);torch.manual_seed(17);torch.backends.cuda.matmul.allow_tf32=False
    deadline=dt.datetime(2026,10,4,8,5,16,tzinfo=dt.timezone.utc)
    def status(stage,**kw):write_json(OUT/'status.json',dict(stage=stage,pid=__import__('os').getpid(),utc=now().isoformat(),**kw))
    status('extracting_features');initial=BASE/'initial_heads.pt'
    model=SurfaceCurveDepthPoseModel(nn.Identity(),DenseHead(),MultiInstanceMouseHead(num_keypoints=23)).cuda().eval()
    model.load_state_dict(torch.load(initial,map_location='cpu',weights_only=True),strict=True);model.requires_grad_(False)
    sidecars={r['scene']:r for r in json.loads((GT/'manifest.json').read_text())['records'] if r['accepted_for_training']}
    rows=[r for r in json.loads((BASE/'records.json').read_text()) if r['scene'] in sidecars or r['kind']=='curve']
    data=collections.defaultdict(list);meta=[]
    for r in rows:
        assert now()<deadline
        item=torch.load(BASE/'cache'/f"{r['key']}.pt",map_location='cpu',weights_only=False);inst=[];pose=[]
        a=model.instance_pose_head.instance_decoder.register_forward_hook(lambda m,args,o:inst.append(o.detach()))
        b=model.instance_pose_head.keypoint_decoder.register_forward_hook(lambda m,args,o:pose.append(o.detach()))
        try:
            with torch.no_grad():pred=model.forward_cached(item['batch']['images'].cuda(),[None if f is None else f.cuda() for f in item['cached']],item['patch_start'],coupling='off')
        finally:a.remove();b.remove()
        assert len(inst)==len(pose)==1
        pt=pose[0].reshape(2,6,23,256);target=instance_targets_from_batch(move_batch(item['batch'],'cuda'));mt=dict(target);mt['keypoint_valid']=torch.zeros_like(mt['keypoint_valid']);matches=hungarian_match_instances(pred,mt)[0]
        h,w=item['batch']['images'].shape[-2:]
        if r['kind']=='curve':cur=item['aux']['curve_target'][0].numpy()/[w,h];curve_hash=fingerprint(ROOT/r['scene']/'surface_curve.npz')
        else:
            record=sidecars[r['scene']];assert fingerprint(ROOT/r['scene']/'keypoints.npz')==record['source_annotation_sha256']
            assert item['source_hashes'][r['scene']+'/keypoints.npz']==record['source_annotation_sha256']
            with np.load(ROOT/record['sidecar']) as z:
                inds=[z['view_ids'].tolist().index(i) for i in r['views']];cur=z['uv'][inds].copy()
                assert z['position_valid'][inds].all()
            for v,vid in enumerate(r['views']):
                # Cached source RGB provenance identifies the exact original dimensions.
                source=next(ROOT/p for p in item['source_hashes'] if '/blended_images/' in p and Path(p).stem==f'{vid:08d}')
                with Image.open(source) as im:sw,sh=im.size
                cur[v]/=[sw,sh]
            curve_hash=fingerprint(ROOT/record['sidecar'])
        for v,(qs,ms) in enumerate(matches):
            for q,m in zip(qs.tolist(),ms.tolist()):
                tok=inst[0][v,q];data['det'].append(tok.repeat(4));data['pose'].append(pt[v,q,[4,5,6,7]].flatten())
                data['boxes'].append(pred['instance_boxes_cxcywh'][0,v,q]);data['gt'].append(torch.tensor(cur[v,m],device='cuda',dtype=torch.float32));data['pixels'].append(torch.tensor([w,h],device='cuda',dtype=torch.float32))
                data['base_logits'].append(model.surface_curve_head.net(tok).reshape(6,2))
                meta.append(dict(key=r['key'],scene=r['scene'],role=r['role'],kind=r['kind'],dataset=r.get('dataset',Path(r['scene']).parent.name),source_group=r.get('source_group','new_static3'),view=r['views'][v],query=q,instance=m,curve_sha256=curve_hash))
        status('extracting_features',records=len({x['key'] for x in meta}),total=len(rows))
    tensors={k:torch.stack(v).detach() for k,v in data.items()};del model,data;torch.cuda.empty_cache()
    torch.save(dict(tensors={k:v.cpu() for k,v in tensors.items()},metadata=meta),OUT/'frozen_inputs.pt')
    train=[i for i,m in enumerate(meta) if m['role']=='train'];val=[i for i,m in enumerate(meta) if m['role']=='validation'];assert train and val
    assert not {meta[i]['scene'] for i in train}&{meta[i]['scene'] for i in val}
    buckets=collections.defaultdict(list)
    for i in train:buckets[meta[i]['dataset']].append(i)
    rng=random.Random(17);keys=sorted(buckets);schedule=[[rng.choice(buckets[rng.choice(keys)]) for _ in range(8)] for _ in range(2400)]
    arms=[(f,b) for f in ['det','pose'] for b in [True,False]]
    write_json(OUT/'protocol.json',dict(source_initial_sha256=fingerprint(initial),steps=2400,seed=17,batch_size=8,learning_rate=.001,
        arms=[dict(conditioning=f,bounded=b) for f,b in arms],conditioning='1024dim: repeated instance token vs four neck/back/tail-root keypoint decoder tokens. SAME residual-network parameters, initialization and minibatch schedule.',
        no_ground_truth_in_forward=True,existing_heads_frozen=True,train_observations=len(train),validation_observations=len(val),source_records=len(rows),metadata=meta,
        loss='Known synthetic amodal curves only; no real curve target; no visibility supervision',
        sampling='Uniform dataset then uniform instance observation; same2400minibatches in all arms',
        limitations=['Only15old train poses+3newposes;30old validation poses','Validation source groups inherited; same assets, not real generalization','Coordinate mappings differ in effective step size','Matched queries use boxes/objectness for supervised association; this is not detection-aware success','No checkpoint selection; report fixed final step and trajectories']))
    def evaluate(probe,feature,ids):
        values=collections.defaultdict(list)
        with torch.no_grad():
            pred=probe(tensors[feature][ids],tensors['boxes'][ids],tensors['base_logits'][ids])*tensors['pixels'][ids,None]
            gt=tensors['gt'][ids]*tensors['pixels'][ids,None]
            for p,g in zip(pred,gt):
                values['pred_to_gt_px'].extend(points_to_polyline(p,g,torch.ones(len(g),device='cuda',dtype=torch.bool))[0].sqrt().tolist())
                values['gt_to_pred_px'].extend(points_to_polyline(g,p,torch.ones(len(p),device='cuda',dtype=torch.bool))[0].sqrt().tolist());values['endpoints_px'].extend((p[[0,-1]]-g[[0,-1]]).norm(dim=-1).tolist())
        return {k:dict(mean=float(np.mean(v)),median=float(np.median(v)),p90=float(np.quantile(v,.9))) for k,v in values.items()}
    results=[];predictions={}
    for feature,bounded in arms:
        arm=feature+('_bounded' if bounded else '_free');torch.manual_seed(17);probe=FeatureProbe(bounded).cuda();optimizer=torch.optim.AdamW(probe.parameters(),lr=.001,weight_decay=.001)
        history=[dict(step=0,train=evaluate(probe,feature,train),validation=evaluate(probe,feature,val))]
        for step,ids in enumerate(schedule,1):
            assert now()<deadline
            optimizer.zero_grad(set_to_none=True);pixels=tensors['pixels'][ids];p=probe(tensors[feature][ids],tensors['boxes'][ids],tensors['base_logits'][ids])*pixels[:,None];g=tensors['gt'][ids]*pixels[:,None]
            loss=torch.stack([curve_loss(pp,gg,torch.ones(len(gg),device='cuda',dtype=torch.bool),full_curve=True,scale=768) for pp,gg in zip(p,g)]).mean()
            assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(probe.parameters(),1.);assert torch.isfinite(norm);optimizer.step()
            if step%400==0:
                row=dict(step=step,loss=float(loss.detach()),train=evaluate(probe,feature,train),validation=evaluate(probe,feature,val));history.append(row);write_json(OUT/f'{arm}_history.json',history);status('training',arm=arm,step=step);print(arm,step,row['train']['pred_to_gt_px']['mean'],row['validation']['pred_to_gt_px']['mean'],flush=True)
        torch.save(dict(state=probe.state_dict(),feature=feature,bounded=bounded,step=2400),OUT/f'{arm}.pt');results.append(dict(arm=arm,**history[-1]));write_json(OUT/'partial_results.json',results)
        with torch.no_grad():predictions[arm]=probe(tensors[feature],tensors['boxes'],tensors['base_logits']).cpu().numpy()
    np.savez_compressed(OUT/'predictions.npz',**predictions);write_json(OUT/'results.json',dict(results=results,inspection_pending=True,not_promoted=True));status('complete')
if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(exist_ok=True);write_json(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()));raise
