"""Immutable real DEVELOPMENT compact inputs, no training or pseudo labels."""
import collections,datetime as dt,fcntl,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'vggt-omega'),str(ROOT/'tools')]
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from vggt_omega.models.heads.dense_head import DenseHead
from vggt_omega.models.heads.instance_pose_head import MultiInstanceMouseHead
from mousegeoforge.models.surface_curve_pose import SurfaceCurveDepthPoseModel
from training.instance_pose_losses import instance_targets_from_batch,hungarian_match_instances
from probe_diverse_curve_features_20261003 import FeatureProbe
from run_depth_pose_ablation import write_json,fingerprint,move_batch
CYCLE=ROOT/'output/surface_curve_cycle_20261003';PILOT=CYCLE/'training_v1/pilot'
OUT=CYCLE/'real_curve_adaptation_inputs_v1'


def main():
    assert not OUT.exists();lock=(ROOT/'output/teacher_temporal_cycle_20261001/gpu_job.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert torch.cuda.is_available();OUT.mkdir();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    deadline=dt.datetime(2026,10,4,8,5,16,tzinfo=dt.timezone.utc)
    def status(stage,**kw):write_json(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),**kw))
    status('loading')
    initial=PILOT/'initial_heads.pt';model=SurfaceCurveDepthPoseModel(nn.Identity(),DenseHead(),MultiInstanceMouseHead(num_keypoints=23)).cuda().eval()
    model.load_state_dict(torch.load(initial,map_location='cpu',weights_only=True),strict=True);model.requires_grad_(False)
    assert torch.count_nonzero(model.surface_curve_head.net[-1].weight)==0
    records=[r for r in json.loads((PILOT/'records.json').read_text()) if r['kind']=='real']
    reference=json.loads((CYCLE/'real_curve_transfer_v1/results.json').read_text())['rows']
    with np.load(CYCLE/'real_curve_transfer_v1/predictions.npz') as z:reference_pred=z['base_seed17'].copy()
    base=FeatureProbe(False).cuda().eval();base.load_state_dict(torch.load(CYCLE/'curve_data_scaling_v1/expanded_seed17.pt',map_location='cpu',weights_only=False)['state']);base.requires_grad_(False)
    seen=set();data=collections.defaultdict(list);meta=[];sources=[];max_error=0.
    for n,r in enumerate(records):
        assert dt.datetime.now(dt.timezone.utc)<deadline and r['session'] not in ('GR2','UR2','UR3')
        item=torch.load(PILOT/'cache'/f"{r['key']}.pt",map_location='cpu',weights_only=False)
        assert all(fingerprint(ROOT/p)==sha for p,sha in item['source_hashes'].items())
        batch=move_batch(item['batch'],'cuda');captured=[]
        hook=model.instance_pose_head.keypoint_decoder.register_forward_hook(lambda m,a,o:captured.append(o.detach()))
        try:
            with torch.no_grad():prediction=model.forward_cached(batch['images'],[None if x is None else x.cuda() for x in item['cached']],item['patch_start'],coupling='off')
        finally:hook.remove()
        target=instance_targets_from_batch(batch);matching=dict(target);matching['keypoint_valid']=torch.zeros_like(target['keypoint_valid']);matches=hungarian_match_instances(prediction,matching)[0]
        h,w=batch['images'].shape[-2:];pt=captured[0].reshape(2,6,23,256)
        for v,(qs,ms) in enumerate(matches):
            identity=(r['session'],r['frame'],r['views'][v])
            if identity in seen:continue
            seen.add(identity);assert len(qs)==1 and ms.tolist()==[0];q=int(qs[0]);i=len(meta)
            assert (reference[i]['session'],reference[i]['frame'],reference[i]['view'])==identity and reference[i]['query']==q
            pose=pt[v,q,[4,5,6,7]].flatten();box=prediction['instance_boxes_cxcywh'][0,v,q];logits=model.surface_curve_head.net[-1].bias.reshape(6,2)
            with torch.no_grad():p=base(pose[None],box[None],logits[None])[0].cpu().numpy()
            error=float(abs(p-reference_pred[i]).max());max_error=max(max_error,error);assert error<1e-6
            for key,value in dict(pose=pose,boxes=box,base_logits=logits,pixels=torch.tensor([w,h]),
                observed_xy=target['keypoint_xy'][0,v,0,5:7],observed_valid=target['keypoint_valid'][0,v,0,5:7],
                mask_small=F.interpolate(target['masks'][0,v,0][None,None].float(),(192,176),mode='nearest')[0,0].bool(),
                mask_valid=target['mask_valid'][0,v,0],box_valid=target['box_valid'][0,v,0]).items():data[key].append(value.detach().cpu().clone())
            meta.append(dict(session=r['session'],frame=r['frame'],view=r['views'][v],role=r['role'],query=q,cache_key=r['key'],cache_view_index=v,scene=r['scene'],kind='real',dataset='PyMouseStudio',image_hw=[h,w]))
        sources.append(dict(key=r['key'],source_hashes=item['source_hashes']));status('extracting',completed=n+1,total=len(records))
    tensors={k:torch.stack(v) for k,v in data.items()};assert len(meta)==236
    torch.save(dict(tensors=tensors,metadata=meta),OUT/'frozen_inputs.pt')
    write_json(OUT/'manifest.json',dict(observations=len(meta),counts={role:sum(m['role']==role for m in meta) for role in ['train','validation']},
        observed_points=int(tensors['observed_valid'].sum()),max_reference_prediction_error=max_error,source_initial_sha256=fingerprint(initial),
        sources=sources,visibility='unknown; only observed eligibility used',mask_small='192x176 nearest resize, sanity metric only',
        source_instance_association='Originalbox/objectnessonly,never06/07',no_locked_observations=True,no_gt_in_forward=True,no_pseudolabels=True))
    status('complete')


if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(exist_ok=True);write_json(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()));raise
