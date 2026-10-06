"""Frozen spatial maps for EXACT existing real development observations."""
import collections,datetime as dt,fcntl,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'vggt-omega'),str(ROOT/'tools')]
import torch
from torch import nn
from vggt_omega.models.heads.dense_head import DenseHead
from vggt_omega.models.heads.instance_pose_head import MultiInstanceMouseHead
from mousegeoforge.models.surface_curve_pose import SurfaceCurveDepthPoseModel
from run_depth_pose_ablation import fingerprint,write_json
CYCLE=ROOT/'output/surface_curve_cycle_20261003'
SOURCE=CYCLE/'back_point_probe_inputs_v1/real_inputs.pt'
OUT=CYCLE/'real_point_spatial_memory_v1'


def main():
    assert not OUT.exists()
    lock=(ROOT/'output/teacher_temporal_cycle_20261001/gpu_job.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert torch.cuda.is_available();OUT.mkdir();(OUT/'memory').mkdir();torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    deadline=dt.datetime(2026,10,4,8,5,16,tzinfo=dt.timezone.utc)
    def status(stage,**kw):write_json(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),**kw))
    status('loading');saved=torch.load(SOURCE,map_location='cpu',weights_only=False);groups=collections.defaultdict(list)
    for i,m in enumerate(saved['metadata']):
        assert m['session'] not in ('GR2','UR2','UR3');groups[m['cache_key']].append(i)
    model=SurfaceCurveDepthPoseModel(nn.Identity(),DenseHead(),MultiInstanceMouseHead(num_keypoints=23)).cuda().eval()
    initial=CYCLE/'training_v1/pilot/initial_heads.pt';model.load_state_dict(torch.load(initial,map_location='cpu',weights_only=True),strict=True);model.requires_grad_(False)
    rows=[];max_delta=0.
    for n,(key,ids) in enumerate(sorted(groups.items())):
        assert dt.datetime.now(dt.timezone.utc)<deadline
        item=torch.load(CYCLE/f'training_v1/pilot/cache/{key}.pt',map_location='cpu',weights_only=False)
        assert all(fingerprint(ROOT/p)==sha for p,sha in item['source_hashes'].items())
        memory=[];pose=[]
        a=model.instance_pose_head.instance_decoder.register_forward_pre_hook(lambda m,args:memory.append(args[1].detach()))
        b=model.instance_pose_head.keypoint_decoder.register_forward_hook(lambda m,args,out:pose.append(out.detach()))
        images=item['batch']['images'].cuda()
        try:
            with torch.no_grad():model.forward_cached(images,[None if x is None else x.cuda() for x in item['cached']],item['patch_start'],coupling='off')
        finally:a.remove();b.remove()
        h,w=images.shape[-2:];maps=memory[0].transpose(1,2).reshape(2,256,h//16,w//16).cpu();pt=pose[0].reshape(2,6,23,256).cpu();bindings=[]
        assert torch.isfinite(maps).all()
        for i in ids:
            m=saved['metadata'][i];v=m['cache_view_index'];assert item['record']['views'][v]==m['view']
            token=pt[v,m['query'],[4,5,6,7]].flatten();delta=float((token-saved['tensors']['pose'][i]).abs().max());max_delta=max(max_delta,delta)
            assert delta<2e-4,(key,i,delta);bindings.append(dict(observation=i,view_index=v,query=m['query']))
        path=OUT/'memory'/f'{key}.pt';torch.save(dict(memory=maps.half(),image_hw=[h,w],bindings=bindings),path)
        rows.append(dict(key=key,path=path.relative_to(ROOT).as_posix(),sha256=fingerprint(path),bindings=bindings,source_hashes=item['source_hashes']))
        status('extracting',completed=n+1,total=len(groups))
    write_json(OUT/'manifest.json',dict(records=rows,observations=len(saved['metadata']),source_inputs_sha256=fingerprint(SOURCE),initial_heads_sha256=fingerprint(initial),max_pose_token_reproduction_error=max_delta,memory_semantics='Frozen original decoder memory including position; nativepatch16fp16. No label-based sampling or optimization. Same chosen instances and paired contexts as all previous point controls.'))
    status('complete')


if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(exist_ok=True);write_json(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()));raise
