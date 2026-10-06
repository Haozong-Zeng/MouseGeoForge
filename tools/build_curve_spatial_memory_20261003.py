"""Cache frozen image-space decoder memory for the existing326observations."""
import collections, datetime as dt, fcntl, json, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'vggt-omega'),str(ROOT/'tools')]
import torch
from vggt_omega.models import VGGTOmega
from mousegeoforge.models.surface_curve_pose import SurfaceCurveDepthPoseModel
from run_surface_curve_pilot_20261003 import load_item
from run_depth_pose_ablation import fingerprint,write_json
CYCLE=ROOT/'output/surface_curve_cycle_20261003'
SOURCE=CYCLE/'expanded_curve_features_v1'
OUT=CYCLE/'curve_spatial_memory_v2'


def main():
    assert not OUT.exists()
    lock=(ROOT/'output/teacher_temporal_cycle_20261001/gpu_job.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert torch.cuda.is_available();OUT.mkdir();(OUT/'memory').mkdir()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    deadline=dt.datetime(2026,10,4,8,5,16,tzinfo=dt.timezone.utc)
    def status(stage,**kw):write_json(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),**kw))
    status('loading_model')
    saved=torch.load(SOURCE/'frozen_inputs.pt',map_location='cpu',weights_only=False)
    groups=collections.defaultdict(list)
    for i,m in enumerate(saved['metadata']):groups[m['key']].append(i)
    new_records={r['key']:r for r in json.loads((SOURCE/'manifest.json').read_text())['source_records']}
    spatial=VGGTOmega(enable_camera=False,enable_depth=True,enable_multi_instance_pose=True,num_keypoints=23)
    checkpoint=torch.load(ROOT/'vggt-omega/vggt_omega_1b_512.pt',map_location='cpu',weights_only=True,mmap=True)
    selected={k:v for k,v in checkpoint.items() if k.startswith(('aggregator.','dense_head.'))}
    result=spatial.load_state_dict(selected,strict=False)
    assert not result.unexpected_keys and all(k.startswith('instance_pose_head.') for k in result.missing_keys)
    del selected,checkpoint
    model=SurfaceCurveDepthPoseModel.from_spatial_model(spatial).cuda().eval();del spatial
    initial=CYCLE/'training_v1/pilot/initial_heads.pt'
    assert fingerprint(initial)==json.loads((SOURCE/'manifest.json').read_text())['source_initial_sha256']
    result=model.load_state_dict(torch.load(initial,map_location='cpu',weights_only=True),strict=False)
    assert not result.unexpected_keys and all(k.startswith('aggregator.') for k in result.missing_keys)
    model.requires_grad_(False)
    records=[];max_delta=0.
    for n,(key,ids) in enumerate(sorted(groups.items())):
        assert dt.datetime.now(dt.timezone.utc)<deadline
        # Original compact-feature extraction left cuDNN TF32 at True; the
        # later expanded TRAIN builder explicitly used False. Reproduce each
        # source's numeric path, rather than silently changing its features.
        torch.backends.cudnn.allow_tf32 = key not in new_records
        if key in new_records:
            item=load_item(new_records[key]['record'])
            assert item['source_hashes']==new_records[key]['source_hashes']
        else:
            item=torch.load(CYCLE/f'training_v1/pilot/cache/{key}.pt',map_location='cpu',weights_only=False)
            assert all(fingerprint(ROOT/p)==sha for p,sha in item['source_hashes'].items())
        memory=[];pose=[]
        a=model.instance_pose_head.instance_decoder.register_forward_pre_hook(lambda m,args:memory.append(args[1].detach()))
        b=model.instance_pose_head.keypoint_decoder.register_forward_hook(lambda m,args,out:pose.append(out.detach()))
        images=item['batch']['images'].cuda()
        try:
            with torch.no_grad():
                if 'cached' in item:pred=model.forward_cached(images,[None if x is None else x.cuda() for x in item['cached']],item['patch_start'],coupling='off')
                else:pred=model(images,coupling='off')
        finally:a.remove();b.remove()
        h,w=images.shape[-2:];ph,pw=h//16,w//16
        assert memory[0].shape==(2,ph*pw,256)
        maps=memory[0].transpose(1,2).reshape(2,256,ph,pw).cpu()
        assert torch.isfinite(maps).all()
        pt=pose[0].reshape(2,6,23,256).cpu();bindings=[]
        for i in ids:
            m=saved['metadata'][i];v=item['record']['views'].index(m['view']);q=m['query']
            delta=float((pt[v,q,[4,5,6,7]].flatten()-saved['tensors']['pose'][i]).abs().max())
            max_delta=max(max_delta,delta)
            assert delta<2e-4,(key,i,delta)
            assert torch.allclose(pred['instance_boxes_cxcywh'][0,v,q].cpu(),saved['tensors']['boxes'][i],atol=2e-5,rtol=0)
            bindings.append(dict(observation=i,view_index=v,query=q))
        path=OUT/'memory'/f'{key}.pt'
        torch.save(dict(memory=maps.half(),image_hw=[h,w],views=item['record']['views'],bindings=bindings),path)
        records.append(dict(key=key,path=path.relative_to(ROOT).as_posix(),sha256=fingerprint(path),bindings=bindings,source_hashes=item['source_hashes'],cudnn_allow_tf32=torch.backends.cudnn.allow_tf32))
        status('extracting',completed=n+1,total=len(groups));print('MEMORY',n+1,len(groups),flush=True)
    write_json(OUT/'manifest.json',dict(records=records,observations=len(saved['metadata']),
        source_inputs_sha256=fingerprint(SOURCE/'frozen_inputs.pt'),initial_heads_sha256=fingerprint(initial),
        max_pose_token_reproduction_error=max_delta,
        memory_semantics='Original instance decoder image memory including sine position encoding,256channels,patch16. Frozen, no GT locations.',
        storage='fp16 maps per paired record, not per instance; preserve original compact features/targets unchanged',
        no_training_or_gt_query_sampling=True))
    status('complete')


if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(exist_ok=True);write_json(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()));raise
