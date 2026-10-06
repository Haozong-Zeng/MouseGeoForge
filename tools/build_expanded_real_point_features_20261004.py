"""Append eligible real TRAIN features; preserve original236observations exactly."""
import collections,datetime as dt,fcntl,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'vggt-omega'),str(ROOT/'tools')]
import numpy as np
import torch
from vggt_omega.models import VGGTOmega
from mousegeoforge.models.surface_curve_pose import SurfaceCurveDepthPoseModel
from mousegeoforge.uncertain_back_points import back_axes
from training.instance_pose_losses import instance_targets_from_batch,hungarian_match_instances
from run_surface_curve_pilot_20261003 import load_item
from run_depth_pose_ablation import fingerprint,write_json,move_batch
CYCLE=ROOT/'output/surface_curve_cycle_20261003';DERIVED=CYCLE/'expanded_real_point_train_v1';OUT=CYCLE/'expanded_real_point_features_v1'


def main():
    assert not OUT.exists();report=json.loads((DERIVED/'real_derived_report.json').read_text());assert report['passed']
    lock=(ROOT/'output/teacher_temporal_cycle_20261001/gpu_job.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert torch.cuda.is_available();OUT.mkdir();(OUT/'memory').mkdir();(OUT/'parts').mkdir();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False
    deadline=dt.datetime(2026,10,4,8,5,16,tzinfo=dt.timezone.utc)
    def status(stage,**kw):write_json(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),**kw))
    status('loading_model');old_path=CYCLE/'back_point_probe_inputs_v1/real_inputs.pt';old=torch.load(old_path,map_location='cpu',weights_only=False)
    spatial=VGGTOmega(enable_camera=False,enable_depth=True,enable_multi_instance_pose=True,num_keypoints=23)
    raw=torch.load(ROOT/'vggt-omega/vggt_omega_1b_512.pt',map_location='cpu',weights_only=True,mmap=True);selected={k:v for k,v in raw.items() if k.startswith(('aggregator.','dense_head.'))}
    result=spatial.load_state_dict(selected,strict=False);assert not result.unexpected_keys and all(k.startswith('instance_pose_head.') for k in result.missing_keys);del raw,selected
    model=SurfaceCurveDepthPoseModel.from_spatial_model(spatial).cuda().eval();del spatial
    initial=CYCLE/'training_v1/pilot/initial_heads.pt';result=model.load_state_dict(torch.load(initial,map_location='cpu',weights_only=True),strict=False)
    assert not result.unexpected_keys and all(k.startswith('aggregator.') for k in result.missing_keys);model.requires_grad_(False)

    def forward_new(item):
        batch=move_batch(item['batch'],'cuda');torch.backends.cudnn.allow_tf32=False
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):features,start=model.aggregator(batch['images'])
        cached=[x if j in (4,11,17,23) else None for j,x in enumerate(features)];del features
        torch.backends.cudnn.allow_tf32=True;memory=[];tokens=[]
        a=model.instance_pose_head.instance_decoder.register_forward_pre_hook(lambda m,args:memory.append(args[1].detach()))
        b=model.instance_pose_head.keypoint_decoder.register_forward_hook(lambda m,args,out:tokens.append(out.detach()))
        try:
            with torch.no_grad():pred=model.forward_cached(batch['images'],cached,start,coupling='off')
        finally:a.remove();b.remove()
        h,w=batch['images'].shape[-2:]
        return batch,pred,memory[0].transpose(1,2).reshape(2,256,h//16,w//16).cpu(),tokens[0].reshape(2,6,23,256).cpu()

    golden_meta=old['metadata'][0];golden=torch.load(CYCLE/f'training_v1/pilot/cache/{golden_meta["cache_key"]}.pt',map_location='cpu',weights_only=False)
    _,_,_,tokens=forward_new(golden);delta=float((tokens[golden_meta['cache_view_index'],golden_meta['query'],[4,5,6,7]].flatten()-old['tensors']['pose'][0]).abs().max())
    assert delta<2e-4,('Golden numeric path mismatch',delta);write_json(OUT/'golden_reproduction.json',dict(max_pose_delta=delta,reference_observation=0,source_sha256=fingerprint(old_path),aggregator='bf16autocast/cudnnTF32false',decoder='float32/cudnnTF32true',no_new_validation=True))
    values={k:[x.clone() for x in v] for k,v in old['tensors'].items()};meta=list(old['metadata']);seen={(m['session'],m['frame'],m['view']) for m in meta}
    original_manifest=json.loads((CYCLE/'real_point_spatial_memory_v1/manifest.json').read_text());memory_records=list(original_manifest['records']);records=[]
    for r in report['records']:
        assert r['role']=='train' and r['session'] not in ('GR2','UR2','UR3','AZR2L2','FR2')
        ids=r['views'];pairs=[ids[i:i+2] for i in range(0,len(ids),2)]
        if len(pairs[-1])==1:pairs[-1].append(ids[0])
        for pair in pairs:records.append(dict(r,views=pair,kind='real',image_hw=[768,704],key=f'e{len(records):04d}'))
    write_json(OUT/'records.json',records)
    for n,r in enumerate(records):
        assert dt.datetime.now(dt.timezone.utc)<deadline;item=load_item(r);batch,pred,maps,tokens=forward_new(item)
        target=instance_targets_from_batch(batch);matching=dict(target);matching['keypoint_valid']=torch.zeros_like(target['keypoint_valid']);matches=hungarian_match_instances(pred,matching)[0];bindings=[];part_values=collections.defaultdict(list);part_meta=[]
        for v,(qs,ms) in enumerate(matches):
            identity=(r['session'],r['frame'],r['views'][v])
            if identity in seen:continue
            seen.add(identity);assert len(qs)==1 and ms.tolist()==[0];q=int(qs[0]);i=len(meta);pose=tokens[v,q,[4,5,6,7]].flatten()
            with torch.no_grad():coarse=model.instance_pose_head.coarse_keypoint_xy(tokens[v,q,5:7].cuda()).sigmoid().cpu()
            xy=target['keypoint_xy'][0,v,0,5:7].cpu();valid=target['keypoint_valid'][0,v,0,5:7].cpu();pixels=torch.tensor([704.,768.]);t=item['aux']['tangent'][0,v,0,5:7];tv=item['aux']['tangent_valid'][0,v,0,5:7]
            pt,pr=back_axes(xy*pixels,valid,primary_tangent=t,primary_valid=tv);ft,fr=back_axes(xy*pixels,valid,primary_tangent=t,primary_valid=tv,use_pair_fallback=True)
            row=dict(pose=pose,coarse_xy=coarse,gt_xy=xy,valid=valid,pixels=pixels,primary_tangent=pt,primary_reliability=pr,pair_tangent=ft,pair_reliability=fr)
            for k,value in row.items():values[k].append(value.clone());part_values[k].append(value.clone())
            m=dict(session=r['session'],frame=r['frame'],view=r['views'][v],role='train',query=q,cache_key=r['key'],cache_view_index=v,scene=r['scene'],kind='real',dataset='PyMouseStudio',image_hw=[768,704]);meta.append(m);part_meta.append(m);bindings.append(dict(observation=i,view_index=v,query=q))
        path=OUT/'memory'/f'{r["key"]}.pt';torch.save(dict(memory=maps.half(),image_hw=[768,704],views=r['views'],bindings=bindings),path)
        memory_records.append(dict(key=r['key'],path=path.relative_to(ROOT).as_posix(),sha256=fingerprint(path),bindings=bindings,source_hashes=item['source_hashes']))
        torch.save(dict(tensors={k:torch.stack(v) for k,v in part_values.items()},metadata=part_meta),OUT/'parts'/f'{r["key"]}.pt')
        status('extracting',completed=n+1,total=len(records),observations=len(meta));print('FEATURES',n+1,len(records),flush=True)
    tensors={k:torch.stack(v) for k,v in values.items()};assert meta[:236]==old['metadata']
    for k,v in old['tensors'].items():assert torch.equal(tensors[k][:236],v),k
    assert len(meta)==1091 and sum(m['role']=='validation' for m in meta)==48 and all(m['role']=='train' for m in meta[236:])
    torch.save(dict(tensors=tensors,metadata=meta),OUT/'real_inputs.pt')
    write_json(OUT/'memory_manifest.json',dict(records=memory_records,observations=len(meta),source_inputs_sha256=fingerprint(OUT/'real_inputs.pt'),old_inputs_sha256=fingerprint(old_path),original_prefix_count=236,original_prefix_bitwise_equal=True,golden_pose_delta=delta))
    write_json(OUT/'manifest.json',dict(observations=len(meta),train=1043,validation=48,additional_train=855,train_groups=156,validation_groups=8,source_initial_sha256=fingerprint(initial),original_prefix_bitwise_equal=True,no_new_validation=True,no_labels_changed=True,no_locked_holdout=True))
    status('complete')


if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(exist_ok=True);write_json(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()));raise
