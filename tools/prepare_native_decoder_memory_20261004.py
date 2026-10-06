"""Native FP32 decoder memories through original numeric paths, exact goldens."""
import datetime as dt,fcntl,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import torch
from vggt_omega.models import VGGTOmega
from mousegeoforge.models.surface_curve_pose import SurfaceCurveDepthPoseModel
from run_surface_curve_pilot_20261003 import load_item
from run_depth_pose_ablation import fingerprint,write_json,move_batch
CYCLE=ROOT/'output/core23_spatial_cycle_20261004';OLD=ROOT/'output/surface_curve_cycle_20261003';OUT=CYCLE/'native_decoder_memory_v1'

def main():
    plan=json.loads((CYCLE/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'));lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not OUT.exists();OUT.mkdir();(OUT/'parts').mkdir();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False
    def status(stage,**kw):write_json(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),**kw))
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='extracting_native_fp32_decoder_memory',active_job=OUT.name,active_pid=os.getpid(),updated_utc=dt.datetime.now(dt.timezone.utc).isoformat());write_json(CYCLE/'state.json',state);status('loading')
    source=CYCLE/'detection_features_v1/manifest.json';manifest=json.loads(source.read_text());original=json.loads((CYCLE/'features_expanded_all_v1/manifest.json').read_text());initial=OLD/'training_v1/pilot/initial_heads.pt';encoder=ROOT/'vggt-omega/vggt_omega_1b_512.pt'
    assert fingerprint(initial)==original['initial_sha256'] and fingerprint(encoder)==original['encoder_sha256']
    spatial=VGGTOmega(enable_camera=False,enable_depth=True,enable_multi_instance_pose=True,num_keypoints=23);raw=torch.load(encoder,map_location='cpu',weights_only=True,mmap=True);chosen={k:v for k,v in raw.items() if k.startswith(('aggregator.','dense_head.'))};result=spatial.load_state_dict(chosen,strict=False);assert not result.unexpected_keys and all(k.startswith('instance_pose_head.') for k in result.missing_keys);del raw,chosen
    model=SurfaceCurveDepthPoseModel.from_spatial_model(spatial).cuda().eval().requires_grad_(False);del spatial;heads=torch.load(initial,map_location='cpu',weights_only=True);result=model.load_state_dict(heads,strict=False);assert not result.unexpected_keys and all(k.startswith('aggregator.') for k in result.missing_keys)
    decoder_keys=('instance_pose_head.keypoint_decoder.','instance_pose_head.keypoint_queries.','instance_pose_head.coarse_keypoint_xy.')
    decoder_initial=OUT/'initial_decoder.pt';torch.save({k[len('instance_pose_head.'):]:v for k,v in heads.items() if k.startswith(decoder_keys)},decoder_initial);del heads
    records=[];images=0;max_gold=max_replay=max_coarse=0.;byte_count=0
    for n,rec in enumerate(manifest['records']):
        assert dt.datetime.now(dt.timezone.utc)<deadline
        if not rec['count']:continue
        assert fingerprint(ROOT/rec['path'])==rec['sha256'];gold=torch.load(ROOT/rec['path'],map_location='cpu',weights_only=False)['observations'];r=rec['record'];assert r.get('session') not in ['GR2','UR2','UR3']
        cache=OLD/f'training_v1/pilot/cache/{r["key"]}.pt';item=torch.load(cache,map_location='cpu',weights_only=False) if r['key'].isdigit() else load_item(r)
        assert item['source_hashes']==rec['source_hashes'] and all(fingerprint(ROOT/p)==sha for p,sha in rec['source_hashes'].items());batch=move_batch(item['batch'],'cuda');memory=[];inst=[];pose=[];head=model.instance_pose_head
        hooks=[head.instance_decoder.register_forward_pre_hook(lambda m,a:memory.append(a[1].detach())),head.instance_decoder.register_forward_hook(lambda m,a,o:inst.append(o.detach())),head.keypoint_decoder.register_forward_hook(lambda m,a,o:pose.append(o.detach()))]
        try:
            if 'cached' in item:
                torch.backends.cudnn.allow_tf32=True
                with torch.no_grad():pred=model.forward_cached(batch['images'],[None if x is None else x.cuda() for x in item['cached']],item['patch_start'],coupling='off')
            else:
                torch.backends.cudnn.allow_tf32=False
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):features,start=model.aggregator(batch['images'])
                cached=[x if j in (4,11,17,23) else None for j,x in enumerate(features)];del features;torch.backends.cudnn.allow_tf32=not r['key'].startswith('x')
                with torch.no_grad():pred=model.forward_cached(batch['images'],cached,start,coupling='off')
                del cached
        finally:
            for hook in hooks:hook.remove()
        assert len(memory)==len(inst)==len(pose)==1 and memory[0].dtype==torch.float32
        with torch.no_grad():
            seed=inst[0][:,:,None]+head.keypoint_queries.weight[None,None];replay=head.keypoint_decoder(seed.flatten(1,2),memory[0]);coarse=head.coarse_keypoint_xy(replay.reshape(len(r['views']),6,23,256)).sigmoid()
        replay_delta=float((replay-pose[0]).abs().max());coarse_delta=float((coarse-pred['instance_keypoint_coarse_xy'][0]).abs().max());assert replay_delta==0 and coarse_delta==0;max_replay=max(max_replay,replay_delta);max_coarse=max(max_coarse,coarse_delta)
        tokens=pose[0].reshape(len(r['views']),6,23,256).cpu()
        for o in gold:
            v=o['metadata']['view_index'];delta=float((tokens[v]-o['pose_tokens']).abs().max());assert delta==0,(r['key'],v,delta);max_gold=max(max_gold,delta);torch.testing.assert_close(inst[0][v].cpu(),o['instance_tokens'],atol=0,rtol=0)
        h,w=batch['images'].shape[-2:];native=memory[0].transpose(1,2).reshape(len(r['views']),256,h//head.patch_size,w//head.patch_size).cpu();path=OUT/'parts'/f'{r["key"]}.pt';torch.save(dict(memory=native,metadata=[o['metadata'] for o in gold]),path);byte_count+=path.stat().st_size;images+=len(gold)
        records.append(dict(key=r['key'],path=path.relative_to(ROOT).as_posix(),sha256=fingerprint(path),detection_part=rec['path'],detection_sha256=rec['sha256'],count=len(gold),record=r,source_hashes=rec['source_hashes']))
        status('extracting',completed=n+1,total=len(manifest['records']),images=images,bytes=byte_count,max_gold_delta=max_gold);print('NATIVE_MEMORY',n+1,len(manifest['records']),images,flush=True)
    assert images==manifest['images']
    write_json(OUT/'manifest.json',dict(records=records,images=images,bytes=byte_count,max_gold_delta=max_gold,max_replay_delta=max_replay,max_coarse_delta=max_coarse,initial_sha256=fingerprint(initial),decoder_initial_sha256=fingerprint(decoder_initial),source_sha256=fingerprint(source),encoder_sha256=fingerprint(encoder),script_sha256=fingerprint(Path(__file__)),contract='Native-size FP32 memory INCLUDING positional embeddings from original instance_decoder prehook. Original6instance and138keypoint outputs exactlymatch; full decoder replay exactlyequal. No GT enters extraction. Readout continues its originalfp16roundtrip48x48path separately. Source labels/splits untouched.'))
    status('complete',images=images,max_gold_delta=max_gold,max_replay_delta=max_replay);state.update(stage='native_fp32_decoder_memory_ready',active_job=None,active_pid=None,updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),next='Nativefp32memory parity passed. Implement matched keypointdecoder frozen/unfrozen ablation: all6x23queries, preserve spatialreadout memoryfp16roundtrip, explicit pose-token gradients only when enabled; keep originalinstance decoder, ROI detector, encoder frozen. Same data/schedules/init. Start bounded800steps and3seeds, inspect convergence before extension. No promotion.');state['completed'].append('native_fp32_decoder_memory_exact_goldens');write_json(CYCLE/'state.json',state)

if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(exist_ok=True);write_json(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()));raise
