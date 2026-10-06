"""Bounded three-arm 768 variable-view adaptation with frozen contextual caches."""
import argparse
from collections import Counter, defaultdict, OrderedDict
from dataclasses import replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from types import SimpleNamespace

from run_depth_pose_ablation import (ROOT, torch, np, VGGTOmega, DepthPoseModel, load_batch,
    load_scene, fingerprint, write_json, restore, tensor_state, state_hash, cached_forward,
    depth_loss, instance_pose_loss, InstancePoseLossWeights, instance_targets_from_batch)
from training.finetune_blendedmvs import median_camera_baseline
from mousegeoforge.pilot_metrics import evaluate_scene
from mousegeoforge.view_sampling import sample_view_ids

CONFIG = ROOT/'configs/experiments/variable_view_adaptation_v1.json'
OUTPUT = ROOT/'output/variable_view_adaptation_v1_retry1'
CACHE_SOURCE = ROOT/'output/variable_view_adaptation_v1'


def spaced(records,count):
    if len(records)<count or count<2: raise ValueError('Insufficient source poses')
    return [records[round(i*(len(records)-1)/(count-1))] for i in range(count)]


def mean_metrics(records):
    groups = defaultdict(list)
    for row in records: groups[row['group']].append(row['metrics'])
    return {group:{key:float(np.mean([m[key] for m in metrics if m[key] is not None]))
                   if any(m[key] is not None for m in metrics) else None for key in metrics[0]}
            for group,metrics in groups.items()}


def tensor_bytes(value):
    if isinstance(value,torch.Tensor): return value.numel()*value.element_size()
    if isinstance(value,dict): return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value,(tuple,list)): return sum(tensor_bytes(v) for v in value)
    return 0


def build_protocol(cfg):
    original = json.loads((ROOT/cfg['manifest']).read_text())
    pools = {split:sorted([r for r in original['records'] if r['split']==split and r['expected_instances']==2],
                         key=lambda r:r['frame']) for split in ('train','val')}
    train = spaced(pools['train'],cfg['train_scenes']); dev = spaced(pools['val'],cfg['development_scenes'])
    if {r['source_group'] for r in train}&{r['source_group'] for r in dev}: raise ValueError('Session leakage')
    if {r['scene'] for r in train}&{r['scene'] for r in dev}: raise ValueError('Pose leakage')
    records, arms, evaluation = {}, {a:[] for a in cfg['arms']}, []
    sources = {}
    def register(base,order,scale):
        r = dict(base,views=order,diagnostic_view_order=order,normalization_scale=scale,
                 source_revision=sources[base['scene']])
        key = hashlib.sha256(json.dumps(r,sort_keys=True).encode()).hexdigest()
        records[key] = r
        return key
    for base in train+dev:
        scene_dir = ROOT/base['scene']
        digest = hashlib.sha256()
        for p in sorted(scene_dir.rglob('*')):
            if p.is_file():
                digest.update(str(p.relative_to(scene_dir)).encode()); digest.update(fingerprint(p).encode())
        sources[base['scene']] = digest.hexdigest()
        scene = load_scene(scene_dir); cameras = sorted(scene.frames)
        if cameras != [0,1,2,3]: raise ValueError('This bounded pilot requires exactly four audited real cameras')
        scale = median_camera_baseline(np.stack([scene.frames[v].extrinsic for v in cameras]))
        if base in train:
            i = train.index(base)
            arms['fixed_two'].append([register(base,[2,0],scale)])
            arms['random_two'].append([register(base,sample_view_ids(cameras,sample_index=i*100+j*4+1),scale)
                                       for j in range(cfg['subset_variants_per_scene'])])
            arms['variable_one_to_four'].append([register(base,sample_view_ids(cameras,sample_index=i*8+j),scale)
                                                 for j in range(cfg['subset_variants_per_scene'])])
        else:
            i = dev.index(base)
            for n in range(1,5):
                order = sample_view_ids(cameras,sample_index=10000+i*4+n-1)
                evaluation.append(dict(key=register(base,order,scale),group=f'V{n}'))
            evaluation.append(dict(key=register(base,[2,0],scale),group='fixed_pair'))
    rng = random.Random(cfg['seed']); schedule = []
    while len(schedule)<cfg['steps']:
        block = list(range(len(train))); rng.shuffle(block); schedule.extend(block)
    return dict(config=cfg,records=records,arms=arms,evaluation=evaluation,schedule=schedule[:cfg['steps']],
        cache_execution='bounded RAM LRU, at most 32 GiB; no new feature cache files; reuse validated pre-training cache where available',
        cache_source_protocol_sha256=fingerprint(CACHE_SOURCE/'protocol.json'),
        checkpoint_sha256=fingerprint(ROOT/cfg['checkpoint']),
        encoder_sha256=fingerprint(ROOT/'vggt-omega/vggt_omega_1b_512.pt'),
        manifest_sha256=fingerprint(ROOT/cfg['manifest']),
        code_sha256={p:fingerprint(ROOT/p) for p in ('tools/train_variable_view_adaptation.py',
            'tools/run_depth_pose_ablation.py','mousegeoforge/view_sampling.py','mousegeoforge/models/depth_pose.py',
            'vggt-omega/training/finetune_blendedmvs.py','vggt-omega/training/instance_pose_losses.py')},
        scope='32 original dual train poses; 8 original validation poses reused as development; no independent test claim')


def run(out,cfg,prepare_only):
    out.mkdir(parents=True,exist_ok=True)
    with (out/'running.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        lock.seek(0); lock.truncate(); lock.write(str(os.getpid())); lock.flush()
        status_path = out/'status.json'
        if status_path.exists() and json.loads(status_path.read_text())['state'] not in ('prepared',):
            raise RuntimeError('Existing run state; inspect it, never silently restart/overwrite')
        began = time.monotonic()
        def status(state,**extra):
            write_json(status_path,dict(state=state,pid=os.getpid(),elapsed_s=time.monotonic()-began,**extra))
        def budget():
            if time.monotonic()-began>cfg['max_run_hours']*3600: raise RuntimeError('Wall-clock budget exhausted')
        try:
            status('preparing')
            spec = build_protocol(cfg); protocol_path = out/'protocol.json'
            if protocol_path.exists() and json.loads(protocol_path.read_text())!=spec: raise ValueError('Protocol changed')
            write_json(protocol_path,spec); status('prepared',cached_records=len(spec['records']))
            if prepare_only: return
            torch.set_num_threads(2); torch.manual_seed(cfg['seed'])
            torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
            torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
            spatial = VGGTOmega(enable_camera=False,enable_depth=True,enable_multi_instance_pose=True,num_keypoints=23)
            weights = torch.load(ROOT/'vggt-omega/vggt_omega_1b_512.pt',weights_only=True,mmap=True)
            selected = {k:v for k,v in weights.items() if k.startswith(('aggregator.','dense_head.'))}
            loaded = spatial.load_state_dict(selected,strict=False)
            if loaded.unexpected_keys or any(not k.startswith('instance_pose_head.') for k in loaded.missing_keys): raise ValueError('Encoder load mismatch')
            del selected,weights
            model = DepthPoseModel.from_spatial_model(spatial).cuda().eval(); del spatial
            checkpoint = torch.load(ROOT/cfg['checkpoint'],weights_only=True,map_location='cpu')
            if checkpoint['checkpoint_sha256']!=spec['encoder_sha256']: raise ValueError('Wrong encoder')
            restore(model,checkpoint['model']); del checkpoint
            initial = tensor_state(model); initial_hash = state_hash(initial)
            old_spec=json.loads((CACHE_SOURCE/'protocol.json').read_text())
            if old_spec['encoder_sha256']!=spec['encoder_sha256']: raise ValueError('Cache encoder mismatch')
            for source in ('tools/run_depth_pose_ablation.py','vggt-omega/training/finetune_blendedmvs.py'):
                if old_spec['code_sha256'][source]!=spec['code_sha256'][source]: raise ValueError('Cache loader changed')
            memory=OrderedDict(); size=0
            cache_stats=dict(hits=0,disk_reads=0,reencoded=0,peak_bytes=0)
            def get_item(key):
                nonlocal size
                budget()
                if key in memory:
                    cache_stats['hits']+=1; memory.move_to_end(key); return memory[key]
                r=spec['records'][key]; source=CACHE_SOURCE/'cache'/f'{key}.pt'
                if source.exists():
                    if old_spec['records'].get(key)!=r: raise ValueError('Cache source record mismatch')
                    item=torch.load(source,weights_only=True)
                    if item['record']!=r: raise ValueError('Cached record mismatch')
                    cache_stats['disk_reads']+=1
                else:
                    batch=load_batch(r,num_views=len(r['views']))
                    model.aggregator.eval()
                    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                        cached,start=model.aggregator(batch['images'].cuda())
                    item=dict(record=r,batch=batch,patch_start=start,
                        cached=[x.detach().cpu() if x is not None and j in (4,11,17,23) else None for j,x in enumerate(cached)])
                    cache_stats['reencoded']+=1
                if item['batch']['view_ids'][0]!=','.join(map(str,r['views'])): raise AssertionError('View order mismatch')
                needed=tensor_bytes(item); limit=min(32,cfg['cache_limit_gib'])*1024**3
                if needed>limit: raise RuntimeError('One cache item exceeds memory budget')
                while size+needed>limit:
                    _,removed=memory.popitem(last=False); size-=tensor_bytes(removed)
                memory[key]=item; size+=needed; cache_stats['peak_bytes']=max(size,cache_stats['peak_bytes'])
                return item
            first=get_item(next(iter(spec['records'])))
            with torch.no_grad():
                _,a=cached_forward(model,first,'off'); b=model(first['batch']['images'].cuda(),coupling='off')
                for k in a: torch.testing.assert_close(a[k],b[k],atol=0,rtol=0)
            del first,a,b
            def evaluate(tag):
                model.eval(); rows=[]; folder=out/tag; folder.mkdir(exist_ok=True)
                for i,e in enumerate(spec['evaluation']):
                    budget(); item=get_item(e['key'])
                    with torch.no_grad():
                        batch,pred=cached_forward(model,item,'off'); target=instance_targets_from_batch(batch)
                        metrics,assignment=evaluate_scene(pred,batch,target)
                    rows.append(dict(group=e['group'],scene=item['record']['scene'],views=item['record']['views'],metrics=metrics))
                    torch.save(dict(record=item['record'],group=e['group'],oracle_assignment=assignment,
                        predictions={k:pred[k].cpu() for k in ('instance_keypoint_xy','instance_boxes_cxcywh','instance_objectness_logits')},
                        targets={k:v.cpu() for k,v in target.items() if k!='masks'},metrics=metrics),folder/f'prediction_{i:03d}.pt')
                    del item,batch,pred,target
                result=dict(records=rows,summary=mean_metrics(rows)); write_json(folder/'evaluation.json',result)
                return result
            status('initial_evaluation'); baseline=evaluate('initial')
            args=SimpleNamespace(semantic_label_weights='',body_depth_balance=.5,body_depth_fraction=.5,depth_gradient_weight=0.)
            loss_weights=InstancePoseLossWeights(keypoint_3d=0.,reprojection=0.,local_heatmap=0.,keypoint_uncertainty=0.)
            results={}
            for arm in cfg['arms']:
                torch.manual_seed(cfg['seed']); restore(model,initial)
                if state_hash(tensor_state(model))!=initial_hash: raise AssertionError('Unequal arm initialization')
                optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                    lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
                history=[]; visits=Counter(); images_seen=0; arm_start=time.monotonic()
                torch.cuda.reset_peak_memory_stats(); model.train()
                for step,scene_index in enumerate(spec['schedule'],1):
                    budget(); choices=spec['arms'][arm][scene_index]
                    key=choices[visits[scene_index]%len(choices)]; visits[scene_index]+=1
                    item=get_item(key)
                    optimizer.zero_grad(set_to_none=True)
                    batch,pred=cached_forward(model,item,'off'); target=instance_targets_from_batch(batch)
                    dl=depth_loss(pred['depth'],batch['depths'],batch['depth_masks'],batch['body_masks'],batch['semantic_masks'],args)['loss_depth']
                    pl,logs=instance_pose_loss(pred,target,loss_weights,image_size=batch['images'].shape[-2:])
                    loss=dl+pl
                    if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
                    loss.backward()
                    if any(p.grad is not None for p in model.aggregator.parameters()): raise AssertionError('Encoder gradient')
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True)
                    optimizer.step(); v=len(item['record']['views']); images_seen+=v
                    history.append(dict(step=step,scene=item['record']['scene'],views=item['record']['views'],
                        loss=float(loss.detach()),depth_loss=float(dl.detach()),pose_loss=float(pl.detach()),gradient_norm=float(norm),
                        images_seen=images_seen,elapsed_s=time.monotonic()-arm_start))
                    del item,batch,pred,target,loss,dl,pl
                    if step%25==0 or step==1:
                        write_json(out/arm/'history.json',history)
                        status('training',arm=arm,step=step,steps=cfg['steps'],images_seen=images_seen,cache_gib=size/1024**3,cache_stats=cache_stats)
                        print(arm,json.dumps(history[-1]),flush=True)
                trained=tensor_state(model); trained_hash=state_hash(trained)
                final=out/arm/'final_heads.pt'
                torch.save(dict(model=trained,checkpoint_sha256=spec['encoder_sha256'],protocol_sha256=fingerprint(protocol_path),
                    initial_hash=initial_hash,trained_hash=trained_hash,steps=cfg['steps'],arm=arm),final)
                reload=torch.load(final,weights_only=True); restore(model,reload['model'])
                if state_hash(tensor_state(model))!=trained_hash: raise AssertionError('Checkpoint reload mismatch')
                del trained,reload,optimizer
                status('final_evaluation',arm=arm); result=evaluate(arm)
                results[arm]=dict(**result,images_seen=images_seen,patch_tokens_seen=images_seen*(432//16)*(768//16),
                    training_seconds=history[-1]['elapsed_s'],peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3,
                    initialization_hash=initial_hash,checkpoint_reload_verified=True,encoder_gradients_absent=True)
                write_json(out/'results.partial.json',results)
            write_json(out/'results.json',dict(baseline=baseline,arms=results,config=cfg,protocol_sha256=fingerprint(protocol_path),
                       cache_gib=size/1024**3,cache_stats=cache_stats,scope=spec['scope'],production_promoted=False))
            status('completed',arms=list(results),total_steps=cfg['steps']*len(results))
            print('COMPLETED',flush=True)
        except BaseException as error:
            status('failed',error=repr(error)); raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--launch',action='store_true')
    args=parser.parse_args()
    if args.launch:
        args.output.mkdir(parents=True,exist_ok=True)
        with (args.output/'run.log').open('a') as log:
            proc=subprocess.Popen([sys.executable,'-u',__file__,'--output',str(args.output)],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(pid=proc.pid,log=str(args.output/'run.log')))); return
    run(args.output,json.loads(CONFIG.read_text()),args.prepare_only)


if __name__=='__main__': main()
