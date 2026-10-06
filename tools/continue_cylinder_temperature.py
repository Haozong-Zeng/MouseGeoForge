"""Paired +800-step continuation; initial optimizer reset is explicit and symmetric.

Only run after a saved continuation_decision.json authorizes the specific stage.
Historical 800-step runners/checkpoints are immutable. Later stages restore AdamW.
"""
import argparse
from collections import OrderedDict
import fcntl
import json
import os
import time
from types import SimpleNamespace
import numpy as np
from run_depth_pose_ablation import (ROOT,torch,VGGTOmega,DepthPoseModel,tensor_state,state_hash,
    fingerprint,write_json,cached_forward,instance_targets_from_batch,depth_loss,instance_pose_loss,InstancePoseLossWeights)
from train_cylinder_view_balance import oriented_batch,physical_metrics,scene_digest
from train_variable_view_replay import mean_metrics,tensor_bytes
from mousegeoforge.adaptive_temperature import install_adaptive_temperature,restore_adaptive,original_state_projection

BASE=ROOT/'output/cylinder_source_mix_v1'
FIRST=ROOT/'output/cylinder_adaptive_temperature_v1'
OUT=ROOT/'output/cylinder_temperature_continuation_v1'


def stage_paths(start):
    if start==800:
        return {'off':BASE/'source_mix','on':FIRST/'adaptive_temperature'}
    return {a:OUT/f'step_{start}'/a for a in ('off','on')}


def run(start,end):
    if end!=start+800 or start not in (800,1600,2400):raise ValueError('Only declared +800 stages, at most3200')
    output=OUT/f'step_{end}';output.mkdir(parents=True,exist_ok=True)
    with (ROOT/'output/cylinder_training.lock').open('a+') as glock,(output/'running.lock').open('a+') as lock:
        fcntl.flock(glock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (output/'protocol.json').exists():raise RuntimeError('Do not restart a recorded stage')
        began=time.monotonic()
        def status(state,**kw):write_json(output/'status.json',dict(state=state,pid=os.getpid(),elapsed_s=time.monotonic()-began,**kw))
        def budget():
            if time.monotonic()-began>7200:raise RuntimeError('Per-stage2h safety limit')
        try:
            decision_path=OUT/f'decision_{start}_to_{end}.json'
            decision=json.loads(decision_path.read_text());assert decision['authorized'] and decision['from_step']==start and decision['to_step']==end
            spec=json.loads((BASE/'protocol.json').read_text());first=json.loads((FIRST/'protocol.json').read_text())
            assert json.loads((BASE/'delivery_verification.json').read_text())['passed']
            assert json.loads((FIRST/'delivery_verification.json').read_text())['passed']
            if start>800:assert json.loads((OUT/f'step_{start}'/'delivery_verification.json').read_text())['passed']
            paths=stage_paths(start)
            code={**spec['code_sha256'],**first['code_sha256']}
            code['tools/continue_cylinder_temperature.py']=fingerprint(ROOT/'tools/continue_cylinder_temperature.py')
            for p,h in code.items():assert fingerprint(ROOT/p)==h
            assert all(scene_digest(ROOT/p)==h for p,h in spec['source_sha256'].items())
            checkpoint_files={a:p/('final_heads.pt' if start==800 else 'resume.pt') for a,p in paths.items()}
            protocol=dict(from_step=start,to_step=end,phase_steps=800,records=spec['records'],evaluation=spec['evaluation'],
                phase_schedule=spec['arms']['source_mix'],slots=spec['slots'],config=dict(spec['config'],steps=end,arms=['off','on']),
                source_sha256=spec['source_sha256'],code_sha256=code,encoder_sha256=spec['encoder_sha256'],
                checkpoints={a:dict(path=str(p.relative_to(ROOT)),sha256=fingerprint(p)) for a,p in checkpoint_files.items()},
                decision_sha256=fingerprint(decision_path),base_protocol_sha256=fingerprint(BASE/'protocol.json'),
                off_learning_rate=3e-6,on_old_learning_rate=3e-6,on_new_learning_rate=3e-4,
                optimizer_policy='Both AdamW reset at800 because historic OFF did not save optimizer; later stages restore BOTH exact optimizer/RNG states. Not uninterrupted original training.',
                scope='Same cyclic800 source_mix slots, no new poses/views/losses. Compare OFF/ON at equal cumulative steps; no production promotion or best-checkpoint selection.')
            write_json(output/'protocol.json',protocol);status('preparing')
            torch.set_num_threads(2);torch.manual_seed(17)
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
            spatial=VGGTOmega(enable_camera=False,enable_depth=True,enable_multi_instance_pose=True,num_keypoints=23)
            encoder=ROOT/'vggt-omega/vggt_omega_1b_512.pt';assert fingerprint(encoder)==spec['encoder_sha256']
            weights=torch.load(encoder,weights_only=True,mmap=True)
            loaded=spatial.load_state_dict({k:v for k,v in weights.items() if k.startswith(('aggregator.','dense_head.'))},strict=False)
            assert not loaded.unexpected_keys and all(k.startswith('instance_pose_head.') for k in loaded.missing_keys)
            del weights
            model=install_adaptive_temperature(DepthPoseModel.from_spatial_model(spatial)).cuda().eval();del spatial
            memory=OrderedDict();cache_size=0;cache_stats=dict(hits=0,reencoded=0,peak_bytes=0)
            def get_item(key):
                nonlocal cache_size
                budget()
                if key in memory:cache_stats['hits']+=1;memory.move_to_end(key);return memory[key]
                record=spec['records'][key];batch=oriented_batch(record,num_views=len(record['views']))
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):features,patch_start=model.aggregator(batch['images'].cuda())
                item=dict(record=record,batch=batch,patch_start=patch_start,
                    cached=[x.detach().cpu() if x is not None and i in (4,11,17,23) else None for i,x in enumerate(features)])
                needed=tensor_bytes(item);limit=32*1024**3
                if needed>limit:raise RuntimeError('Feature item exceeds cache limit')
                while memory and cache_size+needed>limit:
                    _,removed=memory.popitem(last=False);cache_size-=tensor_bytes(removed)
                memory[key]=item;cache_size+=needed;cache_stats['reencoded']+=1;cache_stats['peak_bytes']=max(cache_stats['peak_bytes'],cache_size)
                return item
            def evaluate(arm,tag,compare_to=None):
                model.eval();rows=[];folder=output/arm/tag;folder.mkdir(parents=True,exist_ok=True)
                for idx,entry in enumerate(spec['evaluation']):
                    item=get_item(entry['key'])
                    with torch.no_grad():
                        batch,pred=cached_forward(model,item,'off');target=instance_targets_from_batch(batch)
                        metric,assignment=physical_metrics(pred,batch,target)
                    if compare_to is not None:
                        saved=torch.load(compare_to/f'prediction_{idx:03d}.pt',weights_only=True,map_location='cpu')
                        for k in ('instance_keypoint_xy','instance_boxes_cxcywh','instance_objectness_logits'):
                            torch.testing.assert_close(pred[k].cpu(),saved['predictions'][k],atol=0,rtol=0)
                    record=item['record'];rows.append(dict(group=entry['group'],scene=record['scene'],views=record['views'],metrics=metric))
                    if compare_to is None:
                        torch.save(dict(record=record,group=entry['group'],oracle_assignment=assignment,metrics=metric,
                            predictions={k:v.cpu() for k,v in pred.items() if k.startswith('instance_') and 'mask' not in k},
                            targets={k:v.cpu() for k,v in target.items() if k!='masks'},
                            temperature_scale=model.instance_pose_head.last_temperature_scale.cpu()),folder/f'prediction_{idx:03d}.pt')
                    del item,batch,pred,target
                value=dict(records=rows,summary=mean_metrics(rows));write_json(folder/'evaluation.json',value);return value
            args=SimpleNamespace(semantic_label_weights='',body_depth_balance=.5,body_depth_fraction=.5,depth_gradient_weight=0.)
            lw=InstancePoseLossWeights(keypoint_3d=0.,reprojection=0.,local_heatmap=0.,keypoint_uncertainty=0.)
            results={};starting={};checks={}
            for arm in ('off','on'):
                torch.manual_seed(17)
                source=torch.load(checkpoint_files[arm],weights_only=True,map_location='cpu');restore_adaptive(model,source['model'])
                model.instance_pose_head.adaptive_temperature_enabled=arm=='on'
                model.instance_pose_head.adaptive_temperature.requires_grad_(arm=='on')
                assert state_hash(original_state_projection(tensor_state(model)))==state_hash(original_state_projection(source['model']))
                initial_hash=state_hash(tensor_state(model))
                special=list(model.instance_pose_head.adaptive_temperature.parameters());ids={id(p) for p in special}
                groups=[dict(params=[p for p in model.parameters() if p.requires_grad and id(p) not in ids],lr=3e-6)]
                if arm=='on':groups.append(dict(params=special,lr=3e-4))
                optimizer=torch.optim.AdamW(groups,lr=3e-6,weight_decay=.01)
                if start>800:optimizer.load_state_dict(source['optimizer'])
                reference=paths[arm] if start==800 else paths[arm]/'final'
                status('initial_evaluation',arm=arm)
                starting[arm]=evaluate(arm,'initial',compare_to=reference)
                assert state_hash(tensor_state(model))==initial_hash
                if start>800:
                    torch.set_rng_state(source['cpu_rng']);torch.cuda.set_rng_state(source['cuda_rng'])
                del source
                model.train();history=[];scale_history=[];train_start=time.monotonic();images_seen=0;tokens_seen=0
                for phase_step,choices in enumerate(protocol['phase_schedule'],1):
                    item=get_item(choices[0]);optimizer.zero_grad(set_to_none=True)
                    batch,pred=cached_forward(model,item,'off');target=instance_targets_from_batch(batch)
                    dl=depth_loss(pred['depth'],batch['depths'],batch['depth_masks'],batch['body_masks'],batch['semantic_masks'],args)['loss_depth']
                    pl,logs=instance_pose_loss(pred,target,lw,image_size=batch['images'].shape[-2:]);loss=pl+dl
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                    loss.backward();assert not any(p.grad is not None for p in model.aggregator.parameters())
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
                    record=item['record'];h,w=record['image_hw'];v=len(record['views']);images_seen+=v;tokens_seen+=v*(h//16)*(w//16)
                    scale=model.instance_pose_head.last_temperature_scale
                    history.append(dict(step=start+phase_step,phase_step=phase_step,scene=record['scene'],views=record['views'],image_hw=record['image_hw'],
                        slot_kind=spec['slots'][phase_step-1]['kind'],loss=float(loss.detach()),pose_loss=float(pl.detach()),depth_loss=float(dl.detach()),
                        gradient_norm=float(norm),images_seen=images_seen,patch_tokens_seen=tokens_seen,learning_rate=3e-6,
                        keypoint_loss=float(logs['loss_instance_keypoint_l1'].detach()),elapsed_s=time.monotonic()-train_start))
                    scale_history.append(dict(step=start+phase_step,mean=float(scale.mean()),min=float(scale.min()),max=float(scale.max())))
                    if arm=='off':assert scale.min()==scale.max()==1
                    del item,batch,pred,target,dl,pl,logs,loss
                    if phase_step%100==0:
                        write_json(output/arm/'history.json',history);write_json(output/arm/'temperature_history.json',scale_history)
                        status('training',arm=arm,step=start+phase_step,end_step=end,cache_stats=cache_stats)
                        print(arm,json.dumps(history[-1]),flush=True)
                model_state=tensor_state(model);trained_hash=state_hash(model_state)
                assert trained_hash!=initial_hash
                path=output/arm/'resume.pt'
                torch.save(dict(model=model_state,optimizer=optimizer.state_dict(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
                    step=end,arm=arm,trained_hash=trained_hash,initial_hash=initial_hash,protocol_sha256=fingerprint(output/'protocol.json')),path)
                reloaded=torch.load(path,weights_only=True,map_location='cpu');restore_adaptive(model,reloaded['model'])
                assert state_hash(tensor_state(model))==trained_hash
                del reloaded,model_state,optimizer
                status('final_evaluation',arm=arm)
                results[arm]=dict(**evaluate(arm,'final'),phase_images_seen=images_seen,phase_patch_tokens_seen=tokens_seen,
                    cumulative_steps=end,training_seconds=history[-1]['elapsed_s'],checkpoint_reload_verified=True,encoder_gradients_absent=True)
                checks[arm]=dict(start_prediction_max_difference=0,original_start_tensor_projection_equal=True,final_checkpoint_sha256=fingerprint(path))
                write_json(output/'results.partial.json',results)
            for p,h in code.items():assert fingerprint(ROOT/p)==h
            assert all(scene_digest(ROOT/p)==h for p,h in spec['source_sha256'].items())
            assert all(fingerprint(p)==protocol['checkpoints'][a]['sha256'] for a,p in checkpoint_files.items())
            for arm in ('off','on'):
                history=json.loads((output/arm/'history.json').read_text())
                assert len(history)==800
                for i,row in enumerate(history):
                    record=spec['records'][protocol['phase_schedule'][i][0]]
                    assert row['scene']==record['scene'] and row['views']==record['views'] and row['image_hw']==record['image_hw']
                    assert row['slot_kind']==spec['slots'][i]['kind'] and row['step']==start+i+1
                assert history[-1]['images_seen']==1800 and history[-1]['patch_tokens_seen']==3540384
            value=dict(baseline=json.loads((BASE/'results.json').read_text())['baseline'],starting=starting,arms=results,
                config=protocol['config'],protocol_sha256=fingerprint(output/'protocol.json'),production_promoted=False,cache_stats=cache_stats)
            write_json(output/'results.json',value)
            write_json(output/'delivery_verification.json',dict(passed=True,checks=checks,phase_steps_per_arm=800,total_gpu_updates=1600,
                same_sampling_and_budget=True,source_code_checkpoints_unchanged=True,optimizer_reset_at_800_only=True))
            status('completed',cumulative_steps_per_arm=end);print('COMPLETED',flush=True)
        except BaseException as exc:
            status('failed',error=repr(exc));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--from-step',type=int,required=True);parser.add_argument('--to-step',type=int,required=True)
    args=parser.parse_args();run(args.from_step,args.to_step)
