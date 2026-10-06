"""Frozen actual-data soft-spatial-prior audit; OFF exact, ON intentionally nonzero."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from run_depth_pose_ablation import ROOT, torch, VGGTOmega, tensor_state, state_hash, fingerprint, write_json, instance_targets_from_batch
from train_cylinder_adaptive_temperature import TemperatureModel
from train_cylinder_view_balance import oriented_batch, scene_digest, physical_metrics
from train_variable_view_replay import mean_metrics
from mousegeoforge.adaptive_temperature import restore_adaptive
from mousegeoforge.spatial_query_prior import install_spatial_query_prior, spatial_query_context
from training.instance_pose_losses import instance_pose_loss, InstancePoseLossWeights

OUT=ROOT/'output/spatial_query_prior_preflight_v2'
PARENT=ROOT/'output/cylinder_adaptive_temperature_v1'
CHECKPOINT=PARENT/'adaptive_temperature/final_heads.pt'
DEADLINE=1789194663


def run():
    start_time=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'output/cylinder_training.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (OUT/'protocol.json').exists():raise RuntimeError('Preserve prior run')
        def status(state,**kw):write_json(OUT/'status.json',dict(state=state,pid=os.getpid(),elapsed_s=time.monotonic()-start_time,**kw))
        def budget():
            if time.monotonic()-start_time>1200 or time.time()>=DEADLINE:raise RuntimeError('Time cap')
        status('preparing')
        try:
            spec=json.loads((PARENT/'protocol.json').read_text())
            for p,h in spec['code_sha256'].items():assert fingerprint(ROOT/p)==h
            for p,h in spec['source_sha256'].items():assert scene_digest(ROOT/p)==h
            code={p:fingerprint(ROOT/p) for p in ('mousegeoforge/spatial_query_prior.py','tests/test_spatial_query_prior.py','tools/preflight_spatial_query_prior_v2.py')}
            inputs={str(p):fingerprint(p) for p in (PARENT/'protocol.json',CHECKPOINT)}
            train_keys=[]
            for views in (1,2,4):
                key=next(x[0] for x in spec['arms']['adaptive_temperature']
                         if len(spec['records'][x[0]]['views'])==views and
                         (views!=2 or spec['records'][x[0]]['expected_instances']==2))
                train_keys.append(key)
            write_json(OUT/'protocol.json',dict(evaluation=spec['evaluation'],train_keys=train_keys,
                input_sha256=inputs,code_sha256=code,source_sha256=spec['source_sha256'],
                scope='OFF exact saved prediction replay88; ON initial nonzero Gaussian bias88 with exact depth;3 train-only gradient probes V1,2,4; zero optimization.',
                mechanism='SMCA-inspired conditional Gaussian log-attention prior at two instance cross-attention layers.2108 parameters; movable grid-initialized centers, sigma in(.08,1), strength initialized.1 bounded<=2, finite bias>=-20. No hard partition, no inter-query repulsion. Shared conditional weight initiallyzero; first layer anchors not image-conditioned until query contains image features. Not fullSMCA/DAB.',
                acceptance='OFF output tolerance0; ON need not equal OFF because initial prior is nonzero. Every ON output finite, ON depth/depth_conf exactOFF; all new tensor gradients finite nonzero per trainprobe; weights/RNG initialization/flags preserved; hooks cleaned. No learned-effect or old-gradient-parity claim.',
                optimizer_steps=0,production_promoted=False))
            torch.set_num_threads(2);torch.manual_seed(17)
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
            encoder=ROOT/'vggt-omega/vggt_omega_1b_512.pt';assert fingerprint(encoder)==spec['encoder_sha256']
            spatial=VGGTOmega(enable_camera=False,enable_depth=True,enable_multi_instance_pose=True,num_keypoints=23)
            weights=torch.load(encoder,weights_only=True,mmap=True)
            loaded=spatial.load_state_dict({k:v for k,v in weights.items() if k.startswith(('aggregator.','dense_head.'))},strict=False)
            assert not loaded.unexpected_keys and all(k.startswith('instance_pose_head.') for k in loaded.missing_keys)
            del weights
            model=TemperatureModel.from_spatial_model(spatial).cuda().eval();del spatial
            original=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)['model'];restore_adaptive(model,original)
            flags={n:p.requires_grad for n,p in model.named_parameters()}
            encoder_hash=state_hash({k:v.cpu() for k,v in model.aggregator.state_dict().items()})
            cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone()
            install_spatial_query_prior(model)
            assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
            new_initial=state_hash({k:v.detach().cpu() for k,v in model.instance_pose_head.spatial_query_prior.state_dict().items()})
            assert sum(p.numel() for p in model.instance_pose_head.spatial_query_prior.parameters())==2108
            rows={'off':[],'on_initial':[]};max_change=0.
            for index,e in enumerate(spec['evaluation']):
                budget();r=spec['records'][e['key']];batch=oriented_batch(r,num_views=len(r['views']));images=batch['images'].cuda()
                target={k:v.cuda() for k,v in instance_targets_from_batch(batch).items()}
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):cache,patch_start=model.aggregator(images)
                with torch.no_grad():
                    off=model.forward_cached(images,cache,patch_start,coupling='off')
                    with spatial_query_context(model,enabled=True):on=model.forward_cached(images,cache,patch_start,coupling='off')
                saved=torch.load(PARENT/'adaptive_temperature'/f'prediction_{index:03d}.pt',weights_only=True,map_location='cpu')
                for key,value in saved['predictions'].items():torch.testing.assert_close(off[key].cpu(),value,atol=0,rtol=0)
                for key,value in on.items():
                    assert torch.isfinite(value).all(),key
                    if key in ('depth','depth_conf'):torch.testing.assert_close(value,off[key],atol=0,rtol=0)
                max_change=max(max_change,float((on['instance_boxes_cxcywh']-off['instance_boxes_cxcywh']).abs().max()))
                for tag,pred in (('off',off),('on_initial',on)):
                    metrics,assignment=physical_metrics(pred,batch,target)
                    rows[tag].append(dict(group=e['group'],scene=r['scene'],views=r['views'],metrics=metrics))
                    if index in (0,7,9,13):
                        write_json(OUT/tag/'marker.json',dict(frozen_initial=True))
                        torch.save(dict(record=r,oracle_assignment=assignment,
                            predictions={k:v.cpu() for k,v in pred.items() if k in ('instance_boxes_cxcywh','instance_objectness_logits','instance_keypoint_xy')},
                            targets={k:v.cpu() for k,v in target.items() if k!='masks'},mask_logits=pred['instance_mask_logits'].cpu()),
                            OUT/tag/f'visual_{index:03d}.pt')
                del off,on,cache,images,batch,target
                if (index+1)%16==0:status('frozen_eval',records=index+1);print('EVAL',index+1,flush=True)
            gradients=[]
            for key in train_keys:
                budget();r=spec['records'][key];batch=oriented_batch(r,num_views=len(r['views']));images=batch['images'].cuda()
                target={k:v.cuda() for k,v in instance_targets_from_batch(batch).items()}
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):cache,patch_start=model.aggregator(images)
                model.train();model.zero_grad(set_to_none=True)
                with spatial_query_context(model,enabled=True):
                    pred=model.forward_cached(images,cache,patch_start,coupling='off')
                    loss,_=instance_pose_loss(pred,target,InstancePoseLossWeights(keypoint_3d=0.,reprojection=0.,local_heatmap=0.,keypoint_uncertainty=0.),image_size=images.shape[-2:])
                    assert torch.isfinite(loss);loss.backward()
                norms={}
                for name,param in model.instance_pose_head.spatial_query_prior.named_parameters():
                    assert param.grad is not None and torch.isfinite(param.grad).all()
                    norms[name]=float(param.grad.norm());assert norms[name]>0
                assert all(p.grad is None for p in model.aggregator.parameters())
                gradients.append(dict(key=key,views=len(r['views']),pose_loss=float(loss.detach()),new_gradient_norms=norms))
                model.zero_grad(set_to_none=True);model.eval();del pred,loss,target,batch,cache,images
            assert flags=={n:p.requires_grad for n,p in model.named_parameters() if n in flags}
            assert state_hash({k:v for k,v in tensor_state(model).items() if not k.startswith('instance_pose_head.spatial_query_prior.')})==state_hash(original)
            assert state_hash({k:v.detach().cpu() for k,v in model.instance_pose_head.spatial_query_prior.state_dict().items()})==new_initial
            assert encoder_hash==state_hash({k:v.cpu() for k,v in model.aggregator.state_dict().items()})
            assert not model.instance_pose_head._forward_pre_hooks
            assert all(not x.multihead_attn._forward_pre_hooks for x in model.instance_pose_head.instance_decoder.layers)
            assert all(fingerprint(ROOT/p)==h for p,h in code.items())
            assert all(fingerprint(__import__('pathlib').Path(p))==h for p,h in inputs.items())
            report=dict(passed=True,original_saved_prediction_max_difference=0,depth_output_max_difference=0,
                initial_on_equals_off=False,initial_box_max_change=max_change,optimizer_steps=0,
                new_parameters=2108,new_gradients=gradients,initialization_rng_unchanged=True,
                weights_encoder_flags_unchanged=True,hooks_cleaned=True,evaluation_records=88,
                evaluations={tag:dict(records=r,summary=mean_metrics(r)) for tag,r in rows.items()},
                code_sha256=code,input_sha256=inputs,production_promoted=False)
            write_json(OUT/'report.json',report);status('completed');print('PASSED',flush=True)
        except BaseException as error:status('failed',error=repr(error));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--launch',action='store_true');args=parser.parse_args()
    if args.launch:
        if (OUT/'status.json').exists():raise RuntimeError('Already launched')
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/'run.log').open('a') as log:
            proc=subprocess.Popen([sys.executable,'-u',__file__],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(pid=proc.pid,output=str(OUT))))
    else:run()


