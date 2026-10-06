"""Default-OFF/zero-ON parity and actual-train gradient probe; no optimization."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from run_depth_pose_ablation import ROOT,torch,VGGTOmega,tensor_state,state_hash,fingerprint,write_json,instance_targets_from_batch
from train_cylinder_adaptive_temperature import TemperatureModel
from train_cylinder_view_balance import oriented_batch,scene_digest
from mousegeoforge.adaptive_temperature import restore_adaptive
from mousegeoforge.query_position import install_query_position,query_position_context
from training.instance_pose_losses import instance_pose_loss,InstancePoseLossWeights

OUT=ROOT/'output/query_position_preflight_v2'
PARENT=ROOT/'output/cylinder_adaptive_temperature_v1'
CHECKPOINT=PARENT/'adaptive_temperature/final_heads.pt'


def run():
    started=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'output/cylinder_training.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (OUT/'protocol.json').exists():raise RuntimeError('Preserve existing preflight')
        def status(state,**kw):write_json(OUT/'status.json',dict(state=state,pid=os.getpid(),elapsed_s=time.monotonic()-started,**kw))
        status('preparing')
        try:
            spec=json.loads((PARENT/'protocol.json').read_text())
            probe=ROOT/'output/query_position_gradient_probe_v2/report.json'
            assert json.loads(probe.read_text())['passed_probe']
            inputs={str(p):fingerprint(p) for p in (PARENT/'protocol.json',CHECKPOINT,probe)}
            for p,h in spec['code_sha256'].items():assert fingerprint(ROOT/p)==h
            assert all(scene_digest(ROOT/p)==h for p,h in spec['source_sha256'].items())
            code={p:fingerprint(ROOT/p) for p in ('tools/preflight_query_position_v2.py','mousegeoforge/query_position.py','tests/test_query_position.py')}
            trainkey=next(s[0] for s in spec['arms']['adaptive_temperature'] if spec['records'][s[0]]['expected_instances']==2)
            write_json(OUT/'protocol.json',dict(evaluation=spec['evaluation'],train_probe_key=trainkey,
                source_sha256=spec['source_sha256'],input_sha256=inputs,code_sha256=code,
                mechanism='Add independent zero-initialized learned6x256 query-position vectors to cross-attention Q at each of2 decoder layers.3072 parameters. Not anchor coordinates, not forced spatial separation; original memory/all queries remain.',
                scope='Same original seed17 starting checkpoint, not failed diversity continuation.88 frozen evaluations OFF/zero ON, then one actual train-input forward/backward OFF/ON. No optimizer or data change.',
                decision='Require exact original/zero-ON predictions, finite nonzero new gradients, common-gradient relative L2 difference <=1e-6, cleanup and unchanged weights. Gradient criterion replaces impossible bitwise backward test, motivated before this run by OFF/OFF relative difference6.74e-8 vs OFF/ON5.73e-8 in independently preserved probe_v2. Original strict preflight stays failed; no forward tolerance changed.',
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
            enc_hash=state_hash({n:p.cpu() for n,p in model.aggregator.state_dict().items()})
            rng=torch.random.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone()
            install_query_position(model)
            assert torch.equal(rng,torch.random.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
            rows=[]
            for i,e in enumerate(spec['evaluation']):
                if time.monotonic()-started>1200:raise RuntimeError('20minute cap')
                r=spec['records'][e['key']];batch=oriented_batch(r,num_views=len(r['views']));images=batch['images'].cuda()
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):cached,start=model.aggregator(images)
                with torch.no_grad():
                    off=model.forward_cached(images,cached,start,coupling='off')
                    with query_position_context(model,enabled=True):on=model.forward_cached(images,cached,start,coupling='off')
                for k,v in off.items():torch.testing.assert_close(on[k],v,atol=0,rtol=0)
                saved=torch.load(PARENT/'adaptive_temperature'/f'prediction_{i:03d}.pt',map_location='cpu',weights_only=True)
                for k,v in saved['predictions'].items():torch.testing.assert_close(off[k].cpu(),v,atol=0,rtol=0)
                rows.append(dict(index=i,views=len(r['views']),image_hw=r['image_hw'],max_difference=0))
                del off,on,batch,images,cached
                if (i+1)%16==0:status('zero_parity',records=i+1,total=88);print('PARITY',i+1,flush=True)
            # The actual training probe is train-only and does not change any weights.
            r=spec['records'][trainkey];batch=oriented_batch(r,num_views=len(r['views']));images=batch['images'].cuda()
            target={k:v.cuda() for k,v in instance_targets_from_batch(batch).items()}
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):cached,start=model.aggregator(images)
            model.train();lossweights=InstancePoseLossWeights(keypoint_3d=0.,reprojection=0.,local_heatmap=0.,keypoint_uncertainty=0.)
            model.zero_grad(set_to_none=True);off=model.forward_cached(images,cached,start,coupling='off')
            loss,_=instance_pose_loss(off,target,lossweights,image_size=images.shape[-2:]);loss.backward()
            loss_off=float(loss.detach());common={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.grad is not None}
            pred_off={k:v.detach().cpu() for k,v in off.items()};del off,loss
            assert 'instance_pose_head.query_position.offsets' not in common
            model.zero_grad(set_to_none=True)
            with query_position_context(model,enabled=True):
                on=model.forward_cached(images,cached,start,coupling='off')
                for k,v in pred_off.items():torch.testing.assert_close(on[k].detach().cpu(),v,atol=0,rtol=0)
                loss,_=instance_pose_loss(on,target,lossweights,image_size=images.shape[-2:]);loss.backward()
            assert loss_off==float(loss.detach())
            delta_sq=0.;baseline_sq=0.;maximum=0.
            for n,p in model.named_parameters():
                if n in common:
                    assert p.grad is not None
                    current=p.grad.detach().cpu();assert torch.isfinite(current).all()
                    difference=(current-common[n]).double()
                    maximum=max(maximum,float(difference.abs().max()))
                    delta_sq+=float(difference.square().sum());baseline_sq+=float(common[n].double().square().sum())
                elif p.grad is not None:assert n=='instance_pose_head.query_position.offsets'
            relative_gradient_difference=(delta_sq/max(baseline_sq,1e-30))**.5
            assert relative_gradient_difference<=1e-6
            grad=model.instance_pose_head.query_position.offsets.grad
            assert grad is not None and torch.isfinite(grad).all() and grad.norm()>0
            layer_norms=grad.flatten(1).norm(dim=1).tolist();assert all(x>0 for x in layer_norms)
            model.zero_grad(set_to_none=True);model.eval()
            assert flags=={n:p.requires_grad for n,p in model.named_parameters() if n in flags}
            assert all(len(x.multihead_attn._forward_pre_hooks)==0 for x in model.instance_pose_head.instance_decoder.layers)
            assert state_hash({k:v for k,v in tensor_state(model).items() if not k.startswith('instance_pose_head.query_position.')})==state_hash(original)
            assert torch.count_nonzero(model.instance_pose_head.query_position.offsets)==0
            assert enc_hash==state_hash({n:p.cpu() for n,p in model.aggregator.state_dict().items()})
            assert all(fingerprint(Path(p))==h for p,h in inputs.items())
            assert all(fingerprint(ROOT/p)==h for p,h in code.items())
            report=dict(passed=True,evaluation_records=88,original_saved_predictions_max_difference=0,
                zero_on_all_outputs_max_difference=0,actual_train_forward_max_difference=0,
                common_gradients_max_difference=maximum,common_gradients_relative_l2=relative_gradient_difference,
                original_strict_backward_preflight_passed=False,gradient_relative_l2_limit=1e-6,new_gradient_norm_per_layer=layer_norms,actual_pose_loss=loss_off,
                new_parameters=3072,initialization_cpu_cuda_rng_unchanged=True,original_weights_encoder_flags_unchanged=True,
                hooks_cleaned=True,encoder_forwards=89,head_forwards=178,optimizer_steps=0,rows=rows,
                input_sha256=inputs,code_sha256=code,production_promoted=False,
                limitation='Zero parity and one train-input gradient probe are not evidence of learned gains or correct animal identity.')
            write_json(OUT/'report.json',report);status('completed');print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2),flush=True)
        except BaseException as error:status('failed',error=repr(error));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--launch',action='store_true');args=parser.parse_args()
    if args.launch:
        if (OUT/'status.json').exists():raise RuntimeError('Already launched')
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/'run.log').open('a') as log:
            p=subprocess.Popen([sys.executable,'-u',__file__],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(pid=p.pid,output=str(OUT))))
    else:run()


