"""Frozen scale-sigma heatmap representation: global vs instance-conditioned gate pilot."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import copy,fcntl,json,time
import cv2,numpy as np,torch
from torch.nn import functional as F
from run_depth_pose_ablation import ROOT,VGGTOmega,DepthPoseModel,write_json,fingerprint,instance_targets_from_batch
from train_cylinder_view_balance import oriented_batch
from train_spatial_anchor_diverse32 import loss_fn
from mousegeoforge.adaptive_temperature import install_adaptive_temperature,restore_adaptive
from mousegeoforge.spatial_anchor_experiment import install_spatial_anchor,spatial_anchor_context
from mousegeoforge.spatial_heatmap_probe import spatial_expectation
from mousegeoforge.visible_heatmap_supervision import anchor_scores
from mousegeoforge.deterministic_local_sampling import deterministic_local_refine
from mousegeoforge.instance_adaptive_gate import predicted_quality
from mousegeoforge.instance_coordinate_residual import InstanceCoordinateResidual,reference_from_shape
from mousegeoforge.mask_box_refiner import features_from_logits
from mousegeoforge.affine_batch_experiment import augment_batch
from mousegeoforge.per_view_affine_batch import augment_per_view
from mousegeoforge.pilot_metrics import evaluate_scene
from mousegeoforge.mask_box_candidate import tensor_mapping_sha256

OUT=ROOT/'output/wide_position_residual_v1'
PARENT=ROOT/'output/spatial_anchor_scale_heatmap_v1'
PLAN=ROOT/'output/spatial_anchor_affine_batch_preflight_v1_retry1/plan.json'
STEPS=400


def main():
    torch.set_num_threads(2);cv2.setNumThreads(1);torch.manual_seed(17)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with (ROOT/'output/cylinder_training.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);OUT.mkdir(exist_ok=False)
        began=time.monotonic();handles=[]
        def status(state,**kw):write_json(OUT/'status.json',dict(state=state,pid=os.getpid(),elapsed_s=time.monotonic()-began,**kw))
        def cutoff():
            if time.monotonic()-began>1800 or time.time()>1789233567:raise TimeoutError('Pilot cutoff')
        status('preparing')
        try:
            assert json.loads((PARENT/'verification.json').read_text())['passed']
            parent=json.loads((PARENT/'protocol.json').read_text());plan=json.loads(PLAN.read_text())
            wide_path=ROOT/'output/wide_position_library_v1/plan.json';wide_plan=json.loads(wide_path.read_text())
            assert json.loads((wide_path.parent/'verification.json').read_text())['passed']
            assert wide_plan['parent_protocol_sha256']==fingerprint(PARENT/'protocol.json')
            assert all(fingerprint(ROOT/p)==h for p,h in wide_plan['code_sha256'].items())
            assert all(fingerprint(ROOT/p)==s for p,s in parent['code_sha256'].items())
            assert fingerprint(PLAN)==parent['augmentation_plan_sha256']
            base=ROOT/'output/cylinder_adaptive_temperature_v1/adaptive_temperature/final_heads.pt';encoder=ROOT/'vggt-omega/vggt_omega_1b_512.pt'
            frozen_branch=PARENT/'scale_sigma_step800.pt'
            assert fingerprint(base)==parent['base_sha256'] and fingerprint(encoder)==parent['encoder_sha256']
            paths=[base,encoder,frozen_branch,PLAN,PARENT/'protocol.json',wide_path,ROOT/'output/instance_coordinate_residual_v1/relative_step400.pt']
            hashes={str(p.relative_to(ROOT)):fingerprint(p) for p in paths}
            code=['tools/train_wide_position_residual.py','mousegeoforge/instance_adaptive_gate.py','mousegeoforge/instance_coordinate_residual.py','mousegeoforge/mask_box_refiner.py','mousegeoforge/mask_shape_audit.py','mousegeoforge/deterministic_local_sampling.py','mousegeoforge/affine_batch_experiment.py','mousegeoforge/per_view_affine_batch.py']
            schedule=json.loads((PARENT/'augmented_schedule.json').read_text())[:STEPS]
            protocol=dict(seed=17,steps_per_arm=STEPS,extra_global_replay_steps=0,arms=['narrow','wide'],
                hashes=hashes,code_sha256={p:fingerprint(ROOT/p) for p in code},parent=str(PARENT.relative_to(ROOT)),
                train_keys=parent['train_keys'],evaluation=parent['evaluation'],records=parent['records'],schedule=schedule,
                lr=.001,weight_decay=.01,clip_grad=1.,loss='Original all-valid coarse/refined coordinate loss; heatmap projection frozen so no auxiliary loss optimization',
                input='Detached normalized keypoint tokens, base coarse/spatial xy in global or instance coordinates, same five predicted quality features. Largest predicted mask component reference; no GT camera role/count/visibility.',
                comparison='Both use same8578-parameter mask-relative residual. Narrow vs wider per-view position library, same scenes/scale choices/variant schedule/400 steps; wide includes independently placed views and changed padding context.',
                gate='Zero output MLP exactly preserves frozen mixed coarse. Add extent*2*tanh(delta), clamp coordinates0..1. Global reference center.5 extent1; relative from predicted component, empty fallback global. No production install.',
                stop='400 fixed,100/200/400 descriptive,narrow must exactly replay prior relative400 before wide; max30min/user cutoff',real_training=False,heldout_accessed=False)
            write_json(OUT/'protocol.json',protocol)
            spatial=VGGTOmega(enable_camera=False,enable_depth=True,enable_multi_instance_pose=True,num_keypoints=23)
            weights=torch.load(encoder,map_location='cpu',weights_only=True,mmap=True)
            loaded=spatial.load_state_dict({k:v for k,v in weights.items() if k.startswith(('aggregator.','dense_head.'))},strict=False)
            assert not loaded.unexpected_keys and all(k.startswith('instance_pose_head.') for k in loaded.missing_keys)
            model=install_adaptive_temperature(DepthPoseModel.from_spatial_model(spatial)).cuda().eval()
            restore_adaptive(model,torch.load(base,map_location='cpu',weights_only=True)['model']);del spatial,weights
            install_spatial_anchor(model);head=model.instance_pose_head;branch=head.spatial_anchor
            branch.load_state_dict(torch.load(frozen_branch,map_location='cpu',weights_only=True)['branch']);model.eval().requires_grad_(False)
            oldhash=tensor_mapping_sha256({k:v.reshape(-1) for k,v in model.state_dict().items()});g0=branch.gate.tanh().detach()
            captured={};replay=[]
            def capture(name,inputs=False):
                def hook(m,a,value):
                    captured[name]=value.detach()
                    if inputs:captured['tokens']=a[0].detach()
                return hook
            handles=[head.pyramid_refine[0].register_forward_hook(capture('pixels')),head.coarse_keypoint_xy.register_forward_hook(capture('raw',True)),
                     head.local_pixel_projection.register_forward_hook(capture('local_pixel')),head.local_query_projection.register_forward_hook(capture('local_query'))]
            def sample(record,variant=None,affines=None):
                cutoff();batch=oriented_batch(record,num_views=len(record['views']))
                if variant is not None:batch,_=augment_batch(batch,**{k:v for k,v in variant.items() if k!='name'})
                if affines is not None:batch,_=augment_per_view(batch,affines)
                x=batch['images'].cuda()
                with torch.no_grad():
                    with torch.autocast('cuda',dtype=torch.bfloat16):cached,start=model.aggregator(x)
                    old=model.forward_cached(x,cached,start,coupling='off');s=dict(captured)
                    scores=anchor_scores(branch,s['tokens'],s['pixels'],old['instance_mask_logits'][0])
                    anchor=spatial_expectation(s['raw'],scores,enabled=True).clamp(1e-5,1-1e-5)
                    quality=predicted_quality(scores,old['instance_keypoint_visibility_logits'][0],old['instance_mask_logits'][0],old['instance_objectness_logits'][0])
                    s.update(anchor=anchor,quality=quality,local_query=F.normalize(s['local_query'],dim=-1,eps=1e-6),scale=head.last_temperature_scale.detach(),size=head.local_heatmap_size,radius=head.local_window_radius)
                    with spatial_anchor_context(model,enabled=True):ref=model.forward_cached(x,cached,start,coupling='off')
                    ref={k:v.cpu() for k,v in ref.items()};target=instance_targets_from_batch(batch);metric,matches=evaluate_scene(ref,batch,target)
                    s.update(batch=batch,target=target,matches=matches,reference=ref,metrics=metric,base_coarse=ref['instance_keypoint_coarse_xy'][0].cuda(),
                             reference_geometry=reference_from_shape(features_from_logits(old['instance_mask_logits'][0],batch['images'].shape[-2:])).cuda())
                    # Test relative zero initialization on actual frozen evidence.
                    with torch.random.fork_rng(devices=[]):zero=InstanceCoordinateResidual().cuda()
                    coarse=zero(s['base_coarse'],s['anchor'],s['tokens'],s['quality'],s['reference_geometry'])
                    final=deterministic_local_refine(s,coarse)
                    torch.testing.assert_close(coarse.cpu(),ref['instance_keypoint_coarse_xy'][0],atol=0,rtol=0)
                    torch.testing.assert_close(final.cpu(),ref['instance_keypoint_xy'][0],atol=1e-6,rtol=1e-6)
                    replay.append(float((final.cpu()-ref['instance_keypoint_xy'][0]).abs().max()))
                    del s['pixels'];captured.clear();return s
            cache={}
            for key in parent['train_keys']:cache[key]=sample(parent['records'][key]);status('caching_identity',records=len(cache),total=32)
            for row in plan['records']:
                if row['accepted'] and row['variant']['name']!='identity':
                    key=row['key'];cache[key+':'+row['variant']['name']]=sample(parent['records'][key],row['variant']);status('caching_augmented',records=len(cache),total=160)
            for row in wide_plan['records']:
                if row['variant']['name']!='identity':
                    key=row['key'];cache['wide:'+key+':'+row['variant']['name']]=sample(parent['records'][key],affines=row['affines']);status('caching_wide',records=len(cache),total=288)
            wide_lookup={k:k if k in parent['train_keys'] else 'wide:'+k for k in schedule}
            dev={e['key']:sample(parent['records'][e['key']]) for e in parent['evaluation']}
            for h in handles:h.remove()
            handles=[];write_json(OUT/'preflight.json',dict(passed=True,real_checkpoint_samples=len(replay),zero_coarse_exact=True,final_max_difference=max(replay)))
            def output(gate,s):
                coarse=gate(s['base_coarse'],s['anchor'],s['tokens'],s['quality'],s['reference_geometry'])
                return coarse,deterministic_local_refine(s,coarse),coarse-s['base_coarse']
            def evaluate(gate):
                rows=[]
                with torch.no_grad():
                    for e in parent['evaluation']:
                        s=dev[e['key']];coarse,refined,g=output(gate,s)
                        pred=dict(s['reference'],instance_keypoint_xy=refined[None].cpu(),instance_keypoint_coarse_xy=coarse[None].cpu())
                        metric,matches=evaluate_scene(pred,s['batch'],s['target']);assert matches==s['matches']
                        rows.append(dict(group=e['group'],key=e['key'],metrics=metric,residual_abs_mean=float(g.abs().mean())))
                return rows
            def fixed_loss(gate):
                with torch.no_grad():return float(np.mean([float(loss_fn(*reversed(output(gate,cache[key])[:2]),cache[key])) for key in parent['train_keys']]))
            results={}
            for arm in ('narrow','wide'):
                torch.manual_seed(17);gate=InstanceCoordinateResidual(relative=True).cuda();opt=torch.optim.AdamW(gate.parameters(),lr=.001,weight_decay=.01)
                initial=fixed_loss(gate);history=[]
                for step,key in enumerate(schedule,1):
                    cutoff();opt.zero_grad(set_to_none=True);s=cache[wide_lookup[key] if arm=='wide' else key];coarse,refined,g=output(gate,s);loss=loss_fn(refined,coarse,s)
                    assert torch.isfinite(loss);loss.backward();grad=torch.nn.utils.clip_grad_norm_(gate.parameters(),1.);assert torch.isfinite(grad)
                    opt.step();history.append(dict(step=step,loss=float(loss.detach()),grad_norm=float(grad),residual_abs_mean=float(g.detach().abs().mean())))
                    if step%25==0:status('training',arm=arm,step=step,total=STEPS);print(arm,step,flush=True)
                    if step in (100,200,400):
                        write_json(OUT/f'{arm}_eval_{step}.json',evaluate(gate))
                        torch.save(dict(state={k:v.cpu() for k,v in gate.state_dict().items()},optimizer=opt.state_dict(),relative=True,step=step),OUT/f'{arm}_step{step}.pt')
                results[arm]=dict(initial_fixed_loss=initial,final_fixed_loss=fixed_loss(gate),parameters=sum(p.numel() for p in gate.parameters()),records=evaluate(gate))
                write_json(OUT/f'{arm}_history.json',history)
                if arm=='narrow':
                    ref=torch.load(ROOT/'output/instance_coordinate_residual_v1/relative_step400.pt',map_location='cpu',weights_only=True)
                    for k,v in gate.state_dict().items():torch.testing.assert_close(v.cpu(),ref['state'][k],atol=0,rtol=0)
                    for i,state in opt.state_dict()['state'].items():
                        for k,v in state.items():torch.testing.assert_close(v.cpu(),ref['optimizer']['state'][i][k].cpu(),atol=0,rtol=0)
                    write_json(OUT/'narrow_replay.json',dict(passed=True,weights_and_optimizer_exact=True))
            assert tensor_mapping_sha256({k:v.reshape(-1) for k,v in model.state_dict().items()})==oldhash
            assert all(p.grad is None for p in model.parameters())
            assert all(fingerprint(ROOT/p)==s for p,s in {**hashes,**protocol['code_sha256']}.items())
            write_json(OUT/'results.json',dict(arms=results,baseline=[dict(group=e['group'],key=e['key'],metrics=dev[e['key']]['metrics']) for e in parent['evaluation']]))
            write_json(OUT/'verification.json',dict(passed=True,steps_per_arm=STEPS,extra_replay_steps=0,old_model_and_heatmap_branch_unchanged=True,
                old_gradients_none=True,training_variants=len(cache),dev_records=len(dev),production_promoted=False))
            status('completed');print('COMPLETED',flush=True)
        except BaseException as e:status('failed',error=repr(e));raise
        finally:
            for h in handles:h.remove()


if __name__=='__main__':main()






