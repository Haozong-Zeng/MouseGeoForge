"""Zero-optimization body/tail gradient audit on the locked training schedule."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import copy,fcntl,json,time
from collections import Counter
import cv2,numpy as np,torch
from torch.nn import functional as F
from run_depth_pose_ablation import ROOT,VGGTOmega,DepthPoseModel,write_json,fingerprint,instance_targets_from_batch
from mousegeoforge.body_tail_loss_diagnostic import partition_coordinate_loss
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
from mousegeoforge.mask_conflict_assignment import repair_mask_conflicts
from mousegeoforge.pilot_metrics import evaluate_scene
from mousegeoforge.mask_box_candidate import tensor_mapping_sha256

OUT=ROOT/'output/body_tail_gradient_diagnostic_v1'
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
            paths=[base,encoder,frozen_branch,PLAN,PARENT/'protocol.json',wide_path,ROOT/'output/mask_repaired_position_residual_v1/repaired_step400.pt']
            hashes={str(p.relative_to(ROOT)):fingerprint(p) for p in paths}
            code=['tools/audit_body_tail_gradients.py','mousegeoforge/instance_adaptive_gate.py','mousegeoforge/instance_coordinate_residual.py','mousegeoforge/mask_box_refiner.py','mousegeoforge/mask_shape_audit.py','mousegeoforge/deterministic_local_sampling.py','mousegeoforge/affine_batch_experiment.py','mousegeoforge/per_view_affine_batch.py','mousegeoforge/mask_conflict_assignment.py']

            SOURCE=ROOT/'output/mixed_position_residual_v1'
            source_protocol=json.loads((SOURCE/'protocol.json').read_text())
            assert json.loads((SOURCE/'verification.json').read_text())['passed']
            assert all(fingerprint(ROOT/p)==h for p,h in {**source_protocol['hashes'],**source_protocol['code_sha256']}.items())
            assert fingerprint(SOURCE/'selection.json')==source_protocol['selection_sha256']
            selection=json.loads((SOURCE/'selection.json').read_text())['rows']
            counts=Counter(('wide:'+r['key']) if r['use_wide'] else r['key'] for r in selection)
            needed=set(counts)|set(parent['train_keys'])
            paths.extend([SOURCE/'protocol.json',SOURCE/'selection.json',SOURCE/'mixed_step400.pt'])
            hashes={str(p.relative_to(ROOT)):fingerprint(p) for p in paths}
            code.append('mousegeoforge/body_tail_loss_diagnostic.py')
            protocol=dict(training_steps=0,states=['initial','mixed400'],selection_sha256=fingerprint(SOURCE/'selection.json'),
                hashes=hashes,code_sha256={p:fingerprint(ROOT/p) for p in code},unique_requested=len(needed),
                matching='Frozen repaired training matches; no development or real images.',
                weighting='Per-instance all-valid coordinate denominator, then mean instances. Body+tail reconstruct original loss and parameter gradients.',
                summaries='Original32 uniform and actual400 schedule occurrence weighted; aggregate cosine and per-scene negative cosine rate both reported.',
                warning='Local unpreconditioned gradient conflict is diagnostic, not proof of generalization causality or actual AdamW update.',
                real_training=False,heldout_accessed=False,production_promoted=False)
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
                    repaired_matches,repair_info=repair_mask_conflicts(ref['instance_mask_logits'],target['masks'],target['instance_valid'],target['mask_valid'],matches)
                    s.update(repaired_matches=repaired_matches,repair_info=repair_info)
                    # Test relative zero initialization on actual frozen evidence.
                    with torch.random.fork_rng(devices=[]):zero=InstanceCoordinateResidual().cuda()
                    coarse=zero(s['base_coarse'],s['anchor'],s['tokens'],s['quality'],s['reference_geometry'])
                    final=deterministic_local_refine(s,coarse)
                    torch.testing.assert_close(coarse.cpu(),ref['instance_keypoint_coarse_xy'][0],atol=0,rtol=0)
                    torch.testing.assert_close(final.cpu(),ref['instance_keypoint_xy'][0],atol=1e-6,rtol=1e-6)
                    replay.append(float((final.cpu()-ref['instance_keypoint_xy'][0]).abs().max()))
                    del s['pixels'];captured.clear();return s

            cache={};meta={}
            for key in parent['train_keys']:
                cache[key]=sample(parent['records'][key]);meta[key]=(key,'identity')
                status('caching',records=len(cache),total=len(needed))
            for row in plan['records']:
                key=row['key']+':'+row['variant']['name']
                if key in needed:
                    assert row['accepted'];cache[key]=sample(parent['records'][row['key']],row['variant'])
                    meta[key]=(row['key'],'narrow');status('caching',records=len(cache),total=len(needed))
            for row in wide_plan['records']:
                key='wide:'+row['key']+':'+row['variant']['name']
                if key in needed:
                    cache[key]=sample(parent['records'][row['key']],affines=row['affines'])
                    meta[key]=(row['key'],'wide');status('caching',records=len(cache),total=len(needed))
            assert set(cache)==needed
            for h in handles:h.remove()
            handles=[]
            rows=[];aggregates={};max_loss_error=0.;max_gradient_error=0.;gradient_records={}
            def update_summary(state,label,weight,gb,gt,row):
                key=state+'/'+label
                if key not in aggregates:aggregates[key]=dict(weight=0.,body=np.zeros(gb.shape,dtype=np.float64),tail=np.zeros(gt.shape,dtype=np.float64),negative=0.,defined=0.,body_norm=0.,tail_norm=0.)
                a=aggregates[key];a['weight']+=weight;a['body']+=weight*gb;a['tail']+=weight*gt
                a['body_norm']+=weight*row['body_norm'];a['tail_norm']+=weight*row['tail_norm']
                if row['cosine'] is not None:a['defined']+=weight;a['negative']+=weight*(row['cosine']<0)
            for state in ('initial','mixed400'):
                torch.manual_seed(17);gate=InstanceCoordinateResidual(relative=True).cuda().eval()
                if state=='mixed400':gate.load_state_dict(torch.load(SOURCE/'mixed_step400.pt',map_location='cpu',weights_only=True)['state'])
                gatehash=tensor_mapping_sha256(gate.state_dict());parameters=tuple(gate.parameters())
                for index,(key,rawsample) in enumerate(cache.items()):
                    cutoff();s=dict(rawsample,matches=rawsample['repaired_matches'])
                    coarse=gate(s['base_coarse'],s['anchor'],s['tokens'],s['quality'],s['reference_geometry'])
                    refined=deterministic_local_refine(s,coarse)
                    parts=partition_coordinate_loss(refined,coarse,s);total=loss_fn(refined,coarse,s)
                    torch.testing.assert_close(parts['body']+parts['tail'],total,atol=1e-7,rtol=1e-6)
                    gradients={}
                    for name,loss in [*parts.items(),('total',total)]:
                        values=torch.autograd.grad(loss,parameters,retain_graph=name!='total')
                        gradients[name]=torch.cat([g.reshape(-1) for g in values])
                    combined=gradients['body']+gradients['tail']
                    torch.testing.assert_close(combined,gradients['total'],atol=1e-6,rtol=1e-5)
                    max_loss_error=max(max_loss_error,float((parts['body']+parts['tail']-total).detach().abs()))
                    max_gradient_error=max(max_gradient_error,float((combined-gradients['total']).abs().max()))
                    gb=gradients['body'].cpu().double().numpy();gt=gradients['tail'].cpu().double().numpy()
                    bn=float(np.linalg.norm(gb));tn=float(np.linalg.norm(gt));dot=float(gb@gt)
                    source,variant=meta[key];record=parent['records'][source]
                    instance_counts=sorted(set(len(ms) for qs,ms in s['matches']))
                    multiplicity='single' if max(instance_counts)==1 else 'multi'
                    row=dict(state=state,key=key,source_key=source,source_group=record['source_group'],variant=variant,multiplicity=multiplicity,
                        instance_counts=instance_counts,schedule_count=counts[key],body_loss=float(parts['body'].detach()),tail_loss=float(parts['tail'].detach()),
                        body_norm=bn,tail_norm=tn,dot=dot,cosine=dot/(bn*tn) if bn*tn>1e-15 else None,
                        total_gradient_body_dot=float(gb@(gb+gt)),total_gradient_tail_dot=float(gt@(gb+gt)))
                    rows.append(row);gradient_records[state+'/'+key]=dict(body=gradients['body'].cpu(),tail=gradients['tail'].cpu())
                    if key in parent['train_keys']:
                        for label in ['original32/all','original32/source/'+record['source_group'],'original32/'+multiplicity]:
                            update_summary(state,label,1,gb,gt,row)
                    if counts[key]:
                        for label in ['schedule/all','schedule/'+variant,'schedule/source/'+record['source_group'],'schedule/'+multiplicity,
                                      'schedule/'+variant+'/'+multiplicity]:
                            update_summary(state,label,counts[key],gb,gt,row)
                    if (index+1)%20==0:status('gradients',state_name=state,records=index+1,total=len(cache))
                assert tensor_mapping_sha256(gate.state_dict())==gatehash
                assert all(p.grad is None for p in gate.parameters())
            summary={}
            for key,a in aggregates.items():
                gb=a.pop('body')/a['weight'];gt=a.pop('tail')/a['weight'];bn=float(np.linalg.norm(gb));tn=float(np.linalg.norm(gt));dot=float(gb@gt)
                summary[key]=dict(weight=a['weight'],aggregate_body_norm=bn,aggregate_tail_norm=tn,aggregate_cosine=dot/(bn*tn) if bn*tn>1e-15 else None,
                    per_sample_negative_fraction=a['negative']/a['defined'] if a['defined'] else None,
                    mean_sample_body_norm=a['body_norm']/a['weight'],mean_sample_tail_norm=a['tail_norm']/a['weight'],
                    aggregate_total_dot_body=float(gb@(gb+gt)),aggregate_total_dot_tail=float(gt@(gb+gt)))
            assert tensor_mapping_sha256({k:v.reshape(-1) for k,v in model.state_dict().items()})==oldhash
            assert all(p.grad is None for p in model.parameters())
            assert all(fingerprint(ROOT/p)==h for p,h in {**hashes,**protocol['code_sha256']}.items())
            write_json(OUT/'records.json',rows);write_json(OUT/'summary.json',summary);torch.save(gradient_records,OUT/'gradients.pt')
            write_json(OUT/'verification.json',dict(passed=True,training_steps=0,old_model_and_heatmap_unchanged=True,decoder_weights_unchanged=True,
                loss_reconstruction_max=max_loss_error,gradient_reconstruction_max=max_gradient_error,zero_forward_max=max(replay),
                cached_samples=len(cache),state_samples=len(rows),schedule_weight=sum(counts.values()),heldout_accessed=False,production_promoted=False))
            status('completed');print('COMPLETED',flush=True)
        except BaseException as e:status('failed',error=repr(e));raise
        finally:
            for h in handles:h.remove()


if __name__=='__main__':main()
