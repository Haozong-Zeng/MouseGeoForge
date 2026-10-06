"""Read-only gate gradients/distributions; no optimizer or parameter updates."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import fcntl,json,time
import cv2,numpy as np,torch
from torch.nn import functional as F
from run_depth_pose_ablation import ROOT,VGGTOmega,DepthPoseModel,write_json,fingerprint,instance_targets_from_batch
from train_cylinder_view_balance import oriented_batch
from mousegeoforge.adaptive_temperature import install_adaptive_temperature,restore_adaptive
from mousegeoforge.spatial_anchor_experiment import install_spatial_anchor,spatial_anchor_context
from mousegeoforge.visible_heatmap_supervision import anchor_scores
from mousegeoforge.spatial_heatmap_probe import spatial_expectation
from mousegeoforge.instance_adaptive_gate import InstanceAdaptiveGate,predicted_quality
from mousegeoforge.deterministic_local_sampling import deterministic_local_refine
from mousegeoforge.affine_batch_experiment import augment_batch
from mousegeoforge.pilot_metrics import evaluate_scene
from mousegeoforge.mask_box_candidate import tensor_mapping_sha256

OUT=ROOT/'output/gate_objective_diagnostic_v1'
PILOT=ROOT/'output/instance_gate_pilot_v1'
HEAT=ROOT/'output/spatial_anchor_visible_heatmap_v1'
REAL=ROOT/'output/real_cylinder_bottom_frozen_v1'
BODY=list(range(7))+list(range(14,23))


def coordinate_losses(coarse,refined,s):
    a=[];b=[]
    for v,(qs,ms) in enumerate(s['matches']):
        for q,m in zip(qs,ms):
            valid=s['target']['keypoint_valid'][0,v,m].to(coarse.device);xy=s['target']['keypoint_xy'][0,v,m].to(coarse.device)
            if valid.any():
                a.append(F.smooth_l1_loss(refined[v,q,valid],xy[valid],beta=.05));b.append(F.smooth_l1_loss(coarse[v,q,valid],xy[valid],beta=.05))
    return torch.stack(a).mean(),torch.stack(b).mean()


def flatten_grad(values,parameters):
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g,p in zip(values,parameters)])


def main():
    torch.set_num_threads(2);cv2.setNumThreads(1);torch.manual_seed(17)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with (ROOT/'output/cylinder_training.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);OUT.mkdir(exist_ok=False);began=time.monotonic();handles=[]
        def status(state,**kw):write_json(OUT/'status.json',dict(state=state,pid=os.getpid(),elapsed_s=time.monotonic()-began,**kw))
        status('preparing')
        try:
            p=json.loads((PILOT/'protocol.json').read_text());plan=json.loads((ROOT/'output/spatial_anchor_affine_batch_preflight_v1_retry1/plan.json').read_text())
            real=json.loads((REAL/'protocol.json').read_text())
            assert json.loads((PILOT/'verification.json').read_text())['passed']
            assert all(fingerprint(ROOT/k)==v for k,v in p['code_sha256'].items())
            assert not set(real['sources'])&{'GR2','UR2','UR3'}
            hashes=dict(p['hashes']);hashes.update({str(f.relative_to(ROOT)):fingerprint(f) for f in [PILOT/'static_step400.pt',PILOT/'dynamic_step400.pt']})
            assert all(fingerprint(ROOT/k)==v for k,v in hashes.items())
            write_json(OUT/'protocol.json',dict(hashes=hashes,training_steps=0,train='32 identities + one fixed nonidentity affine per scene by scene index mod4',
                 dev='fixed16 source-disjoint previously reused development',real='9 known development timestamps bottom/side/sides_bottom',
                 gradients='Unweighted refined and coarse SmoothL1 beta.05, same instance matching; parameter cosine and per-gate directional conflict. No optimizer.',
                 distribution='Matched synthetic instances; highest score slot per real view, NOT manual GT. Stratify role and source; raw pooled train/real not comparable populations.',heldout_accessed=False))
            base=ROOT/'output/cylinder_adaptive_temperature_v1/adaptive_temperature/final_heads.pt';encoder=ROOT/'vggt-omega/vggt_omega_1b_512.pt'
            spatial=VGGTOmega(enable_camera=False,enable_depth=True,enable_multi_instance_pose=True,num_keypoints=23)
            weights=torch.load(encoder,map_location='cpu',weights_only=True,mmap=True)
            loaded=spatial.load_state_dict({k:v for k,v in weights.items() if k.startswith(('aggregator.','dense_head.'))},strict=False)
            assert not loaded.unexpected_keys and all(k.startswith('instance_pose_head.') for k in loaded.missing_keys)
            model=install_adaptive_temperature(DepthPoseModel.from_spatial_model(spatial)).cuda().eval()
            restore_adaptive(model,torch.load(base,map_location='cpu',weights_only=True)['model']);del spatial,weights
            install_spatial_anchor(model);head=model.instance_pose_head;branch=head.spatial_anchor
            branch.load_state_dict(torch.load(HEAT/'heatmap_on_step800.pt',map_location='cpu',weights_only=True)['branch']);model.eval().requires_grad_(False)
            oldhash=tensor_mapping_sha256({k:v.reshape(-1) for k,v in model.state_dict().items()});gates={};gatehashes={}
            for arm in ('static','dynamic'):
                g=InstanceAdaptiveGate(dynamic=arm=='dynamic').cuda();g.load_state_dict(torch.load(PILOT/f'{arm}_step400.pt',map_location='cpu',weights_only=True)['state']);g.eval();gates[arm]=g
                gatehashes[arm]=tensor_mapping_sha256({k:v.reshape(-1) for k,v in g.state_dict().items()})
            captured={}
            def capture(name,inputs=False):
                def hook(m,a,value):
                    captured[name]=value.detach()
                    if inputs:captured['tokens']=a[0].detach()
                return hook
            handles=[head.pyramid_refine[0].register_forward_hook(capture('pixels')),head.coarse_keypoint_xy.register_forward_hook(capture('raw',True)),
                     head.local_pixel_projection.register_forward_hook(capture('local_pixel')),head.local_query_projection.register_forward_hook(capture('local_query'))]
            points=[];gradients=[];counter=0
            def inspect(x,metadata,batch=None):
                nonlocal counter
                if time.monotonic()-began>1200 or time.time()>1789233567:raise TimeoutError('Diagnostic cutoff')
                with torch.no_grad():
                    with torch.autocast('cuda',dtype=torch.bfloat16):cache,start=model.aggregator(x)
                    old=model.forward_cached(x,cache,start,coupling='off');s=dict(captured)
                    score=anchor_scores(branch,s['tokens'],s['pixels'],old['instance_mask_logits'][0])
                    anchor=spatial_expectation(s['raw'],score,enabled=True).clamp(1e-5,1-1e-5)
                    quality=predicted_quality(score,old['instance_keypoint_visibility_logits'][0],old['instance_mask_logits'][0],old['instance_objectness_logits'][0])
                    s.update(local_query=F.normalize(s['local_query'],dim=-1,eps=1e-6),scale=head.last_temperature_scale.detach(),size=head.local_heatmap_size,radius=head.local_window_radius)
                    if batch is not None:
                        target=instance_targets_from_batch(batch);_,matches=evaluate_scene({k:v.cpu() for k,v in old.items()},batch,target)
                        s.update(target=target,matches=matches)
                    else:matches=[([int(old['instance_objectness_logits'][0,v].argmax())],[-1]) for v in range(x.shape[1])]
                for arm,gate in gates.items():
                    raw,g=gate(s['raw'],anchor,branch.gate.tanh(),s['tokens'],quality);coarse=raw.sigmoid();refined=deterministic_local_refine(s,coarse)
                    gr=gc=None
                    if batch is not None:
                        lr,lc=coordinate_losses(coarse,refined,s);params=list(gate.parameters())
                        ar=torch.autograd.grad(lr,params,retain_graph=True,allow_unused=True);ac=torch.autograd.grad(lc,params,retain_graph=True,allow_unused=True)
                        vr=flatten_grad(ar,params);vc=flatten_grad(ac,params);den=vr.norm()*vc.norm()
                        gr=torch.autograd.grad(lr,g,retain_graph=True)[0];gc=torch.autograd.grad(lc,g,retain_graph=True)[0]
                        active=(gr.abs()>1e-12)&(gc.abs()>1e-12)
                        gradients.append(dict(**metadata,arm=arm,refined_loss=float(lr.detach()),coarse_loss=float(lc.detach()),
                            gradient_cosine=float((vr@vc)/den) if den>1e-12 else None,refined_grad_norm=float(vr.norm()),coarse_grad_norm=float(vc.norm()),
                            active_gate_count=int(active.sum()),opposed_gate_count=int(((gr*gc<0)&active).sum())))
                    for v,(qs,ms) in enumerate(matches):
                        for q,m in zip(qs,ms):
                            for k in range(23):
                                row=dict(**metadata,arm=arm,view=v,role=metadata['roles'][v],instance=m,query=q,keypoint=k,body=k in BODY,
                                         gate=float(g[v,q,k].detach()),entropy=float(quality[v,q,k,0]),peak=float(quality[v,q,k,1]),
                                         predicted_visibility=float(quality[v,q,k,2]),mask_area=float(quality[v,q,k,3]),objectness=float(quality[v,q,k,4]))
                                if batch is not None:
                                    valid=bool(target['keypoint_valid'][0,v,m,k]);scale=coarse.new_tensor([x.shape[-1],x.shape[-2]])
                                    diag=(target['boxes_cxcywh'][0,v,m,2:].cuda()*scale).norm().clamp_min(1)
                                    gt=target['keypoint_xy'][0,v,m,k].cuda()
                                    row.update(valid=valid,visible=bool(batch['instance_keypoint_visible'][0,v,m,k]) if batch['instance_keypoint_visibility_valid'][0,v,m,k] else None,
                                        coarse_nme=float(((coarse[v,q,k]-gt)*scale).norm().detach()/diag) if valid else None,
                                        refined_nme=float(((refined[v,q,k]-gt)*scale).norm().detach()/diag) if valid else None,
                                        coarse_gate_gradient=float(gc[v,q,k]),refined_gate_gradient=float(gr[v,q,k]))
                                points.append(row)
                    del raw,g,coarse,refined
                captured.clear();counter+=1;status('diagnosing',records=counter,total=107)
            for i,key in enumerate(p['train_keys']):
                record=p['records'][key];batch=oriented_batch(record,num_views=len(record['views']))
                for variant in [plan['variants'][0],plan['variants'][1+i%4]]:
                    b,_=augment_batch(batch,**{k:v for k,v in variant.items() if k!='name'})
                    inspect(b['images'].cuda(),dict(split='train',group=record['source_group'],key=key,variant=variant['name'],roles=record['physical_roles']),b)
            for e in p['evaluation']:
                record=p['records'][e['key']];batch=oriented_batch(record,num_views=len(record['views']))
                inspect(batch['images'].cuda(),dict(split='dev',group=e['group'],key=e['key'],variant='identity',roles=record['physical_roles']),batch)
            for session,frames in real['sources'].items():
                for frame in frames:
                    for condition in ('bottom','side','sides_bottom'):
                        cams=real['conditions'][condition];rgb=[cv2.cvtColor(cv2.imread(str(REAL/f'review/{session}_{frame:06d}_{c}_input.png')),cv2.COLOR_BGR2RGB) for c in cams]
                        x=torch.from_numpy(np.stack(rgb)).permute(0,3,1,2)[None].cuda().float()/255
                        inspect(x,dict(split='real',group=condition,key=f'{session}_{frame:06d}',variant='identity',roles=['bottom' if c=='A' else 'side' for c in cams]))
            summary={}
            for arm in gates:
                summary[arm]={}
                for split in ('train','dev','real'):
                    summary[arm][split]={}
                    for role in ('bottom','top','side'):
                        rows=[r for r in points if r['arm']==arm and r['split']==split and r['role']==role and r['body'] and r.get('valid',True)]
                        if rows:summary[arm][split][role]=dict(n=len(rows),**{k:float(np.mean([r[k] for r in rows])) for k in ('gate','entropy','peak','predicted_visibility','mask_area')},gate_p10=float(np.percentile([r['gate'] for r in rows],10)),gate_p90=float(np.percentile([r['gate'] for r in rows],90)))
                summary[arm]['gradient_groups']={}
                for split,group in dict.fromkeys((r['split'],r['group']) for r in gradients):
                    rows=[r for r in gradients if r['arm']==arm and r['split']==split and r['group']==group]
                    cs=[r['gradient_cosine'] for r in rows if r['gradient_cosine'] is not None]
                    summary[arm]['gradient_groups'][split+'/'+group]=dict(n=len(rows),mean_cosine=float(np.mean(cs)),negative_fraction=float(np.mean(np.array(cs)<0)),
                         opposed_gate_fraction=sum(r['opposed_gate_count'] for r in rows)/max(1,sum(r['active_gate_count'] for r in rows)))
            assert tensor_mapping_sha256({k:v.reshape(-1) for k,v in model.state_dict().items()})==oldhash
            for arm,g in gates.items():
                assert tensor_mapping_sha256({k:v.reshape(-1) for k,v in g.state_dict().items()})==gatehashes[arm]
                assert all(p.grad is None for p in g.parameters())
            assert all(fingerprint(ROOT/k)==v for k,v in hashes.items())
            write_json(OUT/'points.json',points);write_json(OUT/'gradients.json',gradients);write_json(OUT/'summary.json',summary)
            write_json(OUT/'verification.json',dict(passed=True,groups=counter,point_rows=len(points),gradient_records=len(gradients),optimizer_steps=0,all_parameters_unchanged=True,heldout_accessed=False))
            status('completed');print(json.dumps(summary,indent=2),flush=True)
        except BaseException as e:status('failed',error=repr(e));raise
        finally:
            for h in handles:h.remove()


if __name__=='__main__':main()
