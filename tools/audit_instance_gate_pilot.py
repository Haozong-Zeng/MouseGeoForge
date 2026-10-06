"""Frozen readout decomposition; CF outputs are not trained/deployed models."""
import fcntl,json,os,time
import cv2,numpy as np,torch
from torch.nn import functional as F
from run_depth_pose_ablation import ROOT,VGGTOmega,DepthPoseModel,write_json,fingerprint,instance_targets_from_batch
from train_cylinder_view_balance import oriented_batch
from mousegeoforge.pilot_metrics import evaluate_scene
from mousegeoforge.keypoint_breakdown import keypoint_rows,summarize_joint_rows
from mousegeoforge.adaptive_temperature import install_adaptive_temperature,restore_adaptive
from mousegeoforge.spatial_anchor_experiment import install_spatial_anchor,spatial_anchor_context
from mousegeoforge.spatial_heatmap_probe import spatial_expectation
from mousegeoforge.visible_heatmap_supervision import anchor_scores
from mousegeoforge.deterministic_local_sampling import deterministic_local_refine
from mousegeoforge.mask_box_candidate import tensor_mapping_sha256
from mousegeoforge.instance_adaptive_gate import InstanceAdaptiveGate,predicted_quality

OUT=ROOT/'output/instance_gate_pilot_audit_v1'
GATE=ROOT/'output/instance_gate_pilot_v1'
PILOT=ROOT/'output/spatial_anchor_visible_heatmap_v1'
REAL=ROOT/'output/real_cylinder_bottom_frozen_v1'
BODY=list(range(7))+list(range(14,23))
ARMS=['baseline_refined','heatmap_frozen','static_gate','dynamic_gate']


def main():
    torch.set_num_threads(2);cv2.setNumThreads(1);torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with (ROOT/'output/cylinder_training.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);OUT.mkdir(exist_ok=False);(OUT/'review').mkdir()
        began=time.monotonic();handles=[]
        def status(state,**kw):write_json(OUT/'status.json',dict(state=state,pid=os.getpid(),elapsed_s=time.monotonic()-began,**kw))
        status('preparing')
        try:
            assert json.loads((PILOT/'verification.json').read_text())['passed']
            p=json.loads((PILOT/'protocol.json').read_text());spec=json.loads((ROOT/'output/cylinder_adaptive_temperature_v1/protocol.json').read_text())
            real=json.loads((REAL/'protocol.json').read_text());assert not set(real['sources'])&{'GR2','UR2','UR3'}
            assert all(fingerprint(ROOT/k)==v for k,v in p['code_sha256'].items())
            base=ROOT/'output/cylinder_adaptive_temperature_v1/adaptive_temperature/final_heads.pt';encoder=ROOT/'vggt-omega/vggt_omega_1b_512.pt'
            assert json.loads((GATE/'verification.json').read_text())['passed']
            gate_protocol=json.loads((GATE/'protocol.json').read_text())
            assert all(fingerprint(ROOT/k)==v for k,v in gate_protocol['code_sha256'].items())
            candidate=PILOT/'heatmap_on_step800.pt';hashes={str(f.relative_to(ROOT)):fingerprint(f) for f in (base,encoder,candidate)}
            assert fingerprint(base)==p['base_sha256'] and fingerprint(encoder)==p['encoder_sha256']
            hashes.update({str(f.relative_to(ROOT)):fingerprint(f) for f in [GATE/'static_step400.pt',GATE/'dynamic_step400.pt',GATE/'protocol.json']})
            write_json(OUT/'protocol.json',dict(pilot_sha256=fingerprint(PILOT/'protocol.json'),hashes=hashes,arms=ARMS,training_steps=0,
                synthetic_records=88,real_sources=real['sources'],real_conditions=['bottom','side','sides_bottom'],
                candidate='Additional static/dynamic gate400, frozen heatmap_ON800. Same source fixed budget; not parameter-count matched.',
                matching='Original box+objectness Hungarian; identical across all readouts',heldout_accessed=False,
                warning='Static23 vs dynamic8632 added parameters; no real manual GT; complete frozen retention check, not production deployment.'))
            spatial=VGGTOmega(enable_camera=False,enable_depth=True,enable_multi_instance_pose=True,num_keypoints=23)
            weights=torch.load(encoder,map_location='cpu',weights_only=True,mmap=True)
            loaded=spatial.load_state_dict({k:v for k,v in weights.items() if k.startswith(('aggregator.','dense_head.'))},strict=False)
            assert not loaded.unexpected_keys and all(k.startswith('instance_pose_head.') for k in loaded.missing_keys)
            model=install_adaptive_temperature(DepthPoseModel.from_spatial_model(spatial)).cuda().eval()
            restore_adaptive(model,torch.load(base,map_location='cpu',weights_only=True)['model']);del weights,spatial
            model.requires_grad_(False);basehash=tensor_mapping_sha256(model.state_dict())
            install_spatial_anchor(model);head=model.instance_pose_head;branch=head.spatial_anchor
            branch.load_state_dict(torch.load(candidate,map_location='cpu',weights_only=True)['branch']);model.eval().requires_grad_(False)
            branchhash=tensor_mapping_sha256({k:v.reshape(-1) for k,v in branch.state_dict().items()});captured={};replay=[]
            gates={}
            for a in ('static','dynamic'):
                checkpoint=torch.load(GATE/f'{a}_step400.pt',map_location='cpu',weights_only=True)
                gate=InstanceAdaptiveGate(dynamic=a=='dynamic').cuda();gate.load_state_dict(checkpoint['state']);gate.eval().requires_grad_(False);gates[a]=gate
            def capture(name,inputs=False):
                def hook(m,a,value):
                    captured[name]=value.detach()
                    if inputs:captured['tokens']=a[0].detach()
                return hook
            handles=[head.pyramid_refine[0].register_forward_hook(capture('pixels')),head.coarse_keypoint_xy.register_forward_hook(capture('raw',True)),
                     head.local_pixel_projection.register_forward_hook(capture('local_pixel')),head.local_query_projection.register_forward_hook(capture('local_query'))]
            def infer(x):
                if time.monotonic()-began>1200 or time.time()>1789233567:raise TimeoutError('Audit cutoff')
                with torch.no_grad():
                    with torch.autocast('cuda',dtype=torch.bfloat16):cache,start=model.aggregator(x)
                    basepred=model.forward_cached(x,cache,start,coupling='off');s=dict(captured)
                    s.update(local_query=F.normalize(s['local_query'],dim=-1,eps=1e-6),scale=head.last_temperature_scale.detach(),
                             size=head.local_heatmap_size,radius=head.local_window_radius)
                    scores=anchor_scores(branch,s['tokens'],s['pixels'],basepred['instance_mask_logits'][0])
                    anchor=spatial_expectation(s['raw'],scores,enabled=True).clamp(1e-5,1-1e-5)
                    mixed=branch(s['raw'],s['tokens'],s['pixels'],basepred['instance_mask_logits'][0],enabled=True).sigmoid()
                    raw=s['raw'].sigmoid();base_replay=deterministic_local_refine(s,raw)
                    final=deterministic_local_refine(s,mixed);pure_local=deterministic_local_refine(s,anchor)
                    with spatial_anchor_context(model,enabled=True):trained=model.forward_cached(x,cache,start,coupling='off')
                    torch.testing.assert_close(base_replay,basepred['instance_keypoint_xy'][0],atol=1e-6,rtol=1e-6)
                    torch.testing.assert_close(mixed,trained['instance_keypoint_coarse_xy'][0],atol=0,rtol=0)
                    torch.testing.assert_close(final,trained['instance_keypoint_xy'][0],atol=1e-6,rtol=1e-6)
                    replay.append(float((final-trained['instance_keypoint_xy'][0]).abs().max()))
                    for k in ('depth','instance_boxes_cxcywh','instance_mask_logits','instance_objectness_logits'):
                        torch.testing.assert_close(basepred[k],trained[k],atol=0,rtol=0)
                    quality=predicted_quality(scores,basepred['instance_keypoint_visibility_logits'][0],basepred['instance_mask_logits'][0],basepred['instance_objectness_logits'][0])
                    new_outputs=[]
                    for gate in gates.values():
                        new_raw,_=gate(s['raw'],anchor,branch.gate.tanh(),s['tokens'],quality)
                        new_outputs.append(deterministic_local_refine(s,new_raw.sigmoid()))
                    coordinates=[basepred['instance_keypoint_xy'][0],trained['instance_keypoint_xy'][0],*new_outputs]
                    cpu={k:v.cpu() for k,v in basepred.items()}
                    outputs={a:dict(cpu,instance_keypoint_xy=xy[None].cpu()) for a,xy in zip(ARMS,coordinates)}
                    captured.clear();return outputs
            def visualize(rgb,outputs,path,target=None,matches=None):
                panels=[]
                for v,im in enumerate(rgb):
                    h,w=im.shape[:2];cols=[]
                    for a,pred in outputs.items():
                        canvas=cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
                        qs=matches[v][0] if matches is not None else [int(pred['instance_objectness_logits'][0,v].argmax())]
                        if target is not None:
                            for m in matches[v][1]:
                                for k in BODY:
                                    if target['keypoint_valid'][0,v,m,k]:
                                        pt=tuple(np.rint(target['keypoint_xy'][0,v,m,k].numpy()*[w,h]).astype(int));cv2.circle(canvas,pt,5,(0,255,0),1)
                        for q in qs:
                            for k in BODY:
                                pt=tuple(np.rint(pred['instance_keypoint_xy'][0,v,q,k].numpy()*[w,h]).astype(int));cv2.circle(canvas,pt,3,(0,255,255),-1)
                        cv2.rectangle(canvas,(0,0),(w,35),(20,20,20),-1)
                        cv2.putText(canvas,a,(5,24),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
                        cols.append(cv2.resize(canvas,(352,round(h*352/w))))
                    panels.append(np.concatenate(cols,1))
                cv2.imwrite(str(path),np.concatenate(panels,0))
            results={a:[] for a in ARMS};joints={a:[] for a in ARMS};seen=set();readouts={}
            for i,e in enumerate(spec['evaluation']):
                r=spec['records'][e['key']];batch=oriented_batch(r,num_views=len(r['views']));target=instance_targets_from_batch(batch)
                target['keypoint_visibility_valid']=batch['instance_keypoint_visibility_valid']
                outputs=infer(batch['images'].cuda());reference=None
                for a,pred in outputs.items():
                    metric,matches=evaluate_scene(pred,batch,target)
                    if reference is None:reference=matches
                    else:assert matches==reference
                    results[a].append(dict(index=i,group=e['group'],metrics=metric))
                    joints[a].extend(dict(index=i,group=e['group'],**j) for j in keypoint_rows(pred,batch,target,matches))
                if e['group'] not in seen:
                    rgb=np.uint8(np.clip(batch['images'][0].permute(0,2,3,1).numpy()*255,0,255));visualize(rgb,outputs,OUT/f'review/synthetic_{i:03d}_{e["group"]}.png',target,reference);seen.add(e['group'])
                status('synthetic',records=i+1,total=88)
            count=0;imagehashes={}
            for session,frames in real['sources'].items():
                for frame in frames:
                    for condition in ('bottom','side','sides_bottom'):
                        paths=[REAL/f'review/{session}_{frame:06d}_{c}_input.png' for c in real['conditions'][condition]]
                        for f in paths:imagehashes[str(f.relative_to(ROOT))]=fingerprint(f)
                        rgb=[cv2.cvtColor(cv2.imread(str(f)),cv2.COLOR_BGR2RGB) for f in paths]
                        x=torch.from_numpy(np.stack(rgb)).permute(0,3,1,2)[None].cuda().float()/255;outputs=infer(x)
                        name=f'{session}_{frame:06d}_{condition}';visualize(rgb,outputs,OUT/f'review/real_{name}.png')
                        readouts[name]={a:{k:p[k] for k in ('instance_keypoint_xy','instance_objectness_logits')} for a,p in outputs.items()}
                        count+=1;status('real',records=count,total=27)
            summary={};breakdown={}
            for a,rows in results.items():
                summary[a]={};breakdown[a]={}
                for g in dict.fromkeys(r['group'] for r in rows):
                    ms=[r['metrics'] for r in rows if r['group']==g]
                    summary[a][g]={k:float(np.mean([m[k] for m in ms if m[k] is not None])) if any(m[k] is not None for m in ms) else None for k in ms[0]}
                    jr=[r for r in joints[a] if r['group']==g]
                    breakdown[a][g]={name:summarize_joint_rows([r for r in jr if r['keypoint'] in ks]) for name,ks in [('body',BODY),('tail',list(range(7,14)))]}
                    breakdown[a][g].update({name:summarize_joint_rows([r for r in jr if r['visible'] is flag]) for name,flag in [('visible',True),('occluded',False)]})
                    breakdown[a][g]['per_joint']={str(k):summarize_joint_rows([r for r in jr if r['keypoint']==k]) for k in range(23)}
            assert tensor_mapping_sha256({k:v for k,v in model.state_dict().items() if not k.startswith('instance_pose_head.spatial_anchor.')})==basehash
            assert tensor_mapping_sha256({k:v.reshape(-1) for k,v in branch.state_dict().items()})==branchhash
            assert all(fingerprint(ROOT/f)==s for f,s in {**hashes,**imagehashes}.items())
            write_json(OUT/'summary.json',summary);write_json(OUT/'synthetic_results.json',results);write_json(OUT/'joint_summary.json',breakdown)
            write_json(OUT/'joint_records.json',joints);torch.save(readouts,OUT/'real_readouts.pt')
            write_json(OUT/'verification.json',dict(passed=True,synthetic_records=88,real_groups=count,training_steps=0,
                full_head_coordinate_replay_max=max(replay),original_outputs_unchanged=True,base_and_branch_unchanged=True,
                evaluation_matching_unchanged=True,production_promoted=False,source_images_sha256=imagehashes))
            status('completed');print('COMPLETED',flush=True)
        except BaseException as e:status('failed',error=repr(e));raise
        finally:
            for h in handles:h.remove()


if __name__=='__main__':main()


