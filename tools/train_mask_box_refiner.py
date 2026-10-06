"""Seed17 paired frozen-feature box refinement; preregistered800 steps per arm."""
import argparse
import copy
import fcntl
import json
import os
import subprocess
import sys
import time
import torch.nn.functional as F
from run_depth_pose_ablation import ROOT,torch,VGGTOmega,tensor_state,state_hash,fingerprint,write_json,instance_targets_from_batch
from train_cylinder_adaptive_temperature import TemperatureModel
from train_cylinder_view_balance import oriented_batch,physical_metrics,scene_digest
from train_variable_view_replay import mean_metrics
from mousegeoforge.adaptive_temperature import restore_adaptive
from mousegeoforge.mask_box_refiner import MaskConditionedBoxRefiner,features_from_logits
from training.instance_pose_losses import hungarian_match_instances,box_cxcywh_to_xyxy,generalized_box_iou

OUT=ROOT/'output/mask_box_refiner_seed17_v1'
PARENT=ROOT/'output/cylinder_adaptive_temperature_v1'
CHECKPOINT=PARENT/'adaptive_temperature/final_heads.pt'
ARMS=('token_only','mask_conditioned')


def training_loss(boxes,sample):
    terms=[]
    for v,(qs,ms) in enumerate(sample['matches'][0]):
        good=sample['box_valid'][0,v,ms]
        p=boxes[0,v,qs[good]];t=sample['target_boxes'][0,v,ms[good]]
        if len(p):terms.append((p,t))
    if not terms:raise RuntimeError('No supervised training boxes')
    p=torch.cat([x[0] for x in terms]);t=torch.cat([x[1] for x in terms])
    return 5*F.l1_loss(p,t)+2*(1-generalized_box_iou(box_cxcywh_to_xyxy(p),box_cxcywh_to_xyxy(t)).diagonal()).mean()


def run():
    started=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'output/cylinder_training.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (OUT/'protocol.json').exists():raise RuntimeError('Existing run; preserve protocol')
        def status(state,**kw):write_json(OUT/'status.json',dict(state=state,pid=os.getpid(),elapsed_s=time.monotonic()-started,**kw))
        def deadline():
            if time.monotonic()-started>1200 or time.time()>1789148363:raise RuntimeError('20minute or user deadline limit')
        status('preparing')
        try:
            assert json.loads((PARENT/'delivery_verification.json').read_text())['passed']
            assert json.loads((ROOT/'output/mask_box_failures_v1_actual_train/selection_verification.json').read_text())['passed']
            spec=json.loads((PARENT/'protocol.json').read_text());sequence=spec['arms']['adaptive_temperature']
            assert len(sequence)==800 and all(len(s)==1 for s in sequence)
            keys=list(dict.fromkeys(s[0] for s in sequence));assert len(keys)==356
            assert len({spec['records'][k]['scene'] for k in keys})==40
            for p,h in spec['code_sha256'].items():assert fingerprint(ROOT/p)==h
            inputs={str(PARENT/'protocol.json'):fingerprint(PARENT/'protocol.json'),str(CHECKPOINT):fingerprint(CHECKPOINT)}
            code={p:fingerprint(ROOT/p) for p in ('tools/train_mask_box_refiner.py','mousegeoforge/mask_box_refiner.py','mousegeoforge/mask_shape_audit.py','mousegeoforge/pilot_metrics.py','tools/train_cylinder_view_balance.py')}
            sources=spec['source_sha256'];assert all(scene_digest(ROOT/p)==h for p,h in sources.items())
            protocol=dict(config=spec['config'],sequence=sequence,records=spec['records'],evaluation=spec['evaluation'],input_sha256=inputs,code_sha256=code,source_sha256=sources,
                seed=17,steps_per_arm=800,arms=ARMS,lr=.0003,weight_decay=.01,clip_grad=1.,hidden_dim=128,
                training_scope='Both arms train only identical newly initialized box residual MLPs; all existing model parameters frozen. Same356 unique cached inputs, original800 slot order,40 actual training scenes. Mask OFF zeroes the16 mask features; ON reads them. No instance count, role or GT input.',
                loss='5 mean boxL1 +2 mean(1-GIoU), fixed original training box+keypoint Hungarian assignment, only valid GT boxes. No new keypoint/mask/objectness/depth loss. Original box-only eval assignment remains unchanged.',
                noise='Odd optimizer slots add one foreground pixel at each of four image corners to predicted binary masks before extracting features; even slots clean. Same schedule both arms. Original image/mask prediction/GT never changed. Evaluate clean and corner-stress separately.',
                initialization='Random input projection seed17 shared byte-identically, zero final layer. Initial enabled output equals regressed boxes exactly. Residual center scaled by original WH with tanh; size multiplicative exp with fixed [-2,2] bound.',
                decision='Fixed final800 checkpoint only, original88/17 gates. No dev early stopping or threshold sweep. Advance to second seed only after retention review. This is a frozen-feature box-path pilot, not joint full-network training.',production_promoted=False)
            write_json(OUT/'protocol.json',protocol)
            torch.set_num_threads(2);torch.manual_seed(17);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
            encoder=ROOT/'vggt-omega/vggt_omega_1b_512.pt';assert fingerprint(encoder)==spec['encoder_sha256']
            spatial=VGGTOmega(enable_camera=False,enable_depth=True,enable_multi_instance_pose=True,num_keypoints=23)
            weights=torch.load(encoder,weights_only=True,mmap=True)
            loaded=spatial.load_state_dict({k:v for k,v in weights.items() if k.startswith(('aggregator.','dense_head.'))},strict=False)
            assert not loaded.unexpected_keys and all(k.startswith('instance_pose_head.') for k in loaded.missing_keys)
            del weights
            model=TemperatureModel.from_spatial_model(spatial).cuda().eval();del spatial
            original=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)['model'];restore_adaptive(model,original);model.requires_grad_(False)
            tokens={}
            hook=model.instance_pose_head.instance_decoder.register_forward_hook(lambda module,args,output:tokens.update(value=output.detach().cpu()))
            torch.manual_seed(17);initial=MaskConditionedBoxRefiner();initial_state=copy.deepcopy(initial.state_dict())
            cache={};noise_geometry=[];forward_count=0
            def forward(record):
                nonlocal forward_count
                deadline();batch=oriented_batch(record,num_views=len(record['views']));images=batch['images'].cuda()
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):cached,start=model.aggregator(images)
                with torch.no_grad():pred={k:v.cpu() for k,v in model.forward_cached(images,cached,start,coupling='off').items()}
                token=tokens['value'].reshape(*pred['instance_boxes_cxcywh'].shape[:-1],-1)
                forward_count+=1
                return batch,instance_targets_from_batch(batch),pred,token
            for i,key in enumerate(keys):
                r=spec['records'][key];batch,target,pred,token=forward(r)
                shape=features_from_logits(pred['instance_mask_logits'],r['image_hw'])
                noise=features_from_logits(pred['instance_mask_logits'],r['image_hw'],corner_noise=True)
                sample=dict(boxes=pred['instance_boxes_cxcywh'],tokens=token,clean=shape,noise=noise,
                    matches=hungarian_match_instances(pred,target),target_boxes=target['boxes_cxcywh'],box_valid=target.get('box_valid',target['instance_valid']))
                for arm in ARMS:
                    for features in (shape,noise):
                        out=initial(sample['boxes'],token,features,enabled=True,use_mask=arm=='mask_conditioned')
                        torch.testing.assert_close(out,sample['boxes'],atol=0,rtol=0)
                noise_geometry.append(float((noise[...,4:8]-shape[...,4:8]).abs().max()))
                cache[key]=sample
                del batch,target,pred
                if (i+1)%40==0:status('caching',records=i+1,total=len(keys));print('CACHE',i+1,len(keys),flush=True)
            # No optimization has occurred. Validate actual tensors and gradient reach.
            probe=copy.deepcopy(initial);s=cache[keys[0]]
            output=probe(s['boxes'],s['tokens'],s['clean'],enabled=True,use_mask=True)
            loss=training_loss(output,s);loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in probe.parameters())
            assert probe.output.weight.grad.norm()>0
            assert all(p.grad is None and not p.requires_grad for p in model.parameters())
            # Hook and feature extraction must preserve original saved development outputs.
            e=spec['evaluation'][0];r=spec['records'][e['key']];batch,target,pred,token=forward(r)
            saved=torch.load(PARENT/'adaptive_temperature/prediction_000.pt',weights_only=True,map_location='cpu')
            for k,v in saved['predictions'].items():torch.testing.assert_close(pred[k],v,atol=0,rtol=0)
            assert state_hash(tensor_state(model))==state_hash(original)
            write_json(OUT/'preflight.json',dict(passed=True,disabled_identity=True,all356_train_clean_and_noise_initial_difference=0,development_probe_difference=0,
                actual_loss=float(loss.detach()),last_layer_gradient_norm=float(probe.output.weight.grad.norm()),original_weights_gradients_unchanged=True,
                noisy_main_box_changed_records=sum(x>1e-7 for x in noise_geometry),noise_main_box_max_difference=max(noise_geometry),
                caveat='Zero-init invariance is not evidence of learned noise robustness; final stress evaluation required.',optimizer_steps=0))
            torch.save(cache,OUT/'frozen_training_features.pt')
            del probe,batch,target,pred
            history={};trained={}
            for arm in ARMS:
                branch=MaskConditionedBoxRefiner();branch.load_state_dict(initial_state);branch.train()
                opt=torch.optim.AdamW(branch.parameters(),lr=.0003,weight_decay=.01);rows=[]
                for step,slot in enumerate(sequence,1):
                    deadline();s=cache[slot[0]];shape=s['noise' if step%2 else 'clean'];opt.zero_grad(set_to_none=True)
                    out=branch(s['boxes'],s['tokens'],shape,enabled=True,use_mask=arm=='mask_conditioned')
                    loss=training_loss(out,s);assert torch.isfinite(loss);loss.backward()
                    norm=torch.nn.utils.clip_grad_norm_(branch.parameters(),1.);assert torch.isfinite(norm);opt.step()
                    rows.append(dict(step=step,loss=float(loss.detach()),grad_norm=float(norm),corner_noise=bool(step%2)))
                    if step%100==0:status('training',arm=arm,step=step,total=800);write_json(OUT/f'{arm}_history.json',rows);print('TRAIN',arm,step,rows[-1]['loss'],flush=True)
                branch.eval();trained[arm]=branch;history[arm]=rows
                torch.save(dict(model=branch.state_dict(),optimizer=opt.state_dict(),step=800,protocol_sha256=fingerprint(OUT/'protocol.json')),OUT/f'{arm}_final.pt')
            del cache
            results={name:[] for name in ('regressed',*ARMS,'token_only_corner_stress','mask_conditioned_corner_stress')};parity=[]
            for i,e in enumerate(spec['evaluation']):
                r=spec['records'][e['key']];batch,target,pred,token=forward(r)
                saved=torch.load(PARENT/'adaptive_temperature'/f'prediction_{i:03d}.pt',map_location='cpu',weights_only=True)
                for k,v in saved['predictions'].items():torch.testing.assert_close(pred[k],v,atol=0,rtol=0)
                clean=features_from_logits(pred['instance_mask_logits'],r['image_hw']);noise=features_from_logits(pred['instance_mask_logits'],r['image_hw'],corner_noise=True)
                for name in results:
                    if name=='regressed':p=pred
                    else:
                        arm=name.removesuffix('_corner_stress');shape=noise if name.endswith('_corner_stress') else clean
                        with torch.no_grad():boxes=trained[arm](pred['instance_boxes_cxcywh'],token,shape,enabled=True,use_mask=arm=='mask_conditioned')
                        p=dict(pred,instance_boxes_cxcywh=boxes)
                        assert all(p[k] is v for k,v in pred.items() if k!='instance_boxes_cxcywh')
                    metric,assignment=physical_metrics(p,batch,target)
                    if name=='regressed':
                        assert assignment==saved['oracle_assignment']
                        for k,v in saved['metrics'].items():
                            if v is None:assert metric[k] is None
                            else:parity.append(abs(metric[k]-v))
                    results[name].append(dict(scene=r['scene'],views=r['views'],group=e['group'],metrics=metric))
                    folder=OUT/name;folder.mkdir(exist_ok=True)
                    torch.save(dict(record=r,group=e['group'],oracle_assignment=assignment,metrics=metric,predictions={k:p[k] for k in ('instance_keypoint_xy','instance_boxes_cxcywh','instance_objectness_logits')},targets={k:v for k,v in target.items() if k!='masks'}),folder/f'prediction_{i:03d}.pt')
                del batch,target,pred
                if (i+1)%16==0:status('evaluating',records=i+1,total=88);print('EVAL',i+1,flush=True)
            hook.remove();assert max(parity)<1e-6
            assert state_hash(tensor_state(model))==state_hash(original)
            assert all(p.grad is None and not p.requires_grad for p in model.parameters())
            from pathlib import Path
            assert all(fingerprint(Path(p))==h for p,h in inputs.items())
            assert all(fingerprint(ROOT/p)==h for p,h in code.items())
            assert all(scene_digest(ROOT/p)==h for p,h in sources.items())
            write_json(OUT/'results.json',dict(config=spec['config'],arms={a:dict(records=rows,summary=mean_metrics(rows)) for a,rows in results.items()},production_promoted=False,protocol_sha256=fingerprint(OUT/'protocol.json')))
            write_json(OUT/'verification.json',dict(passed=True,original_predictions_difference=0,original_metric_max_difference=max(parity),existing_weights_sources_code_unchanged=True,only_boxes_changed=True,
                new_branch_optimizer_steps={a:len(history[a]) for a in ARMS},existing_model_optimizer_steps=0,encoder_forwards=forward_count,head_forwards=forward_count,
                initial_branch_sha256=state_hash(initial_state),new_parameters=sum(p.numel() for p in initial.parameters()),elapsed_s=time.monotonic()-started))
            status('completed');print('COMPLETED',flush=True)
        except BaseException as error:status('failed',error=repr(error));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--launch',action='store_true');args=parser.parse_args()
    if args.launch:
        if (OUT/'status.json').exists():raise RuntimeError('Already launched; inspect existing run')
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/'run.log').open('a') as log:p=subprocess.Popen([sys.executable,'-u',__file__],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(pid=p.pid,output=str(OUT))))
    else:run()
