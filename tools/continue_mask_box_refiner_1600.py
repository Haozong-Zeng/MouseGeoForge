"""Exact paired AdamW continuation from800 to fixed1600; no parent overwrites."""
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

INITIAL=ROOT/'output/mask_box_refiner_seed17_v1_no_grad'
OUT=ROOT/'output/mask_box_refiner_seed17_v1_step1600'
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


def equal_tree(a,b):
    if torch.is_tensor(a):torch.testing.assert_close(a,b,atol=0,rtol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal_tree(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal_tree(x,y)
    else:assert a==b


@torch.no_grad()
def fixed_training_loss(branch,cache,sequence,arm):
    return float(sum(float(training_loss(branch(cache[slot[0]]['boxes'],cache[slot[0]]['tokens'],
        cache[slot[0]]['noise' if step%2 else 'clean'],enabled=True,use_mask=arm=='mask_conditioned'),cache[slot[0]]))
        for step,slot in enumerate(sequence,1))/len(sequence))


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
            assert json.loads((INITIAL/'verification.json').read_text())['passed']
            parent_spec=json.loads((INITIAL/'protocol.json').read_text())
            for p,h in parent_spec['code_sha256'].items():assert fingerprint(ROOT/p)==h
            spec=json.loads((PARENT/'protocol.json').read_text());sequence=spec['arms']['adaptive_temperature']
            assert len(sequence)==800 and all(len(s)==1 for s in sequence)
            keys=list(dict.fromkeys(s[0] for s in sequence));assert len(keys)==356
            assert len({spec['records'][k]['scene'] for k in keys})==40
            for p,h in spec['code_sha256'].items():assert fingerprint(ROOT/p)==h
            probe_path=ROOT/'output/mask_box_freeze_equivalence_v1/report.json'
            probe_report=json.loads(probe_path.read_text());assert probe_report['weights_unchanged']
            assert all(v==0 for row in probe_report['rows'] if not row['all_requires_grad_false'] for v in row['differences'].values())
            inputs={str(probe_path):fingerprint(probe_path),str(PARENT/'protocol.json'):fingerprint(PARENT/'protocol.json'),str(CHECKPOINT):fingerprint(CHECKPOINT)}
            inputs.update({str(p):fingerprint(p) for p in [INITIAL/'protocol.json',INITIAL/'verification.json',INITIAL/'frozen_training_features.pt',*[INITIAL/f'{a}_final.pt' for a in ARMS]]})
            code={p:fingerprint(ROOT/p) for p in ('tools/continue_mask_box_refiner_1600.py','mousegeoforge/mask_box_refiner.py','mousegeoforge/mask_shape_audit.py','mousegeoforge/pilot_metrics.py','tools/train_cylinder_view_balance.py')}
            sources=spec['source_sha256'];assert all(scene_digest(ROOT/p)==h for p,h in sources.items())
            protocol=dict(config=spec['config'],sequence=sequence,records=spec['records'],evaluation=spec['evaluation'],input_sha256=inputs,code_sha256=code,source_sha256=sources,
                seed=17,additional_steps_per_arm=800,start_step=800,end_step=1600,arms=ARMS,lr=.0003,weight_decay=.01,clip_grad=1.,hidden_dim=128,
                freeze_policy='Preserve original requires_grad flags to preserve dispatch; all existing forwards run under no_grad, cached features detached, optimizer contains new branch only, assert no existing gradients and state/flags unchanged. Strict zero prediction tolerance retained. Original failed preflight is preserved.',
                training_scope='Both arms continue their800-step box residual MLP and AdamW state; all existing model parameters frozen. Same356 unique cached inputs, original800 slot order,40 actual training scenes. Mask OFF zeroes the16 mask features; ON reads them. No instance count, role or GT input.',
                loss='5 mean boxL1 +2 mean(1-GIoU), fixed original training box+keypoint Hungarian assignment, only valid GT boxes. No new keypoint/mask/objectness/depth loss. Original box-only eval assignment remains unchanged.',
                noise='Odd optimizer slots add one foreground pixel at each of four image corners to predicted binary masks before extracting features; even slots clean. Same schedule both arms. Original image/mask prediction/GT never changed. Evaluate clean and corner-stress separately.',
                initialization='Exact resume of each800-step branch and AdamW state; no reset. Same original800 slots repeated at801-1600. Parent model and cache hashes locked.',
                decision='Fixed final1600 checkpoint only; report both800 and1600, never pick best. Original88/17 gates. Stop extending this MLP if ON clean dual IoU gain vs same-budget OFF <.01 or retention fails; otherwise inspect before any repeat. Fixed training-loss replay before/after on identical800 input/noise slots.',production_promoted=False)
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
            original=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)['model'];restore_adaptive(model,original)
            original_flags={n:p.requires_grad for n,p in model.named_parameters()}
            tokens={}
            hook=model.instance_pose_head.instance_decoder.register_forward_hook(lambda module,args,output:tokens.update(value=output.detach().cpu()))
            torch.manual_seed(17);initial=MaskConditionedBoxRefiner();initial_state=copy.deepcopy(initial.state_dict())
            cache=torch.load(INITIAL/'frozen_training_features.pt',map_location='cpu',weights_only=True);assert set(cache)==set(keys)
            forward_count=0
            def forward(record):
                nonlocal forward_count
                deadline();batch=oriented_batch(record,num_views=len(record['views']));images=batch['images'].cuda()
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):cached,start=model.aggregator(images)
                with torch.no_grad():pred={k:v.cpu() for k,v in model.forward_cached(images,cached,start,coupling='off').items()}
                token=tokens['value'].reshape(*pred['instance_boxes_cxcywh'].shape[:-1],-1)
                forward_count+=1
                return batch,instance_targets_from_batch(batch),pred,token
            write_json(OUT/'preflight.json',dict(passed=True,parent_verification_passed=True,cache_keys=len(cache),optimizer_steps=0,
                scope='Parent strict full-forward preflight remains verified; exact optimizer restoration checked before each arm updates.'))
            history={};trained={};fixed_loss={};resume_checks={}
            for arm in ARMS:
                saved_branch=torch.load(INITIAL/f'{arm}_final.pt',map_location='cpu',weights_only=True);assert saved_branch['step']==800
                branch=MaskConditionedBoxRefiner();branch.load_state_dict(saved_branch['model']);branch.train()
                opt=torch.optim.AdamW(branch.parameters(),lr=.0003,weight_decay=.01);opt.load_state_dict(saved_branch['optimizer'])
                equal_tree(branch.state_dict(),saved_branch['model']);equal_tree(opt.state_dict(),saved_branch['optimizer'])
                assert all(float(x['step'])==800 for x in opt.state.values())
                resume_checks[arm]=dict(model_exact=True,optimizer_exact=True,optimizer_step=800)
                fixed_loss[arm]=dict(step800=fixed_training_loss(branch,cache,sequence,arm))
                write_json(OUT/'resume_verification.json',resume_checks);rows=[]
                for step,slot in enumerate(sequence,801):
                    deadline();s=cache[slot[0]];shape=s['noise' if step%2 else 'clean'];opt.zero_grad(set_to_none=True)
                    out=branch(s['boxes'],s['tokens'],shape,enabled=True,use_mask=arm=='mask_conditioned')
                    loss=training_loss(out,s);assert torch.isfinite(loss);loss.backward()
                    norm=torch.nn.utils.clip_grad_norm_(branch.parameters(),1.);assert torch.isfinite(norm);opt.step()
                    rows.append(dict(step=step,loss=float(loss.detach()),grad_norm=float(norm),corner_noise=bool(step%2)))
                    if step%100==0:status('training',arm=arm,step=step,total=1600);write_json(OUT/f'{arm}_history.json',rows);print('TRAIN',arm,step,rows[-1]['loss'],flush=True)
                branch.eval();trained[arm]=branch;history[arm]=rows
                fixed_loss[arm]['step1600']=fixed_training_loss(branch,cache,sequence,arm)
                assert all(float(x['step'])==1600 for x in opt.state.values())
                write_json(OUT/'fixed_training_loss.json',fixed_loss)
                torch.save(dict(model=branch.state_dict(),optimizer=opt.state_dict(),step=1600,protocol_sha256=fingerprint(OUT/'protocol.json')),OUT/f'{arm}_final.pt')
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
            assert all(p.grad is None for p in model.parameters())
            assert {n:p.requires_grad for n,p in model.named_parameters()}==original_flags
            from pathlib import Path
            assert all(fingerprint(Path(p))==h for p,h in inputs.items())
            assert all(fingerprint(ROOT/p)==h for p,h in code.items())
            assert all(scene_digest(ROOT/p)==h for p,h in sources.items())
            write_json(OUT/'results.json',dict(config=spec['config'],arms={a:dict(records=rows,summary=mean_metrics(rows)) for a,rows in results.items()},production_promoted=False,protocol_sha256=fingerprint(OUT/'protocol.json')))
            write_json(OUT/'verification.json',dict(passed=True,original_predictions_difference=0,original_metric_max_difference=max(parity),existing_weights_sources_code_unchanged=True,only_boxes_changed=True,
                new_branch_additional_optimizer_steps={a:len(history[a]) for a in ARMS},new_branch_cumulative_steps={a:1600 for a in ARMS},optimizer_exact_resume=True,existing_model_optimizer_steps=0,encoder_forwards=forward_count,head_forwards=forward_count,
                parent_branch_sha256={a:inputs[str(INITIAL/f'{a}_final.pt')] for a in ARMS},new_parameters=sum(p.numel() for p in initial.parameters()),elapsed_s=time.monotonic()-started))
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
