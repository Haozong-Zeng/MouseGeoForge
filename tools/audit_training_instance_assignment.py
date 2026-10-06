"""Frozen seed17 snapshot of all356 actually used training inputs, no updates."""
import argparse
from collections import Counter
import fcntl
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from torch.nn import functional as F
from run_depth_pose_ablation import ROOT,torch,VGGTOmega,tensor_state,state_hash,fingerprint,write_json,instance_targets_from_batch
from train_cylinder_adaptive_temperature import TemperatureModel
from train_cylinder_view_balance import oriented_batch,scene_digest
from mousegeoforge.adaptive_temperature import restore_adaptive
from mousegeoforge.instance_assignment_audit import matching_costs,assignment,assignment_gap,objectness_terms
from training.instance_pose_losses import hungarian_match_instances

OUT=ROOT/'output/training_instance_assignment_v1'
PARENT=ROOT/'output/cylinder_adaptive_temperature_v1'
CHECKPOINT=PARENT/'adaptive_temperature/final_heads.pt'


def run():
    started=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'output/cylinder_training.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (OUT/'protocol.json').exists():raise RuntimeError('Preserve existing protocol')
        def status(state,**kw):write_json(OUT/'status.json',dict(state=state,pid=os.getpid(),elapsed_s=time.monotonic()-started,**kw))
        status('preparing')
        try:
            spec=json.loads((PARENT/'protocol.json').read_text());sequence=spec['arms']['adaptive_temperature']
            keys=list(dict.fromkeys(s[0] for s in sequence))
            assert len(keys)==356 and len(sequence)==800
            visits=Counter(s[0] for s in sequence)
            cache_path=ROOT/'output/mask_box_refiner_seed17_v1_no_grad/frozen_training_features.pt'
            inputs={str(p):fingerprint(p) for p in (PARENT/'protocol.json',CHECKPOINT,cache_path)}
            for p,h in spec['code_sha256'].items():assert fingerprint(ROOT/p)==h
            assert all(scene_digest(ROOT/p)==h for p,h in spec['source_sha256'].items())
            code={p:fingerprint(ROOT/p) for p in ('tools/audit_training_instance_assignment.py',
                'mousegeoforge/instance_assignment_audit.py','tests/test_instance_assignment_audit.py')}
            write_json(OUT/'protocol.json',dict(seed=17,keys=keys,sequence=sequence,records=spec['records'],
                input_sha256=inputs,code_sha256=code,source_sha256=spec['source_sha256'],encoder_sha256=spec['encoder_sha256'],
                scope='All356 original unique train inputs/40 poses, fixed final temperature seed17. No training, no new data, no development-based optimization.',
                cost='Original -sigmoid(objectness)+5boxL1+5valid-keypointL1, versus original+2maskBCE+2maskDice. All low-resolution pixels, nearest GT exactly matching existing loss; no random sampling, no sweep.',
                decision='If <=1% train view assignments change and annotations/negative supervision complete, do not launch matching-only OFF/ON; investigate representation/data diversity separately. Diagnostic threshold fixed before results.',
                freeze='Preserve requires_grad flags and modes; eval/no_grad forward. Compare all356 boxes,tokens,matches exactly with existing frozen cache; optimizer absent.',
                no_object=.1,gradient='Analytic dL/dlogit of original weighted BCE, checked by unit autograd. Not network-gradient improvement.',
                safety_seconds=1200,model_updates=0,production_promoted=False))
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
            encoder_hash=state_hash({n:p.detach().cpu() for n,p in model.aggregator.state_dict().items()})
            cache=torch.load(cache_path,map_location='cpu',weights_only=True)
            tokens={}
            hook=model.instance_pose_head.instance_decoder.register_forward_hook(lambda module,args,output:tokens.update(value=output.detach().cpu()))
            rows=[]
            for i,key in enumerate(keys):
                if time.monotonic()-started>1200:raise RuntimeError('20minute cap')
                r=spec['records'][key];batch=oriented_batch(r,num_views=len(r['views']));images=batch['images'].cuda()
                target=instance_targets_from_batch(batch)
                assert target['detection_annotation_complete'].all()
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):cached,start=model.aggregator(images)
                with torch.no_grad():pred={k:v.cpu() for k,v in model.forward_cached(images,cached,start,coupling='off').items()}
                token=tokens['value'].reshape(*pred['instance_boxes_cxcywh'].shape[:-1],-1)
                torch.testing.assert_close(pred['instance_boxes_cxcywh'],cache[key]['boxes'],atol=0,rtol=0)
                torch.testing.assert_close(token,cache[key]['tokens'],atol=0,rtol=0)
                matches=hungarian_match_instances(pred,target)
                for (q,m),(cq,cm) in zip(matches[0],cache[key]['matches'][0]):
                    torch.testing.assert_close(q,cq,atol=0,rtol=0);torch.testing.assert_close(m,cm,atol=0,rtol=0)
                labels,weight,gradient=objectness_terms(pred['instance_objectness_logits'],matches,target['detection_annotation_complete'])
                assert (weight[labels==0]>0).all() and (gradient[labels==0]>=0).all()
                for v,(qs,ms) in enumerate(matches[0]):
                    ids,cost=matching_costs(pred,target,0,v)
                    old=assignment(cost['original'],ids);new=assignment(cost['mask_augmented'],ids)
                    assert old=={int(m):int(q) for q,m in zip(qs,ms)}
                    logits=pred['instance_mask_logits'][0,v];binary=logits>0
                    gt=F.interpolate(target['masks'][0,v,ids,None].float(),size=logits.shape[-2:],mode='nearest')[:,0].bool()
                    miou=torch.stack([torch.tensor([float((p&t).sum()/(p|t).sum().clamp_min(1)) for p in binary]) for t in gt])
                    scores=pred['instance_objectness_logits'][0,v].sigmoid()
                    active=torch.where(scores>=.5)[0].tolist()
                    pairs=[]
                    for a,b in itertools.combinations(active,2):
                        best=miou[:,[a,b]].max(0)
                        pairs.append(dict(queries=[a,b],token_cosine=float(F.cosine_similarity(token[0,v,a],token[0,v,b],dim=0)),
                            mask_pair_iou=float((binary[a]&binary[b]).sum()/(binary[a]|binary[b]).sum().clamp_min(1)),
                            duplicate_strong_mask=bool((best.values>=.5).all() and best.indices[0]==best.indices[1])))
                    rows.append(dict(key=key,train_visits=visits[key],scene=r['scene'],frame=r['frame'],dataset=r['dataset'],
                        view=v,camera=r['views'][v],role=r['physical_roles'][v],expected_instances=r['expected_instances'],
                        annotation_complete=bool(target['detection_annotation_complete'][0,v]),gt_instances=len(ids),
                        all_instance_masks_valid=bool(target['mask_valid'][0,v,ids].all()),scores=scores.tolist(),
                        object_targets=labels[0,v].tolist(),object_weights=weight[0,v].tolist(),object_logit_gradient=gradient[0,v].tolist(),
                        original_assignment=old,mask_assignment=new,assignment_changed=old!=new,
                        original_gap=assignment_gap(cost['original']),mask_gap=assignment_gap(cost['mask_augmented']),
                        costs={k:x.tolist() for k,x in cost.items()},native_grid_mask_iou=miou.tolist(),active_pairs=pairs,
                        high_score_unmatched=int(((scores>=.5)&(labels[0,v]==0)).sum())))
                del batch,target,pred,cached,images
                if (i+1)%40==0:status('auditing',records=i+1,total=356);print('AUDIT',i+1,flush=True)
            hook.remove()
            assert all(p.grad is None for p in model.parameters())
            assert flags=={n:p.requires_grad for n,p in model.named_parameters()}
            assert state_hash(tensor_state(model))==state_hash(original)
            assert encoder_hash==state_hash({n:p.detach().cpu() for n,p in model.aggregator.state_dict().items()})
            assert all(fingerprint(Path(p))==h for p,h in inputs.items())
            assert all(fingerprint(ROOT/p)==h for p,h in code.items())
            assert all(scene_digest(ROOT/p)==h for p,h in spec['source_sha256'].items())
            summaries=[]
            for count in (1,2):
                rr=[r for r in rows if r['expected_instances']==count]
                summaries.append(dict(expected_instances=count,views=len(rr),gt_instances=sum(r['gt_instances'] for r in rr),
                    changed=sum(r['assignment_changed'] for r in rr),high_score_unmatched=sum(r['high_score_unmatched'] for r in rr),
                    duplicate_strong_mask_views=sum(any(p['duplicate_strong_mask'] for p in r['active_pairs']) for r in rr)))
            changed=sum(r['assignment_changed'] for r in rows);fraction=changed/len(rows)
            report=dict(passed=True,train_inputs=356,train_scenes=40,view_observations=len(rows),summaries=summaries,
                changed_views=changed,changed_fraction=fraction,all_annotations_complete=True,
                all_negative_query_weights_positive=True,all_masks_valid=all(r['all_instance_masks_valid'] for r in rows),
                no_matching_only_training=fraction<=.01,old_cache_boxes_tokens_matches_exact=True,
                weights_flags_sources_unchanged=True,model_forwards=356,optimizer_steps=0,production_promoted=False,
                input_sha256=inputs,code_sha256=code,limitations=['Snapshot at fixed final seed17 only, not matching dynamics during training.',
                    'Repeated scene/camera across different view contexts are correlated observations.',
                    'Mask IoUs in this report are on native mask grid, not the original full-image evaluator.',
                    'Analytic scalar logit gradients are not evidence the shared decoder can separate instances.'])
            write_json(OUT/'rows.json',rows);write_json(OUT/'report.json',report)
            status('completed',records=356);print(json.dumps(report,indent=2),flush=True)
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
