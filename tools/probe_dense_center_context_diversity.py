"""Paired8 vs16 train poses for the ON context probe; fixed12800 each."""
import argparse
from collections import defaultdict
import fcntl
import json
import os
import subprocess
import sys
import time
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.nn import functional as F
from run_depth_pose_ablation import ROOT,torch,VGGTOmega,tensor_state,state_hash,fingerprint,write_json,instance_targets_from_batch
from train_cylinder_adaptive_temperature import TemperatureModel
from train_cylinder_view_balance import oriented_batch,scene_digest
from mousegeoforge.adaptive_temperature import restore_adaptive
from mousegeoforge.dense_center_probe import DenseCenterProbe,center_targets,center_loss,decode_centers,candidate_metrics
from matplotlib.patches import Rectangle
from mousegeoforge.dense_center_context import ContextCenterProbe
import copy
import gc
from vggt_omega.models.heads.instance_pose_head import _sine_position_encoding

PAIR_OUT=ROOT/'output/dense_center_context_diversity_v1'
OUT=PAIR_OUT/'off'
CONTEXT_ENABLED=True
DIVERSITY_ENABLED=False
DATA_SPEC=ROOT/'output/dual_pose_diversity_seed17_v1/protocol.json'
OLD_CONTROL=ROOT/'output/dense_center_context_budget12800_v1/on'
PARENT=ROOT/'output/cylinder_adaptive_temperature_v1'
CHECKPOINT=PARENT/'adaptive_temperature/final_heads.pt'
DEADLINE=1789194663


def run():
    started=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'output/cylinder_training.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (OUT/'protocol.json').exists():raise RuntimeError('Preserve existing probe')
        def status(state,**kw):write_json(OUT/'status.json',dict(state=state,pid=os.getpid(),elapsed_s=time.monotonic()-started,**kw))
        def budget():
            if time.monotonic()-started>2400 or time.time()>=DEADLINE:raise RuntimeError('Probe time cap')
        status('preparing')
        try:
            original_spec=json.loads((PARENT/'protocol.json').read_text())
            spec=json.loads(DATA_SPEC.read_text())
            assert spec['parent_protocol_sha256']==fingerprint(PARENT/'protocol.json')
            assert spec['evaluation']==original_spec['evaluation']
            assert spec['arms']['diversity_off']==original_spec['arms']['adaptive_temperature']
            control_keys={x[0] for x in spec['arms']['diversity_off']}
            a=[spec['records'][x[0]] for x in spec['arms']['diversity_off']]
            b=[spec['records'][x[0]] for x in spec['arms']['diversity_on']]
            assert len(a)==len(b)==800
            assert sum(x!=y for x,y in zip(a,b))==96
            assert all(x==y for x,y in zip(a,b) if x['expected_instances']==1)
            assert sum(x['expected_instances']==2 for x in a)==sum(x['expected_instances']==2 for x in b)==200
            for p,h in spec['code_sha256'].items():assert fingerprint(ROOT/p)==h
            for p,h in spec['source_sha256'].items():assert scene_digest(ROOT/p)==h
            sequence=[x[0] for x in spec['arms']['diversity_on' if DIVERSITY_ENABLED else 'diversity_off']];assert len(sequence)==800
            train=list(dict.fromkeys(sequence));dev=[e['key'] for e in spec['evaluation']]
            assert len(train)==(364 if DIVERSITY_ENABLED else 356) and len(dev)==88 and not(set(train)&set(dev))
            assert control_keys.issubset(train)
            assert len({spec['records'][k]['frame'] for k in train if spec['records'][k]['expected_instances']==2})==(16 if DIVERSITY_ENABLED else 8)
            assert all(spec['records'][k]['split']=='train' for k in train)
            assert all(spec['records'][k]['split']=='val' for k in dev)
            train_sources={spec['records'][k]['source_group'] for k in train}
            dev_sources={spec['records'][k]['source_group'] for k in dev}
            # Source groups may be shared for some synthetic replays; preserve original split.
            dual_train={spec['records'][k]['source_group'] for k in train if spec['records'][k]['expected_instances']==2}
            dual_dev={spec['records'][k]['source_group'] for k in dev if spec['records'][k]['expected_instances']==2}
            assert not dual_train&dual_dev
            code={p:fingerprint(ROOT/p) for p in ('tools/probe_dense_center_context_diversity.py','mousegeoforge/dense_center_probe.py','mousegeoforge/dense_center_context.py','tests/test_dense_center_probe.py','tests/test_dense_center_context.py')}
            inputs={str(p):fingerprint(p) for p in (CHECKPOINT,PARENT/'protocol.json',DATA_SPEC,OLD_CONTROL/'protocol.json',OLD_CONTROL/'report.json',OLD_CONTROL/'probe.pt')}
            write_json(OUT/'protocol.json',dict(code_sha256=code,input_sha256=inputs,source_sha256=spec['source_sha256'],
                diversity_enabled=DIVERSITY_ENABLED,dual_pose_count=16 if DIVERSITY_ENABLED else 8,changed_slots_per800=96 if DIVERSITY_ENABLED else 0,
                train_keys=train,evaluation=spec['evaluation'],schedule=sequence*16,seed=17,steps=12800,
                train_source_groups=sorted(train_sources),development_source_groups=sorted(dev_sources),dual_source_disjoint=True,
                feature='Frozen original shared/head projected memory minus exact sine encoding; L2 normalized per-pixel. Not rawDINO-only features.',
                context_enabled=True, registered_parameters=17413, off_dormant_neighbor_parameters=0,
                probe='256->64ReLU->depthwise3x3 identity-init->ReLU->5; Both data arms use full3x3 context. DataOFF8 vs DataON16 dual poses, same200 dual slots/800. seed17; final weights scaled.01; heatbias-2.19, offsetbias0, sizebiaslogit(.2). AdamW lr.001 wd.01 clip1; Gaussian focal alpha2 beta4 + offsetL1 +5sizeL1; no coordinate channels.',
                labels='Train GTbox centers floor into native grid, subcell sigmoid offsets, normalized sigmoid wh. Gaussian sigma=min(box grid width,height)/6 clipped[.5,2]. Explicit same-cell collision errors; report Chebyshev<=1 neighbor pairs.3x3 heatmap peaks top6; score.5 and IoU.5, unchanged box/objectness Hungarian.',
                scope='Capacity diagnostic, not instance localization OFF/ON experiment. No change to existing model, decoder, depth, assignment or deployment.',
                decision='Candidate probe qualifies for integration research only if dual score-gatedTP>original53/64, all-viewFP<=original, nondualTP>=original, duplicate views<=original, and296top bothGT have distinct found boxes. Report alltop6 boxmatched coverage separately. Failure is not absent-information proof; no dev threshold/best checkpoint selection.',
                max_run_seconds=2400,cache_ram_limit_gib=8,production_promoted=False))
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
            encoder_hash=state_hash({k:v.cpu() for k,v in model.aggregator.state_dict().items()})
            flags={n:p.requires_grad for n,p in model.named_parameters()}
            memory=[]
            def capture(module,args):memory.append(args[1].detach())
            hook=model.instance_pose_head.instance_decoder.register_forward_pre_hook(capture)
            cache={};cache_bytes=0
            try:
                for index,key in enumerate(train+dev):
                    budget();r=spec['records'][key];batch=oriented_batch(r,num_views=len(r['views']));images=batch['images'].cuda()
                    with torch.no_grad():pred=model(images,coupling='off')
                    assert len(memory)==1
                    mem=memory.pop();ph,pw=images.shape[-2]//16,images.shape[-1]//16
                    pos=_sine_position_encoding(ph,pw,256,dtype=mem.dtype,device=mem.device)
                    feature=(mem-pos).transpose(1,2).reshape(len(r['views']),256,ph,pw).contiguous().cpu()
                    t=instance_targets_from_batch(batch);assert t['instance_valid'].any()
                    boxes=t['boxes_cxcywh'][0];valid=t['instance_valid'][0]
                    target=center_targets(boxes,valid,(ph,pw))
                    item=dict(features=feature,target=target,boxes=boxes,valid=valid,record=r,
                              original_boxes=pred['instance_boxes_cxcywh'][0].cpu(),
                              original_scores=pred['instance_objectness_logits'][0].sigmoid().cpu())
                    if key in dev:
                        di=dev.index(key);saved=torch.load(PARENT/'adaptive_temperature'/f'prediction_{di:03d}.pt',map_location='cpu',weights_only=True)
                        for name,value in saved['predictions'].items():torch.testing.assert_close(pred[name].cpu(),value,atol=0,rtol=0)
                    cache[key]=item;cache_bytes+=feature.numel()*feature.element_size()
                    assert cache_bytes<=8*1024**3
                    del mem,pos,pred,batch,images,t
                    if (index+1)%32==0:status('extracting',records=index+1,total=len(train)+len(dev),cache_gib=cache_bytes/1024**3);print('FEATURES',index+1,flush=True)
            finally:hook.remove()
            assert len(cache)==len(train)+len(dev)
            write_json(OUT/'center_label_audit.json',dict(same_cell_collisions=0,
                train_neighbor_pairs=sum(cache[k]['target']['near_center_pairs'] for k in train),
                development_neighbor_pairs=sum(cache[k]['target']['near_center_pairs'] for k in dev),
                training_inputs=len(train),development_inputs=len(dev)))
            torch.manual_seed(17);probe=ContextCenterProbe(context_enabled=CONTEXT_ENABLED).cuda()
            initial_probe=state_hash({k:v.detach().cpu() for k,v in probe.state_dict().items()})
            gradient_checks=[]
            for views in (1,2,4):
                key=next(k for k in train if len(cache[k]['record']['views'])==views and
                         (views!=2 or cache[k]['record']['expected_instances']==2))
                item=cache[key];probe.zero_grad(set_to_none=True)
                companion=copy.deepcopy(probe);companion.context_enabled=not CONTEXT_ENABLED
                with torch.no_grad():
                    for name,value in probe(item['features'].cuda()).items():
                        torch.testing.assert_close(value,companion(item['features'].cuda())[name],atol=0,rtol=0)
                del companion
                loss,_=center_loss(probe(item['features'].cuda()),{k:v.cuda() if torch.is_tensor(v) else v for k,v in item['target'].items()})
                loss.backward();norms={n:float(p.grad.norm()) for n,p in probe.named_parameters()}
                assert all(torch.isfinite(p.grad).all() and p.grad.norm()>0 for p in probe.parameters())
                gradient_checks.append(dict(views=views,key=key,loss=float(loss.detach()),gradient_norms=norms))
            probe.zero_grad(set_to_none=True)
            assert state_hash({k:v.detach().cpu() for k,v in probe.state_dict().items()})==initial_probe
            assert all(p.grad is None for p in model.parameters())
            write_json(OUT/'gradient_preflight.json',dict(passed=True,optimizer_steps=0,paired_initial_prediction_difference=0,initial_probe_sha256=initial_probe,checks=gradient_checks))
            optimizer=torch.optim.AdamW(probe.parameters(),lr=.001,weight_decay=.01)
            history=[]
            for step,key in enumerate(sequence*16,1):
                budget();item=cache[key];optimizer.zero_grad(set_to_none=True)
                logits=probe(item['features'].cuda());loss,parts=center_loss(logits,{k:v.cuda() if torch.is_tensor(v) else v for k,v in item['target'].items()})
                assert torch.isfinite(loss);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(probe.parameters(),1.,error_if_nonfinite=True);optimizer.step()
                history.append(dict(step=step,key=key,loss=float(loss.detach()),gradient_norm=float(norm),**{k:float(v.detach()) for k,v in parts.items()}))
                if step%100==0:write_json(OUT/'history.json',history);status('training_probe',step=step,steps=12800);print('STEP',step,history[-1]['loss'],flush=True)
            torch.save(dict(probe={k:v.cpu() for k,v in probe.state_dict().items()},optimizer=optimizer.state_dict(),steps=12800,
                            protocol_sha256=fingerprint(OUT/'protocol.json')),OUT/'probe.pt')
            neighbors=probe.context.weight.detach()*(1-probe.center_mask)
            assert bool(neighbors.abs().sum()>0)==CONTEXT_ENABLED
            write_json(OUT/'context_parameter_audit.json',dict(context_enabled=CONTEXT_ENABLED,neighbor_l2=float(neighbors.norm()),registered_parameters=sum(p.numel() for p in probe.parameters())))
            shared_train_counts=dict(tp=0,fp=0,fn=0,gt=0,all_topk_covered=0)
            dual_train_counts=dict(shared_train_counts)
            records=[];visuals={};totals={tag:defaultdict(lambda:dict(tp=0,fp=0,fn=0,gt=0,all_topk_covered=0,duplicate_views=0)) for tag in ('original','candidate')}
            for index,e in enumerate(spec['evaluation']):
                budget();item=cache[e['key']];r=item['record']
                with torch.no_grad():decoded={k:v.cpu() for k,v in decode_centers(probe(item['features'].cuda())).items()}
                if not DIVERSITY_ENABLED:
                    prior=torch.load(OLD_CONTROL/f'candidates_{index:03d}.pt',map_location='cpu',weights_only=True)
                    for name,value in decoded.items():torch.testing.assert_close(value,prior['decoded'][name],atol=0,rtol=0)
                for view in range(len(r['views'])):
                    gt=item['boxes'][view,item['valid'][view]];per={}
                    for tag,boxes,scores in (('original',item['original_boxes'][view],item['original_scores'][view]),
                                            ('candidate',decoded['boxes'][view],decoded['scores'][view])):
                        result=candidate_metrics(boxes,scores,gt);per[tag]=result
                        for key in ('tp','fp','fn','gt','all_topk_covered'):totals[tag][e['group']][key]+=result[key]
                        totals[tag][e['group']]['duplicate_views']+=int(result['duplicate'])
                    records.append(dict(index=index,group=e['group'],frame=r['frame'],view=r['views'][view],role=r['physical_roles'][view],metrics=per))
                saved=dict(record=r,decoded=decoded,boxes=item['boxes'],valid=item['valid'],
                           original_boxes=item['original_boxes'],original_scores=item['original_scores'])
                torch.save(saved,OUT/f'candidates_{index:03d}.pt')
                if index in (0,7,9,13):visuals[index]=saved
            summary={tag:dict(groups) for tag,groups in totals.items()}
            difficult=next(r for r in records if r['group']=='old_dual_pair' and r['frame']==296 and r['role']=='top')
            def total(tag,key,dual=False):return sum(v[key] for g,v in summary[tag].items() if not dual or g!='old_dual_pair')
            assert summary['original']['old_dual_pair']['tp']==53
            criteria=dict(dual_tp_gain=summary['candidate']['old_dual_pair']['tp']>53,
                          fp_not_higher=total('candidate','fp')<=total('original','fp'),
                          nondual_tp_not_lower=total('candidate','tp',True)>=total('original','tp',True),
                          duplicates_not_higher=total('candidate','duplicate_views')<=total('original','duplicate_views'),
                          frame296_two_found=difficult['metrics']['candidate']['tp']==2)
            eligible=all(criteria.values())
            train_counts=dict(tp=0,fp=0,fn=0,gt=0,all_topk_covered=0)
            for key in train:
                budget();item=cache[key]
                with torch.no_grad():decoded={k:v.cpu() for k,v in decode_centers(probe(item['features'].cuda())).items()}
                for view in range(len(item['record']['views'])):
                    result=candidate_metrics(decoded['boxes'][view],decoded['scores'][view],item['boxes'][view,item['valid'][view]])
                    for name in train_counts:
                        train_counts[name]+=result[name]
                        if key in control_keys:shared_train_counts[name]+=result[name]
                        if item['record']['expected_instances']==2:dual_train_counts[name]+=result[name]
            assert state_hash(tensor_state(model))==state_hash(original)
            assert encoder_hash==state_hash({k:v.cpu() for k,v in model.aggregator.state_dict().items()})
            assert flags=={n:p.requires_grad for n,p in model.named_parameters()}
            assert all(p.grad is None for p in model.parameters())
            assert not model.instance_pose_head.instance_decoder._forward_pre_hooks
            assert all(fingerprint(ROOT/p)==h for p,h in code.items())
            for p,h in inputs.items():assert fingerprint(__import__('pathlib').Path(p))==h
            fig,axes=plt.subplots(4,3,figsize=(13,11),layout='constrained')
            for row,index in enumerate((0,7,9,13)):
                item=visuals[index];r=item['record'];view=r['physical_roles'].index('top');h,w=r['image_hw'];scale=np.array([w,h])
                rgb=Image.open(ROOT/r['scene']/'blended_images'/f'{r["views"][view]:08d}.jpg').resize((w,h))
                for col,(boxes,scores,title) in enumerate(((item['original_boxes'][view],item['original_scores'][view],'Original queries'),
                                                           (item['decoded']['boxes'][view],item['decoded']['scores'][view],'Dense center candidates'))):
                    ax=axes[row,col];ax.imshow(rgb)
                    for b in item['boxes'][view,item['valid'][view]].numpy():
                        ax.add_patch(Rectangle((b[:2]-b[2:]/2)*scale,*(b[2:]*scale),fill=False,edgecolor='lime'))
                    for q in torch.where(scores>=.5)[0]:
                        b=boxes[q].numpy();ax.add_patch(Rectangle((b[:2]-b[2:]/2)*scale,*(b[2:]*scale),fill=False,edgecolor='cyan'))
                        ax.text(b[0]*w,b[1]*h,f'{int(q)}:{float(scores[q]):.2f}',color='cyan',fontsize=8)
                    ax.set_title(f'Frame{r["frame"]}: {title}');ax.axis('off')
                axes[row,2].imshow(item['decoded']['heatmap'][view],vmin=0,vmax=1,cmap='magma')
                axes[row,2].set_title('Center heatmap, no GT peak selection');axes[row,2].axis('off')
            fig.savefig(OUT/'fixed_candidates.png',dpi=140);plt.close(fig)
            write_json(OUT/'report.json',dict(passed=True,diagnostic_only=True,context_enabled=CONTEXT_ENABLED,diversity_enabled=DIVERSITY_ENABLED,steps=12800,parameters=sum(p.numel() for p in probe.parameters()),
                feature_records=len(cache),cache_gib=cache_bytes/1024**3,original_model_unchanged=True,original_eval_prediction_max_difference=0,
                summary=summary,training_summary=train_counts,shared_training_summary=shared_train_counts,dual_training_summary=dual_train_counts,control_reproduction_max_difference=None if DIVERSITY_ENABLED else 0,records=records,frame296_top=difficult,criteria=criteria,proposal_route_supported=eligible,
                limitations=['Standalone box candidate capacity probe is not end-to-end poseOFF/ON or proof of keypoint benefit.',
                             'Alltop6 coverage ignores score threshold but uses same box/objectness assignment, not deployment recall.',
                             'Single center per cell and3x3 local peak filter can merge close centers; label audit reported.',
                             'Reused synthetic development; no realCylinder test.'],
                code_sha256=code,production_promoted=False))
            status('completed');print('PASSED probe capacity route',eligible,flush=True)
        except BaseException as error:status('failed',error=repr(error));raise


def run_pair():
    global OUT,CONTEXT_ENABLED,DIVERSITY_ENABLED
    PAIR_OUT.mkdir(parents=True,exist_ok=True)
    write_json(PAIR_OUT/'status.json',dict(state='running',pid=os.getpid(),arm='off'))
    try:
        for arm in ('off','on'):
            OUT=PAIR_OUT/arm;CONTEXT_ENABLED=True;DIVERSITY_ENABLED=arm=='on'
            write_json(PAIR_OUT/'status.json',dict(state='running',pid=os.getpid(),arm=arm))
            run()
            gc.collect();torch.cuda.empty_cache()
        off=json.loads((PAIR_OUT/'off/report.json').read_text())
        on=json.loads((PAIR_OUT/'on/report.json').read_text())
        for arm in ('off','on'):
            pre=json.loads((PAIR_OUT/arm/'gradient_preflight.json').read_text())
            assert pre['paired_initial_prediction_difference']==0
        a=json.loads((PAIR_OUT/'off/gradient_preflight.json').read_text())
        b=json.loads((PAIR_OUT/'on/gradient_preflight.json').read_text())
        assert a['initial_probe_sha256']==b['initial_probe_sha256']
        pa=json.loads((PAIR_OUT/'off/protocol.json').read_text())
        pb=json.loads((PAIR_OUT/'on/protocol.json').read_text())
        for name in ('evaluation','source_sha256','input_sha256','code_sha256'):
            assert pa[name]==pb[name]
        assert off['control_reproduction_max_difference']==0
        write_json(PAIR_OUT/'comparison.json',dict(control_reproduction_max_difference=0,comparison='8vs16 dual train poses, both contextON',passed=True,steps_each=12800,seed=17,
            paired_initial_state_equal=True,original_model_unchanged=True,
            dual_tp_delta=on['summary']['candidate']['old_dual_pair']['tp']-off['summary']['candidate']['old_dual_pair']['tp'],
            off=off['summary']['candidate'],on=on['summary']['candidate'],
            off_train=off['training_summary'],on_train=on['training_summary'],
            off_shared_train=off['shared_training_summary'],on_shared_train=on['shared_training_summary'],
            off_dual_train=off['dual_training_summary'],on_dual_train=on['dual_training_summary'],
            on_qualifies_for_integration=on['proposal_route_supported'],
            limitations=['Both full-context heads identical; dataOFF8 vs dataON16 dual train poses with unchanged200 visits/800.',
                         'Frozen-feature capacity diagnostic, not end-to-end pose gain.',
                         'Reused synthetic dev, no realCylinder.'],production_promoted=False))
        write_json(PAIR_OUT/'status.json',dict(state='completed',pid=os.getpid()))
    except BaseException as error:
        write_json(PAIR_OUT/'status.json',dict(state='failed',pid=os.getpid(),error=repr(error)))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--launch',action='store_true');args=parser.parse_args()
    if args.launch:
        if (PAIR_OUT/'status.json').exists() or any((PAIR_OUT/arm/'protocol.json').exists() for arm in ('off','on')):raise RuntimeError('Already launched')
        PAIR_OUT.mkdir(parents=True,exist_ok=True)
        with (PAIR_OUT/'run.log').open('a') as log:
            p=subprocess.Popen([sys.executable,'-u',__file__],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(pid=p.pid,output=str(PAIR_OUT))))
    else:run_pair()
