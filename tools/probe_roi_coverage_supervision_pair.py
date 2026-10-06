"""Fixed union12 scoring: static source-score labels versus coverage-preserving labels."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from torch.nn import functional as F
from run_depth_pose_ablation import ROOT,torch,VGGTOmega,tensor_state,state_hash,fingerprint,write_json,instance_targets_from_batch
from train_cylinder_adaptive_temperature import TemperatureModel
from train_cylinder_view_balance import oriented_batch,scene_digest
from mousegeoforge.adaptive_temperature import restore_adaptive
from mousegeoforge.dense_center_context import ContextCenterProbe
from mousegeoforge.dense_center_probe import decode_centers,candidate_metrics
from mousegeoforge.pilot_metrics import box_iou
from mousegeoforge.roi_candidate_quality import RegionQualityProbe,region_features,apply_quality_scores
from vggt_omega.models.heads.instance_pose_head import _sine_position_encoding
from mousegeoforge.roi_set_objectness import SetObjectnessProbe,set_objectness_targets
import gc
from mousegeoforge.candidate_pool import merge_candidate_pools
from mousegeoforge.coverage_objectness import coverage_objectness_targets

PAIR_OUT=ROOT/'output/roi_coverage_supervision_pair_v1'
OUT=PAIR_OUT/'off'
RELATION_ENABLED=True
POOL_ENABLED=True
SUPERVISION_ENABLED=False
QUALITY_SOURCE=ROOT/'output/roi_set_objectness_pair_v1/on'
SOURCE=ROOT/'output/dense_center_context_diversity_v1/on'
BASE=ROOT/'output/cylinder_adaptive_temperature_v1'
DATA=ROOT/'output/dual_pose_diversity_seed17_v1/protocol.json'
DEADLINE=1789194663


def run():
    started=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    def status(state,**kw):write_json(OUT/'status.json',dict(state=state,pid=os.getpid(),elapsed_s=time.monotonic()-started,**kw))
    def budget():
        if time.monotonic()-started>2400 or time.time()>=DEADLINE:raise RuntimeError('Time cap')
    with (ROOT/'output/cylinder_training.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (OUT/'protocol.json').exists():raise RuntimeError('Preserve existing experiment')
        status('preparing')
        try:
            spec=json.loads(DATA.read_text());source=json.loads((SOURCE/'report.json').read_text());assert source['passed']
            for p,h in source['code_sha256'].items():assert fingerprint(ROOT/p)==h
            for p,h in spec['source_sha256'].items():assert scene_digest(ROOT/p)==h
            sequence=[x[0] for x in spec['arms']['diversity_on']];train=list(dict.fromkeys(sequence));dev=[e['key'] for e in spec['evaluation']]
            assert len(sequence)==800 and len(train)==364 and len(dev)==88 and not set(train)&set(dev)
            assert all(spec['records'][k]['split']=='train' for k in train)
            assert not {spec['records'][k]['source_group'] for k in train if spec['records'][k]['expected_instances']==2}&{spec['records'][k]['source_group'] for k in dev if spec['records'][k]['expected_instances']==2}
            for p,h in json.loads((QUALITY_SOURCE/'report.json').read_text())['code_sha256'].items():assert fingerprint(ROOT/p)==h
            paths=(DATA,SOURCE/'probe.pt',SOURCE/'protocol.json',SOURCE/'report.json',BASE/'adaptive_temperature/final_heads.pt',QUALITY_SOURCE/'quality.pt',QUALITY_SOURCE/'protocol.json',QUALITY_SOURCE/'report.json')
            paths=paths+(ROOT/'output/roi_candidate_pool_pair_v1/on/quality.pt',ROOT/'output/roi_pool_supervision_audit_v1/report.json')
            inputs={str(p):fingerprint(p) for p in paths}
            code={p:fingerprint(ROOT/p) for p in ('tools/probe_roi_coverage_supervision_pair.py','mousegeoforge/candidate_pool.py','tests/test_candidate_pool.py','mousegeoforge/roi_set_objectness.py','tests/test_roi_set_objectness.py','mousegeoforge/roi_candidate_quality.py','mousegeoforge/dense_center_context.py','mousegeoforge/dense_center_probe.py','tests/test_roi_candidate_quality.py')}
            code.update({p:fingerprint(ROOT/p) for p in ('mousegeoforge/coverage_objectness.py','tests/test_coverage_objectness.py','mousegeoforge/candidate_ceiling.py')})
            write_json(OUT/'protocol.json',dict(input_sha256=inputs,code_sha256=code,source_sha256=spec['source_sha256'],
                seed=17,steps=3200,coverage_supervision=SUPERVISION_ENABLED,pool_enabled=True,candidates_per_view=12,
                schedule=sequence*4,train_keys=train,evaluation=spec['evaluation'],fresh_optimizer=True,
                head='Both identical stage35 relationalON head49985 params, fixed union12; same meanBCE lr.001/wd.01/clip1.',
                objective='OFF fixed5L1-sourceScore match then IoU>=.5 label; ON maximum one-to-one IoU>=.5 coverage first then IoU tie-break. Only training target assignment changes, not inference/metrics.',
                decision='Relative dual/nondual TP nondecreasing, FP/duplicate not higher and dual gain>=4 or FPdrop>=25%. Absolute original retention gates unchanged.',
                limitations='Frozen candidates. GT only supervision/audit. Not end-to-end pose improvement. Same repeatedly used88 development.',
                max_seconds=2400,production_promoted=False))
            torch.set_num_threads(2);torch.manual_seed(17)
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
            encoder=ROOT/'vggt-omega/vggt_omega_1b_512.pt';assert fingerprint(encoder)==spec['encoder_sha256']
            spatial=VGGTOmega(enable_camera=False,enable_depth=True,enable_multi_instance_pose=True,num_keypoints=23)
            weights=torch.load(encoder,weights_only=True,mmap=True)
            loaded=spatial.load_state_dict({k:v for k,v in weights.items() if k.startswith(('aggregator.','dense_head.'))},strict=False)
            assert not loaded.unexpected_keys and all(k.startswith('instance_pose_head.') for k in loaded.missing_keys)
            del weights
            model=TemperatureModel.from_spatial_model(spatial).cuda().eval();del spatial
            original=torch.load(BASE/'adaptive_temperature/final_heads.pt',map_location='cpu',weights_only=True)['model'];restore_adaptive(model,original)
            enc_hash=state_hash({k:v.cpu() for k,v in model.aggregator.state_dict().items()});flags={n:p.requires_grad for n,p in model.named_parameters()}
            center=ContextCenterProbe(context_enabled=True).cuda().eval()
            center_state=torch.load(SOURCE/'probe.pt',map_location='cpu',weights_only=True)['probe'];center.load_state_dict(center_state)
            memory=[];hook=model.instance_pose_head.instance_decoder.register_forward_pre_hook(lambda m,args:memory.append(args[1].detach()))
            cache={}
            try:
                for index,key in enumerate(train+dev):
                    budget();r=spec['records'][key];batch=oriented_batch(r,num_views=len(r['views']));images=batch['images'].cuda()
                    with torch.no_grad():
                        pred=model(images,coupling='off');assert len(memory)==1
                        mem=memory.pop();gh,gw=images.shape[-2]//16,images.shape[-1]//16
                        feature=(mem-_sine_position_encoding(gh,gw,256,dtype=mem.dtype,device=mem.device)).transpose(1,2).reshape(len(r['views']),256,gh,gw)
                        d=decode_centers(center(feature));descriptor=region_features(feature,d['boxes'],d['scores']).cpu()
                    d={k:v.cpu() for k,v in d.items()};t=instance_targets_from_batch(batch);boxes=t['boxes_cxcywh'][0];valid=t['instance_valid'][0]
                    quality=set_objectness_targets(d['boxes'],d['scores'],boxes,valid)
                    assert torch.isfinite(descriptor).all() and torch.isfinite(quality).all() and (quality>=0).all() and (quality<=1).all()
                    if key in dev:
                        di=dev.index(key);prior=torch.load(SOURCE/f'candidates_{di:03d}.pt',map_location='cpu',weights_only=True)
                        for name,value in d.items():torch.testing.assert_close(value,prior['decoded'][name],atol=0,rtol=0)
                        old=torch.load(BASE/'adaptive_temperature'/f'prediction_{di:03d}.pt',map_location='cpu',weights_only=True)
                        for name,value in old['predictions'].items():torch.testing.assert_close(pred[name].cpu(),value,atol=0,rtol=0)
                    center_descriptor=descriptor
                    union_boxes,union_scores=merge_candidate_pools(pred['instance_boxes_cxcywh'][0].cpu(),pred['instance_objectness_logits'][0].sigmoid().cpu(),d['boxes'],d['scores'],POOL_ENABLED)
                    with torch.no_grad():descriptor=region_features(feature,union_boxes.cuda(),union_scores.cuda()).cpu()
                    d=dict(d,boxes=union_boxes,scores=union_scores)
                    quality=set_objectness_targets(d['boxes'],d['scores'],boxes,valid)
                    if SUPERVISION_ENABLED:quality=coverage_objectness_targets(d['boxes'],boxes,valid)
                    cache[key]=dict(features=descriptor,center_features=center_descriptor,target=quality,decoded=d,boxes=boxes,valid=valid,record=r)
                    del batch,images,mem,feature,pred,t
                    if (index+1)%32==0:status('extracting',records=index+1,total=452);print('FEATURES',index+1,flush=True)
            finally:hook.remove()
            positive_sources=dict(original=0,center=0)
            for key in train:
                labels=cache[key]['target']
                positive_sources['center']+=int(labels[:,-6:].sum())
                if POOL_ENABLED:positive_sources['original']+=int(labels[:,:6].sum())
            write_json(OUT/'source_label_audit.json',dict(positive_sources=positive_sources,pool_enabled=POOL_ENABLED,training_inputs=len(train)))
            if POOL_ENABLED:assert min(positive_sources.values())>0, 'A source receives no positive supervision'
            torch.manual_seed(17);head=SetObjectnessProbe(relation_enabled=RELATION_ENABLED).cuda()
            old_head=torch.load(QUALITY_SOURCE/'quality.pt',map_location='cpu',weights_only=True)['head']
            loaded=head.load_state_dict(old_head,strict=False)
            assert not loaded.unexpected_keys and not loaded.missing_keys
            initial_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()};initial=state_hash(initial_state);checks=[];initial_on_difference=0.
            for index,key in enumerate(dev):
                saved=torch.load(QUALITY_SOURCE/f'quality_{index:03d}.pt',map_location='cpu',weights_only=True)
                head.relation_enabled=True
                with torch.no_grad():baseline_score=head(cache[key]['center_features'].cuda()).sigmoid().cpu()
                torch.testing.assert_close(baseline_score,saved['on_scores'],atol=0,rtol=0)
                head.relation_enabled=RELATION_ENABLED
                with torch.no_grad():current_score=head(cache[key]['features'].cuda()).sigmoid().cpu()
                initial_on_difference=max(initial_on_difference,float((current_score[:,-6:]-baseline_score).abs().max()))
            for views in (1,2,4):
                item=next(cache[k] for k in train if len(cache[k]['record']['views'])==views)
                head.zero_grad(set_to_none=True);loss=F.binary_cross_entropy_with_logits(head(item['features'].cuda()),item['target'].cuda());loss.backward()
                assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0 for p in head.parameters() if p.requires_grad)
                assert apply_quality_scores(None,None,item['decoded']['scores'],False) is item['decoded']['scores']
                checks.append(dict(views=views,loss=float(loss.detach()),gradients={n:float(p.grad.norm()) for n,p in head.named_parameters() if p.requires_grad}))
            head.zero_grad(set_to_none=True);assert state_hash({k:v.detach().cpu() for k,v in head.state_dict().items()})==initial
            write_json(OUT/'gradient_preflight.json',dict(passed=True,checks=checks,original_and_center88_output_difference=0,off_score_difference=0,warmstart_quality88_difference=0,initial_on_difference=initial_on_difference,initial_head_sha256=initial))
            opt=torch.optim.AdamW([p for p in head.parameters() if p.requires_grad],lr=.001,weight_decay=.01);history=[]
            for step,key in enumerate(sequence*4,1):
                budget();item=cache[key];opt.zero_grad(set_to_none=True)
                loss=F.binary_cross_entropy_with_logits(head(item['features'].cuda()),item['target'].cuda());assert torch.isfinite(loss);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True);opt.step();history.append(dict(step=step,loss=float(loss.detach()),gradient_norm=float(norm)))
                if step%100==0:write_json(OUT/'history.json',history);status('training',step=step,steps=3200);print('STEP',step,history[-1]['loss'],flush=True)
            torch.save(dict(head={k:v.detach().cpu() for k,v in head.state_dict().items()},optimizer=opt.state_dict(),steps=3200,protocol_sha256=fingerprint(OUT/'protocol.json')),OUT/'quality.pt')
            delta={n:float((v.detach().cpu()-initial_state[n]).norm()) for n,v in head.state_dict().items() if n.startswith(('norm.','attention.'))}
            assert (sum(delta.values())>0)==RELATION_ENABLED
            write_json(OUT/'relation_parameter_audit.json',dict(relation_enabled=RELATION_ENABLED,delta_l2=delta))
            summary={arm:defaultdict(lambda:dict(tp=0,fp=0,fn=0,gt=0,all_topk_covered=0,duplicate_views=0)) for arm in ('off','on')};records=[];visuals=[]
            quality_error={split:dict(absolute_sum=0.,count=0) for split in ('train','dev')}
            for index,key in enumerate(dev+train):
                budget();item=cache[key];r=item['record'];d=item['decoded']
                with torch.no_grad():scores=head(item['features'].cuda()).sigmoid().cpu()
                if not SUPERVISION_ENABLED and index<len(dev):
                    prior=torch.load(ROOT/'output/roi_candidate_pool_pair_v1/on'/f'quality_{index:03d}.pt',map_location='cpu',weights_only=True)
                    torch.testing.assert_close(scores,prior['on_scores'],atol=0,rtol=0)
                error=quality_error['dev' if index<len(dev) else 'train'];error['absolute_sum']+=float((scores-item['target']).abs().sum());error['count']+=scores.numel()
                group=spec['evaluation'][index]['group'] if index<len(dev) else 'train_dual' if r['expected_instances']==2 else 'train_single'
                for v,role in enumerate(r['physical_roles']):
                    per={}
                    for arm,score in (('off',d['scores'][v]),('on',scores[v])):
                        m=candidate_metrics(d['boxes'][v],score,item['boxes'][v,item['valid'][v]]);per[arm]=m
                        for name in ('tp','fp','fn','gt','all_topk_covered'):summary[arm][group][name]+=m[name]
                        summary[arm][group]['duplicate_views']+=int(m['duplicate'])
                    records.append(dict(index=index,group=group,frame=r['frame'],role=role,view=r['views'][v],metrics=per))
                if index<len(dev):torch.save(dict(record=r,boxes=d['boxes'],off_scores=d['scores'],on_scores=scores,gt_boxes=item['boxes'],valid=item['valid']),OUT/f'quality_{index:03d}.pt')
                if index in (0,7,9,13):visuals.append((item,scores))
            summary={k:dict(v) for k,v in summary.items()}
            assert POOL_ENABLED or (summary['off']['old_dual_pair']['tp']==30 and summary['off']['old_dual_pair']['fp']==3)
            def total(arm,name,nondual=False):return sum(x[name] for g,x in summary[arm].items() if not g.startswith('train_') and (not nondual or g!='old_dual_pair'))
            difficult=next(r for r in records if r['group']=='old_dual_pair' and r['frame']==296 and r['role']=='top')
            criteria=dict(dual_gain=summary['on']['old_dual_pair']['tp']>30,fp_not_higher=total('on','fp')<=total('off','fp'),nondual_not_lower=total('on','tp',True)>=total('off','tp',True),duplicates_not_higher=total('on','duplicate_views')<=total('off','duplicate_views'))
            integration=summary['on']['old_dual_pair']['tp']>53 and total('on','fp')<=47 and total('on','tp',True)>=124 and total('on','duplicate_views')<=3 and difficult['metrics']['on']['tp']==2
            assert state_hash(tensor_state(model))==state_hash(original) and enc_hash==state_hash({k:v.cpu() for k,v in model.aggregator.state_dict().items()})
            assert flags=={n:p.requires_grad for n,p in model.named_parameters()}
            assert state_hash({k:v.detach().cpu() for k,v in center.state_dict().items()})==state_hash(center_state)
            assert all(p.grad is None for p in list(model.parameters())+list(center.parameters()))
            for p,h in inputs.items():assert fingerprint(__import__('pathlib').Path(p))==h
            for p,h in code.items():assert fingerprint(ROOT/p)==h
            fig,axes=plt.subplots(4,2,figsize=(11,12),layout='constrained')
            for row,(item,scores) in enumerate(visuals):
                r=item['record'];v=r['physical_roles'].index('top');h,w=r['image_hw'];scale=np.array([w,h]);rgb=Image.open(ROOT/r['scene']/'blended_images'/f'{r["views"][v]:08d}.jpg').resize((w,h))
                for col,s in enumerate((item['decoded']['scores'][v],scores[v])):
                    ax=axes[row,col];ax.imshow(rgb)
                    for b in item['boxes'][v,item['valid'][v]].numpy():ax.add_patch(Rectangle((b[:2]-b[2:]/2)*scale,*(b[2:]*scale),fill=False,edgecolor='lime'))
                    for q in torch.where(s>=.5)[0]:
                        b=item['decoded']['boxes'][v,q].numpy();ax.add_patch(Rectangle((b[:2]-b[2:]/2)*scale,*(b[2:]*scale),fill=False,edgecolor='cyan'));ax.text(b[0]*w,b[1]*h,f'{int(q)}:{float(s[q]):.2f}',color='cyan')
                    ax.set_title(f'Frame{r["frame"]}: '+('Input union scores' if col==0 else 'Unique ROI objectness, boxes unchanged'));ax.axis('off')
            fig.savefig(OUT/'fixed_quality.png',dpi=120);plt.close(fig)
            write_json(OUT/'report.json',dict(passed=True,diagnostic_only=True,parameters=sum(p.numel() for p in head.parameters()),optimized_parameters=sum(p.numel() for p in head.parameters() if p.requires_grad),relation_enabled=True,pool_enabled=POOL_ENABLED,steps=3200,objectness_mae={k:v['absolute_sum']/v['count'] for k,v in quality_error.items()},summary=summary,records=records,frame296_top=difficult,criteria=criteria,region_quality_supported=all(criteria.values()),integration_qualified=bool(integration),original_model_unchanged=True,center_probe_unchanged=True,boxes_unchanged=True,code_sha256=code,production_promoted=False))
            status('completed');print('COMPLETE',criteria,'integration',integration,flush=True)
        except BaseException as error:status('failed',error=repr(error));raise


def run_pair():
    global OUT,RELATION_ENABLED,POOL_ENABLED,SUPERVISION_ENABLED
    PAIR_OUT.mkdir(parents=True,exist_ok=True)
    try:
        for arm in ('off','on'):
            OUT=PAIR_OUT/arm;RELATION_ENABLED=True;POOL_ENABLED=True;SUPERVISION_ENABLED=arm=='on'
            write_json(PAIR_OUT/'status.json',dict(state='running',pid=os.getpid(),arm=arm))
            run();gc.collect();torch.cuda.empty_cache()
        reports={arm:json.loads((PAIR_OUT/arm/'report.json').read_text()) for arm in ('off','on')}
        protocol={arm:json.loads((PAIR_OUT/arm/'protocol.json').read_text()) for arm in ('off','on')}
        for name in ('schedule','train_keys','evaluation','input_sha256','source_sha256','code_sha256'):
            assert protocol['off'][name]==protocol['on'][name]
        pre={arm:json.loads((PAIR_OUT/arm/'gradient_preflight.json').read_text()) for arm in ('off','on')}
        assert pre['off']['initial_head_sha256']==pre['on']['initial_head_sha256']
        assert pre['off']['initial_on_difference']==pre['on']['initial_on_difference']
        sums={arm:reports[arm]['summary']['on'] for arm in ('off','on')}
        def total(arm,name,nondual=False):
            return sum(v[name] for g,v in sums[arm].items() if not g.startswith('train_') and (not nondual or g!='old_dual_pair'))
        off_tp=sums['off']['old_dual_pair']['tp'];on_tp=sums['on']['old_dual_pair']['tp']
        criteria=dict(dual_not_lower=on_tp>=off_tp,nondual_not_lower=total('on','tp',True)>=total('off','tp',True),
            fp_not_higher=total('on','fp')<=total('off','fp'),duplicates_not_higher=total('on','duplicate_views')<=total('off','duplicate_views'),
            meaningful=on_tp-off_tp>=4 or (total('off','fp')>0 and total('on','fp')<=.75*total('off','fp')))
        write_json(PAIR_OUT/'comparison.json',dict(passed=True,steps_each=3200,initial_common_state_equal=True,
            off=sums['off'],on=sums['on'],criteria=criteria,coverage_supported=all(criteria.values()),control_reproduces_stage38_union88=True,
            integration_qualified=reports['on']['integration_qualified'],original_model_and_boxes_unchanged=True,
            limitations=['Identical candidate tokens remain symmetric; learned scoring not guaranteed uniqueness.',
                         'Same union12, relational architecture and meanBCE; only GT training assignment changes.',
                         'Frozen candidate diagnostic, not end-to-end pose improvement.'],production_promoted=False))
        write_json(PAIR_OUT/'status.json',dict(state='completed',pid=os.getpid()))
    except BaseException as error:
        write_json(PAIR_OUT/'status.json',dict(state='failed',pid=os.getpid(),error=repr(error)));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--launch',action='store_true');args=parser.parse_args()
    if args.launch:
        if (PAIR_OUT/'status.json').exists():raise RuntimeError('Already launched')
        PAIR_OUT.mkdir(parents=True,exist_ok=True)
        with (PAIR_OUT/'run.log').open('a') as log:p=subprocess.Popen([sys.executable,'-u',__file__],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(pid=p.pid,output=str(PAIR_OUT))))
    else:run_pair()


