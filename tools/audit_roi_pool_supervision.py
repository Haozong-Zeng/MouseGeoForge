"""Zero-optimization train/dev union-pool static-label and score audit."""
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

PAIR_OUT=ROOT/'output/roi_pool_supervision_audit_v1'
OUT=PAIR_OUT
RELATION_ENABLED=True
POOL_ENABLED=True
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
            paths=paths+tuple(ROOT/'output'/folder/'on'/name for folder in ('roi_candidate_pool_pair_v1','roi_candidate_pool_budget6400_v1') for name in ('quality.pt','report.json','protocol.json'))
            inputs={str(p):fingerprint(p) for p in paths}
            code={p:fingerprint(ROOT/p) for p in ('tools/audit_roi_pool_supervision.py','mousegeoforge/candidate_pool.py','tests/test_candidate_pool.py','mousegeoforge/roi_set_objectness.py','tests/test_roi_set_objectness.py','mousegeoforge/roi_candidate_quality.py','mousegeoforge/dense_center_context.py','mousegeoforge/dense_center_probe.py','tests/test_roi_candidate_quality.py')}
            code['mousegeoforge/candidate_ceiling.py']=fingerprint(ROOT/'mousegeoforge/candidate_ceiling.py')
            write_json(OUT/'protocol.json',dict(input_sha256=inputs,code_sha256=code,model_forwards=452,optimizer_steps=0,training_gt_diagnostic_only=True,production_promoted=False))
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
            from mousegeoforge.candidate_ceiling import geometric_ceiling
            scorers={}
            for step,folder in ((3200,'roi_candidate_pool_pair_v1'),(6400,'roi_candidate_pool_budget6400_v1')):
                head=SetObjectnessProbe(relation_enabled=True).cuda().eval()
                head.load_state_dict(torch.load(ROOT/'output'/folder/'on/quality.pt',map_location='cpu',weights_only=True)['head'])
                scorers[step]=head
            groups=defaultdict(lambda:dict(gt=0,positive_labels=0,geometric_capacity=0,stranded_targets=0,
                positive_count=0,negative_count=0,positive_score3200=0.,positive_score6400=0.,negative_score3200=0.,negative_score6400=0.,
                positive_below3200=0,positive_below6400=0))
            rows=[]
            for key in train+dev:
                budget();item=cache[key];record=item['record'];split='train' if key in train else 'dev'
                group=split+('_dual' if record['expected_instances']==2 else '_single')
                scores={}
                with torch.no_grad():
                    for step,head in scorers.items():scores[step]=head(item['features'].cuda()).sigmoid().cpu()
                if split=='dev':
                    index=dev.index(key)
                    for step,folder in ((3200,'roi_candidate_pool_pair_v1'),(6400,'roi_candidate_pool_budget6400_v1')):
                        saved=torch.load(ROOT/'output'/folder/'on'/f'quality_{index:03d}.pt',map_location='cpu',weights_only=True)
                        torch.testing.assert_close(scores[step],saved['on_scores'],atol=0,rtol=0)
                for v,role in enumerate(record['physical_roles']):
                    gt=item['boxes'][v,item['valid'][v]];labels=item['target'][v].bool()
                    bound=geometric_ceiling(item['decoded']['boxes'][v],gt)
                    actual=candidate_metrics(item['decoded']['boxes'][v],item['decoded']['scores'][v],gt)
                    g=groups[group];g['gt']+=len(gt);g['positive_labels']+=int(labels.sum());g['geometric_capacity']+=bound['covered']
                    g['stranded_targets']+=bound['covered']-int(labels.sum())
                    g['positive_count']+=int(labels.sum());g['negative_count']+=int((~labels).sum())
                    for step in scorers:
                        g[f'positive_score{step}']+=float(scores[step][v,labels].sum())
                        g[f'negative_score{step}']+=float(scores[step][v,~labels].sum())
                        g[f'positive_below{step}']+=int((scores[step][v,labels]<.5).sum())
                    if bound['covered']>int(labels.sum()) or (record['frame']==296 and role=='top'):
                        rows.append(dict(key=key,group=group,frame=record['frame'],role=role,bound=bound,
                            static_matches=actual['matches'],labels=labels.tolist(),scores={str(k):v0[v].tolist() for k,v0 in scores.items()}))
            for g in groups.values():
                for step in scorers:
                    g[f'positive_score{step}']/=max(1,g['positive_count'])
                    g[f'negative_score{step}']/=max(1,g['negative_count'])
            assert state_hash(tensor_state(model))==state_hash(original)
            assert all(p.grad is None for p in list(model.parameters())+list(center.parameters()))
            for p,h in inputs.items():assert fingerprint(__import__('pathlib').Path(p))==h
            for p,h in code.items():assert fingerprint(ROOT/p)==h
            write_json(OUT/'report.json',dict(passed=True,groups=dict(groups),rows=rows,dev88_scores_exact=True,optimizer_steps=0,
                code_sha256=code,original_model_unchanged=True,production_promoted=False,
                limitation='GT-assisted supervision diagnosis. Geometric capacity not deployable recall; frozen source-score assignment differs from current-score evaluation. No changed labels used in training.'))
            status('completed');print(json.dumps(dict(groups=groups)),flush=True)
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
