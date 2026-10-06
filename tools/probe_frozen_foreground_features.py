"""Train-only linear foreground probe on frozen pre-decoder features; no GT inference."""
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
from mousegeoforge.foreground_feature_probe import ForegroundLinearProbe,balanced_foreground_loss,mask_counts,ratios
from vggt_omega.models.heads.instance_pose_head import _sine_position_encoding

OUT=ROOT/'output/frozen_foreground_probe_v1'
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
            spec=json.loads((PARENT/'protocol.json').read_text())
            for p,h in spec['code_sha256'].items():assert fingerprint(ROOT/p)==h
            for p,h in spec['source_sha256'].items():assert scene_digest(ROOT/p)==h
            sequence=[x[0] for x in spec['arms']['adaptive_temperature']];assert len(sequence)==800
            train=list(dict.fromkeys(sequence));dev=[e['key'] for e in spec['evaluation']]
            assert len(train)==356 and len(dev)==88 and not(set(train)&set(dev))
            assert all(spec['records'][k]['split']=='train' for k in train)
            assert all(spec['records'][k]['split']=='val' for k in dev)
            train_sources={spec['records'][k]['source_group'] for k in train}
            dev_sources={spec['records'][k]['source_group'] for k in dev}
            # Source groups may be shared for some synthetic replays; preserve original split.
            dual_train={spec['records'][k]['source_group'] for k in train if spec['records'][k]['expected_instances']==2}
            dual_dev={spec['records'][k]['source_group'] for k in dev if spec['records'][k]['expected_instances']==2}
            assert not dual_train&dual_dev
            code={p:fingerprint(ROOT/p) for p in ('tools/probe_frozen_foreground_features.py','mousegeoforge/foreground_feature_probe.py','tests/test_foreground_feature_probe.py')}
            inputs={str(p):fingerprint(p) for p in (CHECKPOINT,PARENT/'protocol.json')}
            write_json(OUT/'protocol.json',dict(code_sha256=code,input_sha256=inputs,source_sha256=spec['source_sha256'],
                train_keys=train,evaluation=spec['evaluation'],schedule=sequence*2,seed=17,steps=1600,
                train_source_groups=sorted(train_sources),development_source_groups=sorted(dev_sources),dual_source_disjoint=True,
                feature='Frozen original shared/head projected memory minus exact sine encoding; L2 normalized per-pixel. Not rawDINO-only features.',
                probe='257 zero initialized scalar linear1x1 parameters; AdamW lr.01 wd.01, gradient clip1; balanced positive/negative soft-mask BCE; no coordinate channels.',
                labels='Train-only union masks area-downsampled to patch grid; fixed probability threshold.5. Native-grid GT for reporting uses area fraction>=.5.',
                scope='Capacity diagnostic, not instance localization OFF/ON experiment. No change to existing model, decoder, depth, assignment or deployment.',
                decision='If held-source dual foregroundIoU>=.5 and precision>=.7 and frame296 top instance recalls each>=.7, prioritize image-derived proposal initialization. If not, do not conclude feature information absent: linear-probe capacity/scale/data may limit it. No threshold or checkpoint selection.',
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
                    masks=t['masks'][0].bool() & t['instance_valid'][0,:,:,None,None]
                    fractions=F.interpolate(masks.any(1)[:,None].float(),size=(ph,pw),mode='area')[:,0]
                    v,n,h,w=masks.shape
                    instances=F.interpolate(masks.reshape(v*n,1,h,w).float(),size=(ph,pw),mode='area').reshape(v,n,ph,pw)>=.5
                    item=dict(features=feature,fractions=fractions,instances=instances,valid=t['instance_valid'][0],record=r)
                    if key in dev:
                        di=dev.index(key);saved=torch.load(PARENT/'adaptive_temperature'/f'prediction_{di:03d}.pt',map_location='cpu',weights_only=True)
                        for name,value in saved['predictions'].items():torch.testing.assert_close(pred[name].cpu(),value,atol=0,rtol=0)
                    cache[key]=item;cache_bytes+=feature.numel()*feature.element_size()
                    assert cache_bytes<=8*1024**3
                    del mem,pos,pred,batch,images,t,masks
                    if (index+1)%32==0:status('extracting',records=index+1,total=444,cache_gib=cache_bytes/1024**3);print('FEATURES',index+1,flush=True)
            finally:hook.remove()
            assert len(cache)==444
            probe=ForegroundLinearProbe().cuda();optimizer=torch.optim.AdamW(probe.parameters(),lr=.01,weight_decay=.01)
            history=[]
            for step,key in enumerate(sequence*2,1):
                budget();item=cache[key];optimizer.zero_grad(set_to_none=True)
                logits=probe(item['features'].cuda());loss=balanced_foreground_loss(logits,item['fractions'].cuda())
                assert torch.isfinite(loss);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(probe.parameters(),1.,error_if_nonfinite=True);optimizer.step()
                history.append(dict(step=step,key=key,loss=float(loss.detach()),gradient_norm=float(norm)))
                if step%100==0:write_json(OUT/'history.json',history);status('training_probe',step=step,steps=1600);print('STEP',step,history[-1]['loss'],flush=True)
            torch.save(dict(probe={k:v.cpu() for k,v in probe.state_dict().items()},optimizer=optimizer.state_dict(),steps=1600,
                            protocol_sha256=fingerprint(OUT/'protocol.json')),OUT/'probe.pt')
            records=[];group_counts=defaultdict(lambda:dict(tp=0,fp=0,fn=0));visuals={}
            for index,e in enumerate(spec['evaluation']):
                budget();item=cache[e['key']];r=item['record']
                with torch.no_grad():prob=probe(item['features'].cuda()).sigmoid().cpu()
                truth=item['fractions']>=.5
                for view in range(len(r['views'])):
                    counts=mask_counts(prob[view],truth[view])
                    for k,v in counts.items():group_counts[e['group']][k]+=v
                    recalls=[]
                    for instance in torch.where(item['valid'][view])[0]:
                        gt=item['instances'][view,instance];area=int(gt.sum())
                        recalls.append(dict(instance=int(instance),gt_native_pixels=area,recall=float(((prob[view]>=.5)&gt).sum())/max(1,area)))
                    records.append(dict(index=index,group=e['group'],frame=r['frame'],view=r['views'][view],role=r['physical_roles'][view],
                                        counts=counts,metrics=ratios(counts),instance_recalls=recalls))
                if index in (0,7,9,13):
                    visuals[index]=(r,prob,truth)
                    torch.save(dict(record=r,probability=prob,truth=truth),OUT/f'foreground_{index:03d}.pt')
            summary={g:dict(counts=c,**ratios(c)) for g,c in group_counts.items()}
            difficult=next(r for r in records if r['group']=='old_dual_pair' and r['frame']==296 and r['role']=='top')
            eligible=summary['old_dual_pair']['iou']>=.5 and summary['old_dual_pair']['precision']>=.7 and all(x['recall']>=.7 and x['gt_native_pixels']>0 for x in difficult['instance_recalls'])
            assert state_hash(tensor_state(model))==state_hash(original)
            assert encoder_hash==state_hash({k:v.cpu() for k,v in model.aggregator.state_dict().items()})
            assert flags=={n:p.requires_grad for n,p in model.named_parameters()}
            assert all(p.grad is None for p in model.parameters())
            assert not model.instance_pose_head.instance_decoder._forward_pre_hooks
            assert all(fingerprint(ROOT/p)==h for p,h in code.items())
            for p,h in inputs.items():assert fingerprint(__import__('pathlib').Path(p))==h
            fig,axes=plt.subplots(4,3,figsize=(13,11),layout='constrained')
            for row,index in enumerate((0,7,9,13)):
                r,prob,truth=visuals[index];view=r['physical_roles'].index('top');h,w=r['image_hw']
                rgb=Image.open(ROOT/r['scene']/'blended_images'/f'{r["views"][view]:08d}.jpg').resize((w,h))
                pp=F.interpolate(prob[view][None,None],size=(h,w),mode='nearest')[0,0].numpy()
                axes[row,0].imshow(rgb);axes[row,0].set_title(f'Frame{r["frame"]}: RGB')
                axes[row,1].imshow(pp,vmin=0,vmax=1,cmap='magma');axes[row,1].set_title('Frozen-feature foreground probability')
                axes[row,2].imshow(rgb);overlay=np.zeros((h,w,4));overlay[...,:3]=[0,1,1];overlay[...,3]=(pp>=.5)*.45
                axes[row,2].imshow(overlay);axes[row,2].set_title('Fixed0.5 threshold, no GT output selection')
                for ax in axes[row]:ax.axis('off')
            fig.savefig(OUT/'fixed_foreground.png',dpi=140);plt.close(fig)
            write_json(OUT/'report.json',dict(passed=True,diagnostic_only=True,steps=1600,parameters=257,
                feature_records=444,cache_gib=cache_bytes/1024**3,original_model_unchanged=True,original_eval_prediction_max_difference=0,
                summary=summary,records=records,frame296_top=difficult,proposal_route_supported=eligible,
                limitations=['Foreground union recall is not instance detection or separability.','Linear probe failure does not prove absent feature information.','Fixed balanced-loss threshold is not calibrated natural-class probability.','Reused synthetic development; no real Cylinder test.'],
                code_sha256=code,production_promoted=False))
            status('completed');print('PASSED probe capacity route',eligible,flush=True)
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
