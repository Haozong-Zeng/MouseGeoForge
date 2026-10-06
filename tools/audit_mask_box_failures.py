"""Frozen, preregistered mask failure diagnostics on eight dev + train scenes."""
import argparse
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
from matplotlib.patches import Rectangle
from torch.nn import functional as F
from audit_mask_box_decode import SOURCES,ARMS
from run_depth_pose_ablation import ROOT,torch,VGGTOmega,tensor_state,state_hash,fingerprint,write_json,instance_targets_from_batch
from train_cylinder_adaptive_temperature import TemperatureModel
from train_cylinder_view_balance import oriented_batch,physical_metrics,scene_digest
from mousegeoforge.adaptive_temperature import restore_adaptive
from mousegeoforge.mask_box_decode import mask_boxes_from_logits
from mousegeoforge.mask_shape_audit import mask_shape_statistics
from mousegeoforge.pilot_metrics import box_iou

OUT=ROOT/'output/mask_box_failures_v1'
PARENT=ROOT/'output/mask_box_decode_v1'


def selected(spec):
    rows=[]
    for idx in (5,6,7,10,21,26,28,31):
        e=spec['evaluation'][idx]
        rows.append(dict(tag=f'dev_{idx:03d}',index=idx,subset='selected_dev_failure',record=spec['records'][e['key']]))
    scenes=sorted({r['scene'] for r in spec['records'].values() if r['split']=='train'})
    for i,scene in enumerate(scenes):
        candidates=[r for r in spec['records'].values() if r['scene']==scene and r['split']=='train']
        # Deterministic selection from already-used train records, before new predictions.
        r=min(candidates,key=lambda x:(abs(len(x['views'])-2),-x['physical_roles'].count('side'),tuple(x['views'])))
        rows.append(dict(tag=f'train_{i:03d}',index=None,subset='train',record=r))
    assert len(scenes)==40
    return rows


def sheet(seed,item,v,masks,target,pred,decoded,stats,folder):
    r=item['record'];pixel=np.array(r['image_hw'][::-1]);rgb=np.array(Image.open(ROOT/r['scene']/'blended_images'/f'{r["views"][v]:08d}.jpg').resize(tuple(pixel)))
    queries=[q for q in range(len(masks)) if float(pred['instance_objectness_logits'][0,v,q].sigmoid())>=.5]
    fig,axes=plt.subplots(1,1+len(queries),figsize=(5*(1+len(queries)),4.8),layout='constrained');axes=np.atleast_1d(axes)
    def draw(ax,b,color):
        b=np.asarray(b);ax.add_patch(Rectangle((b[:2]-b[2:]/2)*pixel,*(b[2:]*pixel),fill=False,edgecolor=color,lw=1.3))
    axes[0].imshow(rgb)
    for m in torch.where(target['instance_valid'][0,v])[0].tolist():
        mask=target['masks'][0,v,m].numpy().astype(bool)
        axes[0].imshow(np.dstack([np.zeros_like(mask),mask,np.zeros_like(mask),mask*.4]))
        draw(axes[0],target['boxes_cxcywh'][0,v,m].numpy(),'lime')
    axes[0].set_title('GT visible masks / boxes')
    for ax,q in zip(axes[1:],queries):
        mask=masks[q];ax.imshow(rgb);ax.imshow(np.dstack([mask,np.zeros_like(mask),mask,mask*.5]))
        draw(ax,pred['instance_boxes_cxcywh'][0,v,q].numpy(),'yellow');draw(ax,decoded[0,v,q].numpy(),'cyan')
        s=stats[q];ax.set_title(f'q{q} score={s["score"]:.5f}\ncomponents={s["components"]}, largest={s["largest_component_fraction"]:.4f}\nmain/full box area={s["largest_to_full_box_area"]:.3f}',fontsize=9)
    for ax in axes:ax.set_xlim(0,pixel[0]);ax.set_ylim(pixel[1],0);ax.axis('off')
    fig.suptitle(f'{seed} {item["tag"]} frame {r["frame"]} / {r["physical_roles"][v]} cam{r["views"][v]}\nPred mask magenta; regressed box yellow; full mask box cyan. No filtering or output changes.')
    fig.savefig(folder/f'view_{v}.png',dpi=130);plt.close(fig)


def run():
    start=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'output/cylinder_training.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (OUT/'protocol.json').exists():raise RuntimeError('Existing protocol; never silently restart')
        def status(state,**kw):write_json(OUT/'status.json',dict(state=state,pid=os.getpid(),elapsed_s=time.monotonic()-start,**kw))
        status('preparing')
        try:
            assert json.loads((PARENT/'verification.json').read_text())['passed']
            spec=json.loads((PARENT/'protocol.json').read_text());items=selected(spec)
            checkpoints={s:p/ARMS[s]/'final_heads.pt' for s,p in SOURCES.items()}
            code={p:fingerprint(ROOT/p) for p in ('tools/audit_mask_box_failures.py','mousegeoforge/mask_shape_audit.py')}
            hashes={str(PARENT/'protocol.json'):fingerprint(PARENT/'protocol.json'),**{str(p):fingerprint(p) for p in checkpoints.values()}}
            for p,h in spec['code_sha256'].items():assert fingerprint(ROOT/p)==h
            sources={r['record']['scene']:spec['source_sha256'][r['record']['scene']] for r in items}
            assert all(scene_digest(ROOT/p)==h for p,h in sources.items())
            write_json(OUT/'protocol.json',dict(items=items,source_sha256=sources,input_sha256=hashes,code_sha256=code,
                scope='Eight preselected development failure records, all40 existing training scenes with deterministic representative view subsets. Two frozen ON checkpoints. No training, threshold sweep or prediction changes.',
                components='8-connected at fixed logits>0 after bilinear upsample; largest component is a diagnostic only, never replaces output or chooses a GT winner.'))
            torch.set_num_threads(2);torch.manual_seed(17);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
            encoder=ROOT/'vggt-omega/vggt_omega_1b_512.pt';assert fingerprint(encoder)==spec['encoder_sha256']
            spatial=VGGTOmega(enable_camera=False,enable_depth=True,enable_multi_instance_pose=True,num_keypoints=23)
            weights=torch.load(encoder,weights_only=True,mmap=True)
            loaded=spatial.load_state_dict({k:v for k,v in weights.items() if k.startswith(('aggregator.','dense_head.'))},strict=False)
            assert not loaded.unexpected_keys and all(k.startswith('instance_pose_head.') for k in loaded.missing_keys)
            del weights
            model=TemperatureModel.from_spatial_model(spatial).cuda().eval();del spatial
            states={s:torch.load(p,map_location='cpu',weights_only=True)['model'] for s,p in checkpoints.items()}
            all_rows=[];comparisons=[]
            for ii,item in enumerate(items):
                if time.monotonic()-start>1200:raise RuntimeError('20minute safety cap')
                r=item['record'];batch=oriented_batch(r,num_views=len(r['views']));target=instance_targets_from_batch(batch);images=batch['images'].cuda()
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):cached,start_idx=model.aggregator(images)
                for seed,state in states.items():
                    restore_adaptive(model,state)
                    with torch.no_grad():pred={k:v.cpu() for k,v in model.forward_cached(images,cached,start_idx,coupling='off').items()}
                    metrics,assignment=physical_metrics(pred,batch,target)
                    if item['index'] is not None:
                        saved=torch.load(PARENT/seed/'regressed'/f'prediction_{item["index"]:03d}.pt',map_location='cpu',weights_only=True)
                        for k,v in saved['predictions'].items():torch.testing.assert_close(pred[k],v,atol=0,rtol=0)
                        assert assignment==saved['oracle_assignment']
                    decoded,_=mask_boxes_from_logits(pred['instance_mask_logits'],pred['instance_boxes_cxcywh'],r['image_hw'],enabled=True)
                    newmetrics,newassignment=physical_metrics(dict(pred,instance_boxes_cxcywh=decoded),batch,target)
                    comparisons.append(dict(seed=seed,tag=item['tag'],subset=item['subset'],dataset=r['dataset'],regressed=metrics,mask=newmetrics,old_assignment=assignment,new_assignment=newassignment))
                    folder=OUT/seed/item['tag'];folder.mkdir(parents=True,exist_ok=True)
                    for v in range(len(r['views'])):
                        masks=(F.interpolate(pred['instance_mask_logits'][0,v,:,None],size=r['image_hw'],mode='bilinear',align_corners=False)[:,0]>0).numpy()
                        for m in torch.where(target['instance_valid'][0,v])[0].tolist():Image.fromarray(target['masks'][0,v,m].numpy().astype(np.uint8)*255).save(folder/f'v{v}_gt{m}.png')
                        stats=[];qs,ms=assignment[v]
                        for q,mask in enumerate(masks):
                            s=mask_shape_statistics(mask);s.update(seed=seed,tag=item['tag'],subset=item['subset'],dataset=r['dataset'],scene=r['scene'],view=v,camera=r['views'][v],role=r['physical_roles'][v],query=q,score=float(pred['instance_objectness_logits'][0,v,q].sigmoid()),matched_instance=ms[qs.index(q)] if q in qs else None)
                            torch.testing.assert_close(torch.tensor(s['full_box_cxcywh']),decoded[0,v,q],atol=2e-7,rtol=0)
                            s['mask_regressed_box_iou']=float(box_iou(decoded[0,v,q:q+1],pred['instance_boxes_cxcywh'][0,v,q:q+1])[0,0])
                            s['gt_mask_ious']=[]
                            for m in torch.where(target['instance_valid'][0,v])[0].tolist():
                                gt=target['masks'][0,v,m].numpy().astype(bool);s['gt_mask_ious'].append(dict(instance=m,iou=float((gt&mask).sum()/max(1,(gt|mask).sum()))))
                            if q in qs:
                                m=ms[qs.index(q)];gtbox=target['boxes_cxcywh'][0,v,m:m+1]
                                s.update(regressed_gt_iou=float(box_iou(gtbox,pred['instance_boxes_cxcywh'][0,v,q:q+1])[0,0]),mask_box_gt_iou=float(box_iou(gtbox,decoded[0,v,q:q+1])[0,0]),diagnostic_largest_box_gt_iou=float(box_iou(gtbox,torch.tensor(s['largest_component_box_cxcywh'])[None])[0,0]))
                            Image.fromarray(mask.astype(np.uint8)*255).save(folder/f'v{v}_q{q}.png');stats.append(s);all_rows.append(s)
                        if item['subset']!='train':sheet(seed,item,v,masks,target,pred,decoded,stats,folder)
                    del pred,decoded
                del cached,images,batch,target
                if (ii+1)%8==0:status('auditing',records=ii+1,total=len(items));print(f'{ii+1}/{len(items)} x2',flush=True)
            from pathlib import Path
            assert all(fingerprint(Path(p))==h for p,h in hashes.items())
            assert all(fingerprint(ROOT/p)==h for p,h in code.items())
            assert all(scene_digest(ROOT/p)==h for p,h in sources.items())
            for state in states.values():restore_adaptive(model,state);assert state_hash(tensor_state(model))==state_hash(state)
            write_json(OUT/'query_statistics.json',all_rows);write_json(OUT/'comparisons.json',comparisons)
            write_json(OUT/'verification.json',dict(passed=True,optimizer_steps=0,encoder_forwards=len(items),head_forwards=2*len(items),development_prediction_max_difference=0,weights_sources_code_unchanged=True,elapsed_s=time.monotonic()-start))
            status('completed');print('COMPLETED',flush=True)
        except BaseException as error:status('failed',error=repr(error));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--launch',action='store_true');args=parser.parse_args()
    if args.launch:
        if (OUT/'status.json').exists():raise RuntimeError('Already launched')
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/'run.log').open('a') as log:p=subprocess.Popen([sys.executable,'-u',__file__],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(pid=p.pid,output=str(OUT))))
    else:run()
