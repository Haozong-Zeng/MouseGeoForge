"""Fixed-teacher quality and gradient feasibility on eight existing replay poses."""
import fcntl
import json
import time
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_depth_pose_ablation import ROOT,torch,VGGTOmega,DepthPoseModel,restore,tensor_state,state_hash,fingerprint,write_json,instance_targets_from_batch,depth_loss,move_batch
from train_cylinder_view_balance import oriented_batch,scene_digest
from mousegeoforge.replay_retention import build_teacher_targets,replay_retention_pose_loss,upstream
from mousegeoforge.pilot_metrics import box_iou

OUT=ROOT/'output/cylinder_retention_preflight_v1'
PARENT=ROOT/'output/cylinder_source_mix_v1'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'output/cylinder_training.lock').open('a+') as glock,(OUT/'running.lock').open('a+') as lock:
        fcntl.flock(glock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (OUT/'protocol.json').exists():raise RuntimeError('Existing preflight; do not overwrite')
        run()


def run():
    start=time.monotonic();spec=json.loads((PARENT/'protocol.json').read_text())
    seq=[spec['records'][k[0]] for k in spec['arms']['source_mix']]
    records=sorted({r['scene']:r for r in seq if r['expected_instances']==2}.values(),key=lambda r:r['frame'])
    assert len(records)==8 and all(r['split']=='train' for r in records)
    assert json.loads((PARENT/'delivery_verification.json').read_text())['passed']
    assert json.loads((PARENT/'status.json').read_text())['state']=='completed'
    with (PARENT/'running.lock').open() as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for p,h in spec['code_sha256'].items():assert fingerprint(ROOT/p)==h
    sources={r['scene']:r['source_revision'] for r in records}
    assert all(scene_digest(ROOT/p)==h for p,h in sources.items())
    paths={'teacher':ROOT/spec['config']['checkpoint'],'student':PARENT/'source_mix/final_heads.pt'}
    code={p:fingerprint(ROOT/p) for p in ('tools/preflight_cylinder_replay_retention.py','mousegeoforge/replay_retention.py')}
    protocol=dict(records=records,source_sha256=sources,code_sha256=code,
        checkpoints={k:dict(path=str(p.relative_to(ROOT)),sha256=fingerprint(p)) for k,p in paths.items()},
        parent_protocol_sha256=fingerprint(PARENT/'protocol.json'),aux_weight=1.,beta=.01,
        teacher_gate='Teacher score>=.5 and GT box IoU>=.5; only existing valid GT keypoints. Identity-based same-target training assignments; never query-index alignment.',
        decision='On the same qualified instance-view cohort, mean teacher IoU >= student and point-weighted body/tail NME <= student; report individual violations too. Student checkpoint median auxiliary/base total gradient ratio <=.25 for shared adapter and full instance head. Original initialization retention/gradient exactly zero; OFF exactly upstream; one student match; all finite.',
        scope='Eight existing dual replay TRAIN poses only, no new data/development selection. Teacher independent original state, student mixture, no epipolar term or optimizer updates.')
    write_json(OUT/'protocol.json',protocol)
    torch.set_num_threads(2);torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    encoder=ROOT/'vggt-omega/vggt_omega_1b_512.pt';assert fingerprint(encoder)==spec['encoder_sha256']
    spatial=VGGTOmega(enable_camera=False,enable_depth=True,enable_multi_instance_pose=True,num_keypoints=23)
    weights=torch.load(encoder,weights_only=True,mmap=True)
    loaded=spatial.load_state_dict({k:v for k,v in weights.items() if k.startswith(('aggregator.','dense_head.'))},strict=False)
    assert not loaded.unexpected_keys and all(k.startswith('instance_pose_head.') for k in loaded.missing_keys)
    del weights
    model=DepthPoseModel.from_spatial_model(spatial).cuda().eval();del spatial
    states={k:torch.load(p,weights_only=True,map_location='cpu')['model'] for k,p in paths.items()}
    groups={'shared_adapter':list(model.shared_adapter.parameters()),'instance_head':list(model.instance_pose_head.parameters())}
    params=[p for ps in groups.values() for p in ps]
    lw=upstream.InstancePoseLossWeights(keypoint_3d=0.,reprojection=0.,local_heatmap=0.,keypoint_uncertainty=0.)
    args=SimpleNamespace(semantic_label_weights='',body_depth_balance=.5,body_depth_fraction=.5,depth_gradient_weight=0.)
    gradients=[];quality=[];parity=0.;teacher_paths={};matcher=upstream.hungarian_match_instances
    for idx,r in enumerate(records):
        batch=move_batch(oriented_batch(r,num_views=len(r['views'])),'cuda');target=instance_targets_from_batch(batch)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):cached,patch_start=model.aggregator(batch['images'])
        restore(model,states['teacher'])
        with torch.no_grad():teacher=model.forward_cached(batch['images'],cached,patch_start,coupling='off')
        teacher_matches=matcher(teacher,target);packed=build_teacher_targets(teacher,target,teacher_matches)
        # Only compact outputs; no cached encoder or source image duplication.
        path=OUT/f'teacher_{idx:02d}.pt'
        cpu_packed=[[{identity:{k:v.cpu() if torch.is_tensor(v) else v for k,v in data.items()} for identity,data in view.items()} for view in packed[0]]]
        torch.save(dict(record=r,teacher_targets=cpu_packed),path);teacher_paths[str(path.relative_to(ROOT))]=fingerprint(path)
        for tag,state in states.items():
            restore(model,state);pred=model.forward_cached(batch['images'],cached,patch_start,coupling='off')
            baseline,_=upstream.instance_pose_loss(pred,target,lw,image_size=tuple(r['image_hw']))
            with patch.object(upstream,'hungarian_match_instances',wraps=matcher) as calls:
                off,_=replay_retention_pose_loss(pred,target,lw,teacher_targets=packed,aux_weight=0.,image_size=tuple(r['image_hw']));assert calls.call_count==1
            parity=max(parity,float((baseline-off).abs().detach()));assert torch.equal(baseline,off)
            del baseline,off
            with patch.object(upstream,'hungarian_match_instances',wraps=matcher) as calls:
                total,logs=replay_retention_pose_loss(pred,target,lw,teacher_targets=packed,aux_weight=1.,image_size=tuple(r['image_hw']));assert calls.call_count==1
            dl=depth_loss(pred['depth'],batch['depths'],batch['depth_masks'],batch['body_masks'],batch['semantic_masks'],args)['loss_depth']
            base=logs['loss_instance_base_pose']+dl;aux=logs['loss_instance_retention']
            ga=torch.autograd.grad(aux,params,retain_graph=True,allow_unused=True);gb=torch.autograd.grad(base,params,allow_unused=True)
            offset=0
            for group,ps in groups.items():
                a=[torch.zeros_like(p) if g is None else g.detach() for g,p in zip(ga[offset:offset+len(ps)],ps)]
                b=[torch.zeros_like(p) if g is None else g.detach() for g,p in zip(gb[offset:offset+len(ps)],ps)]
                na=float(sum(x.square().sum() for x in a).sqrt());nb=float(sum(x.square().sum() for x in b).sqrt())
                assert np.isfinite(na) and np.isfinite(nb) and nb>0
                if tag=='teacher':assert na==0 and float(aux.detach())==0
                gradients.append(dict(checkpoint=tag,record=idx,scene=r['scene'],group=group,base_norm=nb,aux_norm=na,ratio=na/nb,
                    auxiliary_loss=float(aux.detach()),retained_instance_views=logs['retained_instance_views'],retained_points=logs['retained_points']))
                offset+=len(ps)
            matches=matcher(pred,target);pixel=pred['instance_keypoint_xy'].new_tensor(r['image_hw'][::-1])
            for v,(qs,ms) in enumerate(matches[0]):
                for q,m in zip(qs.tolist(),ms.tolist()):
                    identity=int(target['instance_ids'][0,v,m])
                    if identity not in packed[0][v]:continue
                    box=pred['instance_boxes_cxcywh'][0,v,q];gtbox=target['boxes_cxcywh'][0,v,m]
                    iou=float(box_iou(gtbox[None],box[None])[0,0]);diag=(gtbox[2:]*pixel).norm()
                    err=((pred['instance_keypoint_xy'][0,v,q]-target['keypoint_xy'][0,v,m])*pixel).norm(dim=-1)/diag
                    valid=target['keypoint_valid'][0,v,m];body=valid.clone();body[7:14]=False;tail=valid & ~body
                    quality.append(dict(checkpoint=tag,record=idx,role=r['physical_roles'][v],view=v,identity=identity,iou=iou,
                        body_errors=err[body].detach().cpu().tolist(),tail_errors=err[tail].detach().cpu().tolist()))
            assert state_hash(tensor_state(model))==state_hash(state)
            del pred,dl,base,aux,total,logs,ga,gb,a,b
        del cached,batch,target,teacher,packed
        print('Checked',idx+1,'/8',flush=True)
    metrics={}
    for tag in states:
        subset=[q for q in quality if q['checkpoint']==tag]
        metrics[tag]=dict(instance_views=len(subset),box_iou=float(np.mean([q['iou'] for q in subset])),
            body_nme=float(np.mean([e for q in subset for e in q['body_errors']])),tail_nme=float(np.mean([e for q in subset for e in q['tail_errors']])))
    gates={k:metrics['teacher'][k]>=metrics['student'][k] if k=='box_iou' else metrics['teacher'][k]<=metrics['student'][k] for k in ('box_iou','body_nme','tail_nme')}
    ratios={g:float(np.median([r['ratio'] for r in gradients if r['checkpoint']=='student' and r['group']==g])) for g in groups}
    gates.update({g+'_gradient':x<=.25 for g,x in ratios.items()})
    gates['nonempty_qualified_targets']=metrics['teacher']['instance_views']>0
    qt={(q['record'],q['view'],q['identity']):q for q in quality if q['checkpoint']=='teacher'}
    qs={(q['record'],q['view'],q['identity']):q for q in quality if q['checkpoint']=='student'}
    assert qt.keys()==qs.keys()
    violations=dict(box=sum(qt[k]['iou']<qs[k]['iou'] for k in qt),body=sum(np.mean(qt[k]['body_errors'])>np.mean(qs[k]['body_errors']) for k in qt),tail=sum(np.mean(qt[k]['tail_errors'])>np.mean(qs[k]['tail_errors']) for k in qt))
    assert all(fingerprint(p)==protocol['checkpoints'][k]['sha256'] for k,p in paths.items())
    assert all(fingerprint(ROOT/p)==h for p,h in code.items());assert all(scene_digest(ROOT/p)==h for p,h in sources.items())
    report=dict(passed=True,training_supported=all(gates.values()),gates=gates,quality=metrics,individual_quality_violations=violations,student_median_gradient_ratios=ratios,
        zero_weight_max_difference=parity,teacher_initialization_auxiliary_zero=True,student_matcher_calls_per_loss=1,
        teacher_targets_sha256=teacher_paths,encoder_forwards=8,head_forwards=24,optimizer_steps=0,weights_source_code_unchanged=True,elapsed_s=time.monotonic()-start)
    write_json(OUT/'gradients.json',gradients);write_json(OUT/'quality.json',quality);write_json(OUT/'report.json',report);write_json(OUT/'status.json',dict(state='completed'))
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,key in zip(axes,('box_iou','body_nme','tail_nme')):
        ax.bar(['teacher','student'],[metrics[t][key] for t in states]);ax.set_title(key)
    fig.suptitle('Same teacher-qualified dual TRAIN instances | frozen models, no updates')
    fig.savefig(OUT/'teacher_quality.png',dpi=145);plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
