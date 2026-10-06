"""Matched old/new TRAIN query association with frozen ROI detector predictions."""
import argparse,collections,datetime as dt,fcntl,hashlib,json,math,os,random,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from torch.nn import functional as F
from mousegeoforge.models.core23_spatial_readout import Core23SpatialReadout
from mousegeoforge.models.decoder_spatial_pose_readout import DecoderSpatialPoseReadout
from mousegeoforge.detection_readout_loss import match_detection
from mousegeoforge.matched_point_loss import matched_image_point_loss
from mousegeoforge.bottom_view_sampling import balance_bottom_real_batches
from mousegeoforge.pilot_metrics import box_iou
from run_depth_pose_ablation import fingerprint,write_json
CYCLE=ROOT/'output/core23_spatial_cycle_20261004';OUT=CYCLE/'pose_query_alignment_v1';DET=CYCLE/'detection_context_roi_v1'

def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--back-control',action='store_true');parser.add_argument('--decoder-control',action='store_true');parser.add_argument('--bottom-balanced',action='store_true');parser.add_argument('--extend-decoder',action='store_true');args=parser.parse_args()
    if args.bottom_balanced:args.decoder_control=True
    assert not (args.back_control and args.decoder_control)
    if args.back_control:OUT=CYCLE/'real_back_weight_control_v1'
    if args.decoder_control:OUT=CYCLE/'decoder_unfreeze_control_v1'
    if args.bottom_balanced:OUT=CYCLE/'bottom_balanced_decoder_v1'
    assert not args.extend_decoder or args.decoder_control
    resume_dir=OUT if args.extend_decoder else None
    if args.extend_decoder:OUT=CYCLE/('bottom_balanced_extended2400_v1' if args.bottom_balanced else 'decoder_unfrozen_extended2400_v1')
    start_step=800 if args.extend_decoder else 0
    steps=2400 if args.extend_decoder else 800 if args.decoder_control else 1600
    plan=json.loads((CYCLE/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'))
    lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not OUT.exists();OUT.mkdir();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    def status(stage,**kw):write_json(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),**kw))
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='training_pose_query_alignment',active_job=OUT.name,active_pid=os.getpid(),updated_utc=dt.datetime.now(dt.timezone.utc).isoformat());write_json(CYCLE/'state.json',state);status('loading')
    source=CYCLE/'detection_features_v1/manifest.json';manifest=json.loads(source.read_text());rows=[]
    for r in manifest['records']:
        assert fingerprint(ROOT/r['path'])==r['sha256'];rows.extend(torch.load(ROOT/r['path'],map_location='cpu',weights_only=False)['observations'])
    meta=[o['metadata'] for o in rows];assert meta==json.loads((DET/'observations.json').read_text());n=len(rows)
    data={k:torch.stack([o[k] for o in rows]).cuda() for k in ['instance_tokens','pose_tokens','coarse_xy','original_boxes','original_objectness','boxes','box_valid','instance_valid','gt_xy','point_valid','pixels']}
    complete=torch.tensor([m['detection_complete'] for m in meta],device='cuda');valid_box=data['box_valid']&data['instance_valid']&complete[:,None]
    buckets={k:collections.defaultdict(list) for k in ['real','synthetic']};kind=[];role=[]
    for i,m in enumerate(meta):
        r=m['record'];assert r.get('session') not in ['GR2','UR2','UR3'];k='real' if r['kind']=='real' else 'synthetic';kind.append(k);role.append(r['role'])
        if m['expected_instances']>1:data['point_valid'][i,:,7:14]=False
        if m['detection_complete'] and r['role']=='train' and data['point_valid'][i].any():buckets[k][r.get('session') or r['scene'].split('/')[2]].append(i)
    validation=[i for i,m in enumerate(meta) if role[i]=='validation' and m['detection_complete']]
    # Check exactly the same downstream validation pool as the prior ROI audit.
    with np.load(CYCLE/'detection_pose_context_roi_audit_v1/predictions.npz') as a:
        assert validation==a['image_indices'].tolist();golden={s:a[f'seed{s}'][:,1].copy() for s in [17,29,43]}
    if args.decoder_control:
        native_source=CYCLE/'native_decoder_memory_v1/manifest.json';native_manifest=json.loads(native_source.read_text());native_memory=[None]*n;lookup={(m['record']['key'],m['view_index']):i for i,m in enumerate(meta)}
        for rec in native_manifest['records']:
            assert fingerprint(ROOT/rec['path'])==rec['sha256'];raw=torch.load(ROOT/rec['path'],map_location='cpu',weights_only=False);maps=raw['memory'].cuda();assert maps.dtype==torch.float32
            for m in raw['metadata']:
                i=lookup[(m['record']['key'],m['view_index'])];assert m==meta[i];native_memory[i]=maps[m['view_index']]
        assert all(x is not None for x in native_memory)
        decoder_initial=CYCLE/'native_decoder_memory_v1/initial_decoder.pt';assert fingerprint(decoder_initial)==native_manifest['decoder_initial_sha256']
        decoder_weights=torch.load(decoder_initial,map_location='cuda',weights_only=True)
    else:
        hashes=json.loads((CYCLE/'detection_roi_features_v1/manifest.json').read_text())['memory_hashes'];memory=torch.empty(n,256,48,48,device='cuda');last=None;verified=set()
        for i,m in enumerate(meta):
            path=m['memory_part']
            if path!=last:
                if path not in verified:assert fingerprint(ROOT/path)==hashes[path];verified.add(path)
                raw=torch.load(ROOT/path,map_location='cpu',weights_only=False);maps=F.interpolate(raw['memory'].float(),(48,48),mode='bilinear',align_corners=False).cuda();last=path
            memory[i]=maps[m['view_index']]
    del maps,raw
    base=dict(instance_boxes_cxcywh=data['original_boxes'],instance_objectness_logits=data['original_objectness'])
    old_matches=match_detection(base,data['boxes'],valid_box)
    def assignments(matches):
        out=torch.full((n,2),-1,device='cuda',dtype=torch.long)
        for i,(qs,ms) in enumerate(matches):out[i,ms]=qs
        return out
    old=assignments(old_matches);instances=valid_box.nonzero();ii,mm=instances[:,0],instances[:,1]
    def selected_prediction(model,indices,queries,pred_boxes):
        if args.decoder_control:
            images=indices.tolist();groups=collections.defaultdict(list);outputs={}
            for i in dict.fromkeys(images):groups[tuple(native_memory[i].shape[-2:])].append(i)
            for members in groups.values():
                p=model(data['instance_tokens'][members],torch.stack([native_memory[i] for i in members]),pred_boxes[members])['xy']
                for j,i in enumerate(members):outputs[i]=p[j]
            # All138queries have already decoded; target association is applied
            # only now, outside model.forward. No padding or resized native maps.
            return torch.stack([outputs[i][q] for i,q in zip(images,queries.tolist())])
        return model(data['pose_tokens'][indices,queries],data['coarse_xy'][indices,queries],memory[indices],predicted_boxes=pred_boxes[indices,queries])
    def loss_for(model,ids,mapping,pred_boxes,back_weight=1.):
        pairs=valid_box[ids].nonzero();local,m=pairs[:,0],pairs[:,1];im=torch.tensor(ids,device='cuda')[local];q=mapping[im,m];assert (q>=0).all()
        p=selected_prediction(model,im,q,pred_boxes)
        weights=None
        if back_weight!=1.:
            weights=torch.ones_like(data['point_valid'][im,m],dtype=torch.float32)
            is_real=torch.tensor([kind[i]=='real' for i in im.tolist()],device='cuda')
            weights[is_real,5:7]=back_weight
        return matched_image_point_loss(p,data['gt_xy'][im,m],data['point_valid'][im,m],data['pixels'][im],local,len(ids),weights)
    @torch.no_grad()
    def all_validation(model,pred_boxes):
        parts=[]
        for start in range(0,len(validation),8):
            ids=torch.tensor(validation[start:start+8],device='cuda');ims=ids[:,None].expand(-1,6).flatten();qs=torch.arange(6,device='cuda')[None].expand(len(ids),-1).flatten()
            parts.append(selected_prediction(model,ims,qs,pred_boxes).reshape(len(ids),6,23,2).cpu())
        return torch.cat(parts)
    write_json(OUT/'protocol.json',dict(seeds=[17,29,43],additional_steps=1600,initial_pose_step=6400,optimizer='Fresh AdamWwd.001 botharms,cosine1e-4to1e-5,clip1',batch='8real+8synthetic source-uniform, same per-seed image schedule botharms',arms=['original','adapted'],difference='Only TRAIN GT-to-query association: frozenoriginal vs frozenROI detector. Both use adapted predicted box prior; both evaluate adapted association. No GT crop/input to inference.',loss='Per-image equal-weight instances and per-instance valid point mean, SmoothL1beta.01 pixel/768;synthetic+.25real. Invalid points masked; dualtail8-14off. No heatmap/curve loss.',source_sha256=fingerprint(source),detector_predictions_sha256=fingerprint(DET/'predictions.npz'),script_sha256=fingerprint(Path(__file__)),point_loss_sha256=fingerprint(ROOT/'mousegeoforge/matched_point_loss.py'),validation_images=len(validation),production_changed=False))
    write_json(OUT/'observations.json',meta);results=[];saved={}
    if args.decoder_control:
        protocol=json.loads((OUT/'protocol.json').read_text());protocol.update(additional_steps=steps,arms=['frozen','unfrozen'],difference='Only keypointdecoder trainable scope. Both update spatialreadout; detector,instance decoder,encoder,semantic queryembeddings,coarseMLP frozen. Both adapted GT-query association. Not equal-parameter ablation.',optimizer='Fresh AdamWwd.001,clip1;readout1e-4to1e-5;unfrozen decoder1e-5to1e-6;800steps. Same seeds/schedule.',native_manifest_sha256=fingerprint(native_source),decoder_initial_sha256=fingerprint(decoder_initial),model_source_sha256=fingerprint(ROOT/'mousegeoforge/models/decoder_spatial_pose_readout.py'),numeric='FP32native memory H/W grouped,all6x23queryselfattention;readoutmemory fp16roundtrip48x48. CoarseMLP frozen but outputs change with decoder;coarseprior gradient detached. Eval mode/no dropout in both arms.')
        write_json(OUT/'protocol.json',protocol)
        if args.bottom_balanced:
            protocol.update(arms=['balanced'],batch='4camA+4otherreal, source-uniform within each view pool; exact unchanged8synthetic indices from uniform control.',difference='Sampling distribution only versus completed decoder_unfreeze_control_v1/unfrozen. Both decoder+readout trainable, same initialization/optimizer/loss/800steps3seeds. No tiny-fit checkpoint used. No inference cameraID routing.',control='decoder_unfreeze_control_v1/unfrozen',sampler_sha256=fingerprint(ROOT/'mousegeoforge/bottom_view_sampling.py'))
            write_json(OUT/'protocol.json',protocol)
        if args.extend_decoder:
            protocol.update(additional_steps=1600,total_steps=2400,start_step=800,arms=['balanced' if args.bottom_balanced else 'unfrozen'],resume=str(resume_dir.relative_to(ROOT)),optimizer='Resume exact800step model AND AdamWstate; matched low-LR restart readout5e-5to1e-5,decoder1e-5to1e-6 over1600extra steps, wd.001 clip1. Uniform and balanced bothextended.',batch='4camA+4otherreal + same8synthetic' if args.bottom_balanced else '8source-uniformreal+8source-uniformsynthetic')
            write_json(OUT/'protocol.json',protocol)
    if args.back_control:
        write_json(OUT/'back_control_protocol.json',dict(strict_control='pose_query_alignment_v1/adapted_seed{17,29,43} reuse unchanged completed control',new_arms=dict(downweight=.1,omit=0.),denominator='Original valid count preserved; reliable point gradients not renormalized upward.',scope='Only realTRAIN points06/07; syntheticback unchanged, evaluationvalidity unchanged. Same initial6400 checkpoints, same1600steps3seeds same asserted image schedules and adapted TRAIN matching. No label edits or fabricated curves.',interpretation='Measures incremental fine-tuning sensitivity from a checkpoint already exposed to realback labels; NOT from-scratch removal and NOT anatomicalGT validation. Manualback error secondary; reliable real/precise synthetic/generalization primary.'))
    with np.load(DET/'predictions.npz') as a:detector={s:{k:torch.tensor(a[f'seed{s}__{k}'],device='cuda') for k in base} for s in [17,29,43]}
    for seed in [17,29,43]:
        adapted=detector[seed];new=assignments(match_detection(adapted,data['boxes'],valid_box));hit=torch.zeros(n,2,dtype=torch.bool,device='cuda')
        for i,m in instances.tolist():
            q=int(new[i,m]);hit[i,m]=bool(adapted['instance_objectness_logits'][i,q].sigmoid()>=.5 and box_iou(adapted['instance_boxes_cxcywh'][i,q:q+1],data['boxes'][i,m:m+1])[0,0]>=.5)
        write_json(OUT/f'seed{seed}_matching.json',dict(original=old.cpu().tolist(),adapted=new.cpu().tolist(),switches={k:sum(int(old[i,m]!=new[i,m]) for i,m in instances.tolist() if kind[i]==k) for k in buckets}))
        @torch.no_grad()
        def evaluate(model):
            accum=collections.defaultdict(lambda:dict(error=[],detected_error=[],points=0,detected_points=0,pck16=0,pck32=0))
            for start in range(0,len(instances),16):
                im=ii[start:start+16];ms=mm[start:start+16];p=selected_prediction(model,im,new[im,ms],adapted['instance_boxes_cxcywh']);err=((p-data['gt_xy'][im,ms])*data['pixels'][im,None]).norm(dim=-1).cpu();valid=data['point_valid'][im,ms].cpu()
                for j,(i,m) in enumerate(zip(im.tolist(),ms.tolist())):
                    prefix=kind[i]+'_'+role[i];good=valid[j];groups={'all':good,'back':good&torch.tensor([k in [5,6] for k in range(23)]),'reliable':good&torch.tensor([k in [0,14,15,16,17] if kind[i]=='real' else k not in [5,6] for k in range(23)]),'dual' if meta[i]['expected_instances']>1 else 'single':good}
                    groups.update({f'point{k+1:02d}':good&torch.arange(23).eq(k) for k in range(23) if good[k]})
                    if args.decoder_control and kind[i]=='real':
                        cam=f'cam{chr(65+int(meta[i]["view"]))}';groups[cam+'/reliable']=groups['reliable'];groups[cam+'/feet']=good&torch.tensor([k in [14,15,16,17] for k in range(23)])
                    for key,v in groups.items():
                        e=err[j,v].tolist();a=accum[prefix+'/'+key];a['error']+=e;a['points']+=len(e)
                        if hit[i,m]:a['detected_error']+=e;a['detected_points']+=len(e);a['pck16']+=sum(x<=16 for x in e);a['pck32']+=sum(x<=32 for x in e)
            metrics={}
            for key,a in accum.items():
                for field in ['error','detected_error']:
                    values=a.pop(field);a[field+'_mean']=float(np.mean(values)) if values else None;a[field+'_median']=float(np.median(values)) if values else None
                a['gated_pck32']=a['pck32']/max(a['points'],1);metrics[key]=a
            return metrics
        rng=random.Random(seed+8000);schedule=[{k:[rng.choice(buckets[k][rng.choice(sorted(buckets[k]))]) for _ in range(8)] for k in buckets} for _ in range(steps)]
        if args.bottom_balanced:
            control_schedule=schedule
            assert control_schedule[:800]==json.loads((CYCLE/f'decoder_unfreeze_control_v1/seed{seed}_schedule.json').read_text())['steps']
            eligible=[i for group in buckets['real'].values() for i in group];schedule=balance_bottom_real_batches(schedule,meta,eligible,seed+9000)
            assert all(s['synthetic']==c['synthetic'] for s,c in zip(schedule,control_schedule))
            exposure=collections.Counter((meta[i]['record']['session'],int(meta[i]['view'])) for s in schedule for i in s['real'])
            previous=collections.Counter((meta[i]['record']['session'],int(meta[i]['view'])) for s in control_schedule for i in s['real'])
            write_json(OUT/f'seed{seed}_exposures.json',dict(balanced={str(k):v for k,v in exposure.items()},control={str(k):v for k,v in previous.items()},balanced_bottom=sum(v for (s,c),v in exposure.items() if c==0),control_bottom=sum(v for (s,c),v in previous.items() if c==0),total_real=steps*8,synthetic_exactly_preserved=True))
        write_json(OUT/f'seed{seed}_schedule.json',dict(steps=schedule,sha256=hashlib.sha256(json.dumps(schedule,sort_keys=True).encode()).hexdigest()))
        if args.back_control:assert schedule==json.loads((CYCLE/f'pose_query_alignment_v1/seed{seed}_schedule.json').read_text())['steps']
        if args.decoder_control and not args.bottom_balanced:assert schedule[:min(steps,1600)]==json.loads((CYCLE/f'pose_query_alignment_v1/seed{seed}_schedule.json').read_text())['steps'][:min(steps,1600)]
        if args.extend_decoder:assert schedule[:800]==json.loads((resume_dir/f'seed{seed}_schedule.json').read_text())['steps']
        checkpoint=CYCLE/f'readout_post4800_control_v1/spatial_seed{seed}_latest.pt';cp=torch.load(checkpoint,map_location='cuda',weights_only=False);assert cp['step']==6400
        baseline=None
        arms=[('downweight',new,.1),('omit',new,0.)] if args.back_control else [('original',old,1.),('adapted',new,1.)]
        if args.decoder_control:arms=[('frozen',new,1.),('unfrozen',new,1.)]
        if args.bottom_balanced:arms=[('balanced',new,1.)]
        elif args.extend_decoder:arms=[('unfrozen',new,1.)]
        for arm,mapping,back_weight in arms:
            torch.manual_seed(seed)
            if args.decoder_control:
                model=DecoderSpatialPoseReadout().cuda().eval();loaded=model.load_state_dict(decoder_weights,strict=False);assert not loaded.unexpected_keys and all(k.startswith('spatial_readout.') for k in loaded.missing_keys)
                model.spatial_readout.load_state_dict(cp['state'],strict=True);model.set_decoder_trainable(arm in ['unfrozen','balanced'])
                groups=[dict(params=list(model.spatial_readout.parameters()),lr=1e-4,peak_lr=1e-4,floor_lr=1e-5)]
                if arm in ['unfrozen','balanced']:groups.append(dict(params=list(model.keypoint_decoder.parameters()),lr=1e-5,peak_lr=1e-5,floor_lr=1e-6))
                optimizer=torch.optim.AdamW(groups,weight_decay=.001)
            else:model=Core23SpatialReadout().cuda();model.load_state_dict(cp['state'],strict=True);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.001)
            reference=golden[seed]
            if args.extend_decoder:
                resume_path=resume_dir/f'{arm}_seed{seed}_latest.pt';previous=torch.load(resume_path,map_location='cuda',weights_only=False);assert previous['step']==800 and previous['seed']==seed and previous['arm']==arm
                model.load_state_dict(previous['state'],strict=True);optimizer.load_state_dict(previous['optimizer'])
                for gi,g in enumerate(optimizer.param_groups):g['peak_lr']=5e-5 if gi==0 else 1e-5;g['floor_lr']=1e-5 if gi==0 else 1e-6
                with np.load(resume_dir/'predictions.npz') as a:assert validation==a['image_indices'].tolist();reference=a[f'{arm}_seed{seed}'].copy()
                write_json(OUT/f'{arm}_seed{seed}_resume.json',dict(checkpoint_sha256=fingerprint(resume_path),step=800,optimizer_restored=True))
            initial=all_validation(model,adapted['instance_boxes_cxcywh']);delta=float((initial-torch.tensor(reference)).abs().max());assert delta<2e-5,(seed,arm,delta)
            metrics=evaluate(model)
            if baseline is None:baseline=metrics
            elif args.decoder_control:
                for key in metrics:
                    for metric in ['error_mean','error_median']:
                        if metrics[key][metric] is not None:assert abs(metrics[key][metric]-baseline[key][metric])<.002,(key,metric)
            else:assert metrics==baseline
            if args.back_control:assert metrics==json.loads((CYCLE/f'pose_query_alignment_v1/adapted_seed{seed}_history.json').read_text())[0]['metrics']
            if args.bottom_balanced and not args.extend_decoder:
                control_initial=json.loads((CYCLE/f'decoder_unfreeze_control_v1/unfrozen_seed{seed}_history.json').read_text())[0]['metrics']
                for key in metrics:
                    if metrics[key]['error_mean'] is not None:assert abs(metrics[key]['error_mean']-control_initial[key]['error_mean'])<.002
            if args.extend_decoder:
                old_metrics=json.loads((resume_dir/f'{arm}_seed{seed}_history.json').read_text())[-1]['metrics']
                for key in metrics:
                    if metrics[key]['error_mean'] is not None:assert abs(metrics[key]['error_mean']-old_metrics[key]['error_mean'])<.002
            history=[dict(step=start_step,metrics=metrics)];write_json(OUT/f'{arm}_seed{seed}_initial.json',dict(checkpoint_sha256=fingerprint(checkpoint),initial_prediction_delta=delta,comparison='resume800' if args.extend_decoder else 'initial6400'))
            for step,ids in enumerate(schedule[start_step:],start_step+1):
                if dt.datetime.now(dt.timezone.utc)>=deadline:
                    torch.save(dict(state=model.state_dict(),optimizer=optimizer.state_dict(),step=step-1),OUT/f'{arm}_seed{seed}_deadline.pt');status('deadline_stopped');state.update(stage='training_deadline_reached_wrapup_required',active_pid=None,active_job=None);write_json(CYCLE/'state.json',state);return
                fraction=.5*(1+math.cos(math.pi*(step-1-start_step)/(steps-start_step)))
                for g in optimizer.param_groups:g['lr']=g.get('floor_lr',1e-5)+(g.get('peak_lr',1e-4)-g.get('floor_lr',1e-5))*fraction
                optimizer.zero_grad(set_to_none=True);synthetic=loss_for(model,ids['synthetic'],mapping,adapted['instance_boxes_cxcywh'],back_weight);real=loss_for(model,ids['real'],mapping,adapted['instance_boxes_cxcywh'],back_weight);loss=synthetic+.25*real;assert torch.isfinite(loss);loss.backward()
                if args.decoder_control and step==start_step+1:
                    forbidden=[name for name,p in model.named_parameters() if not p.requires_grad and p.grad is not None];assert not forbidden
                    decoder_norm=sum(float(p.grad.detach().square().sum()) for p in model.keypoint_decoder.parameters() if p.grad is not None)**.5
                    assert (decoder_norm>0)==(arm in ['unfrozen','balanced'])
                    write_json(OUT/f'{arm}_seed{seed}_gradient_check.json',dict(decoder_grad_norm=decoder_norm,forbidden_gradients=forbidden,trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),all138queries=True,inputs_frozen=not data['instance_tokens'].requires_grad and all(not x.requires_grad for x in native_memory)))
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm);optimizer.step()
                if step%200==0:
                    metrics=evaluate(model);row=dict(step=step,synthetic_loss=float(synthetic.detach()),real_loss=float(real.detach()),metrics=metrics);history.append(row);write_json(OUT/f'{arm}_seed{seed}_history.json',history)
                    torch.save(dict(state=model.state_dict(),optimizer=optimizer.state_dict(),step=step,seed=seed,arm=arm,initial_checkpoint_sha256=fingerprint(checkpoint)),OUT/f'{arm}_seed{seed}_latest.pt');status('training',arm=arm,seed=seed,step=step);print('ALIGN',arm,seed,step,'real',metrics['real_validation/reliable']['error_mean'],'dual',metrics['synthetic_validation/dual']['error_mean'],flush=True)
            saved[f'{arm}_seed{seed}']=all_validation(model,adapted['instance_boxes_cxcywh']).numpy();results.append(dict(arm=arm,seed=seed,**history[-1]));write_json(OUT/'partial_results.json',results)
            np.savez_compressed(OUT/'predictions.npz',image_indices=np.array(validation),**saved)
    write_json(OUT/'results.json',dict(results=results,no_production_promotion=True));status('complete');state.update(stage='pose_query_alignment_results_ready',active_job=None,active_pid=None,updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),next=f'Review {OUT.name} results/curves and raw overlays, not only mean error. For bottom-balanced compare completed uniform/unfrozen decodercontrol, per-camera side retention and syntheticdual plus exposures. For decodercontrol compare per-camera bottom feet and syntheticdual, verify initial parity/gradient checks; no blind extension. Detector fixed. No promotion.');state['completed'].append(OUT.name+f'_3seeds{steps}steps{len(arms)}arms');write_json(CYCLE/'state.json',state)

if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(exist_ok=True);write_json(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()));raise
