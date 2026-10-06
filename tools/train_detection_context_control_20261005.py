"""Fixed-budget ROI detector context control, no pose/encoder updates."""
import argparse,collections,datetime as dt,fcntl,json,math,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from mousegeoforge.models.spatial_detection_readout import SpatialDetectionReadout
from mousegeoforge.detection_readout_loss import detection_readout_loss,match_detection,synthetic_detection_retention
from mousegeoforge.pilot_metrics import box_iou
from mousegeoforge.view_context_sampling import single_context_flags,random_reference_contexts,full_view_contexts
from run_depth_pose_ablation import fingerprint,write_json
CYCLE=ROOT/'output/core23_spatial_cycle_20261004';OUT=CYCLE/'detection_view_context_control_v1';STEPS=800

def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--order-control',action='store_true');parser.add_argument('--full-context-control',action='store_true');args=parser.parse_args();assert not(args.order_control and args.full_context_control);advanced=args.order_control or args.full_context_control
    contexts=['shared','single','full'] if args.full_context_control else ['paired','single','reversed'] if args.order_control else ['paired','single'];arms=['paired','full'] if args.full_context_control else ['fixed','random_ref'] if args.order_control else ['paired','mixed'];nc=len(contexts)
    if args.order_control:OUT=CYCLE/'detection_reference_control_v1'
    if args.full_context_control:OUT=CYCLE/'detection_full_context_control_v1'
    plan=json.loads((CYCLE/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'));lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not OUT.exists();OUT.mkdir();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    def status(stage,**kw):write_json(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),**kw))
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='training_detection_context_control',active_job=OUT.name,active_pid=os.getpid());write_json(CYCLE/'state.json',state);status('loading')
    source=CYCLE/'detection_features_v1/manifest.json';manifest=json.loads(source.read_text());rows=[]
    for rec in manifest['records']:
        assert fingerprint(ROOT/rec['path'])==rec['sha256'];rows.extend(torch.load(ROOT/rec['path'],map_location='cpu',weights_only=False)['observations'])
    meta=[o['metadata'] for o in rows];n=len(rows);assert n==1350
    paired_manifest=json.loads((CYCLE/'detection_roi_features_v1/manifest.json').read_text());assert fingerprint(ROOT/paired_manifest['path'])==paired_manifest['sha256'];paired=torch.load(ROOT/paired_manifest['path'],map_location='cpu',weights_only=False);assert paired['metadata']==meta
    single_manifest=json.loads((CYCLE/'single_detection_context_v1/manifest.json').read_text());assert fingerprint(ROOT/single_manifest['path'])==single_manifest['sha256'];single=torch.load(ROOT/single_manifest['path'],map_location='cpu',weights_only=False);assert single['metadata']==meta
    features={}
    for key in ['instance_tokens','original_boxes','original_objectness']:
        features[key]=torch.cat([torch.stack([o[key] for o in rows]),single[key]]).float().cuda()
    for key in ['roi','global_memory']:features[key]=torch.cat([paired[key],single[key]]).float().cuda()
    if args.order_control:
        reversed_manifest=json.loads((CYCLE/'reversed_detection_context_v1/manifest.json').read_text());assert fingerprint(ROOT/reversed_manifest['path'])==reversed_manifest['sha256'];reversed_data=torch.load(ROOT/reversed_manifest['path'],map_location='cpu',weights_only=False);assert reversed_data['metadata']==meta;torch.testing.assert_close(reversed_data['available'],single['available'],atol=0,rtol=0)
        for key in features:features[key]=torch.cat([features[key],reversed_data[key].float().cuda()])
        del reversed_data
    if args.full_context_control:
        context_manifests={};newdata={}
        for context,folder in [('shared','shared_detection_context_v1'),('full','fullshared_detection_context_v1')]:
            cm=json.loads((CYCLE/folder/'manifest.json').read_text());assert fingerprint(ROOT/cm['path'])==cm['sha256'] and max(cm['max_golden_delta'].values())<2e-5;d=torch.load(ROOT/cm['path'],map_location='cpu',weights_only=False);assert d['metadata']==meta;torch.testing.assert_close(d['available'],single['available'],atol=0,rtol=0);context_manifests[context]=cm;newdata[context]=d
        for key in features:features[key]=torch.cat([newdata['shared'][key],single[key],newdata['full'][key]]).float().cuda()
        del newdata,d
    targets={k:torch.stack([o[k] for o in rows]).repeat(nc,*([1]*(rows[0][k].ndim))).cuda() for k in ['boxes','box_valid','instance_valid']}
    targets['valid']=targets['box_valid']&targets['instance_valid']
    complete=torch.tensor([m['detection_complete'] for m in meta],device='cuda').repeat(nc);available=single['available'].cuda();assert available.sum()==1335
    sets=collections.defaultdict(list)
    for c,context in enumerate(contexts):
        for i,m in enumerate(meta):
            r=m['record'];assert r.get('session') not in ['GR2','UR2','UR3']
            if not m['detection_complete'] or not available[i]:continue
            k='real' if r['kind']=='real' else 'synthetic';prefix=context+'/'+k+'_'+r['role'];j=c*n+i
            sets[prefix].append(j);sets[prefix+('/dual' if m['expected_instances']>1 else '/single')].append(j)
            if k=='real':sets[prefix+f'/cam{chr(65+int(m["view"]))}'].append(j)
    synth_train=torch.tensor([m['record']['kind']!='real' and m['record']['role']=='train' and m['detection_complete'] for m in meta],device='cuda').repeat(nc)
    del single,paired,rows
    def forward(model,ids):return model(features['instance_tokens'][ids],features['roi'][ids],features['global_memory'][ids],features['original_boxes'][ids],features['original_objectness'][ids])
    @torch.no_grad()
    def predict(model):
        parts=collections.defaultdict(list)
        for start in range(0,nc*n,128):
            for key,value in forward(model,slice(start,start+128)).items():parts[key].append(value)
        return {key:torch.cat(value) for key,value in parts.items()}
    @torch.no_grad()
    def evaluate(p,include_train=False):
        output={}
        for group,ids in sets.items():
            if not include_train and '_train' in group:continue
            pp={k:v[ids] for k,v in p.items()};matches=match_detection(pp,targets['boxes'][ids],targets['valid'][ids]);tp=gt=pred=duplicates=0;ious=[];count_errors=[]
            for j,i in enumerate(ids):
                qs,ms=matches[j];selected=pp['instance_objectness_logits'][j].sigmoid()>=.5;pred+=int(selected.sum());gt+=len(ms);count_errors.append(abs(int(selected.sum())-len(ms)))
                matrix=box_iou(pp['instance_boxes_cxcywh'][j],targets['boxes'][i,ms])
                for q,m in zip(qs.tolist(),ms.tolist()):
                    v=float(matrix[q,ms.tolist().index(m)]);ious.append(v);tp+=int(v>=.5 and bool(selected[q]))
                if len(ms):duplicates+=int(((matrix[selected]>=.5).sum(0)-1).clamp_min(0).sum())
            output[group]=dict(images=len(ids),tp=tp,gt=gt,predictions=pred,precision=tp/max(pred,1),recall=tp/max(gt,1),matched_iou_mean=float(np.mean(ious)),duplicate_box_hits=duplicates,count_absolute_error=float(np.mean(count_errors)))
        return output
    protocol=dict(steps=STEPS,seeds=[17,29,43],arms=['paired','mixed'],initial='Same per-seed ROI1600 checkpoint; fresh optimizer both arms.',difference='Only actual encoder context; exact image schedules and single-context flags reused from view_context_control_v1.8real balanced4bottom/4other+8synthetic. Mixed changes2bottom+2other+4synthetic to singlecontext.',loss='Same detection loss weightedBCE+5L1+2GIoU and synthetic-only original-detector retentionweight1 as originalROI training. No pose/keypoint/mask/encoder/instance decoder loss.',optimizer='FreshAdamWwd.001,cosine1e-4to1e-5,clip1,800steps.',single_manifest_sha256=fingerprint(CYCLE/'single_detection_context_v1/manifest.json'),paired_features_sha256=paired_manifest['sha256'],source_sha256=fingerprint(source),script_sha256=fingerprint(Path(__file__)),evaluation='Confidence>=.5, box/objectness Hungarian match andIoU>=.5; no NMS, no GT instancecount in forward. Not AP. Duplicate_box_hits is overlapping detections perGT, may include a legitimate neighboring mouse in overlapping dual scenes.',pose_frozen='view_context_control_v1/mixed_seed{seed}_latest.pt used only in later downstream evaluation; not loaded/updated here.',production_changed=False)
    write_json(OUT/'protocol.json',protocol);results=[];saved={}
    if args.order_control:
        protocol.update(arms=arms,initial='Per-seed detection_view_context_control_v1/mixed800 checkpoint; fresh optimizer botharms.',difference='Same view_order_control_v1 image schedules, 50percent single/50percent paired botharms. random_ref reverses halfpaired only. Samebottom/image/domain exposures and originalspecialtokens; all6queries. Detector-only800steps3seeds.',pose_frozen='view_order_control_v1/random_ref_seed{seed}_latest.pt for later downstream evaluation only.',reversed_manifest_sha256=fingerprint(CYCLE/'reversed_detection_context_v1/manifest.json'));write_json(OUT/'protocol.json',protocol)
    if args.full_context_control:
        protocol.pop('paired_features_sha256');protocol.update(arms=arms,contexts=contexts,initial='Same perseed detection_view_context_control_v1/mixed800 detector; freshAdamW botharms.',difference='Same full_context_pose_control schedules andsingleexposure. Pairedcontrol50percent single50percent shared-tokenpair; fulltreatment50percent single25percent sharedpair25percent fulleligible sharedcontext. Sameimages/bottom/supervision/steps/params; onlyROI detectortrained. No reference-order augmentation.',pose_frozen='full_context_pose_control_v1/full_seed{seed}_latest.pt onlyused laterdownstreamevaluation, notloaded/updatedhere.',context_features_sha256={c:m['sha256'] for c,m in context_manifests.items()},limitation='Syntheticfulleligiblecache hasonly2views;4/8viewfullRGB stressuntrained. Incomplete15rows unavailable, notnegatives.');write_json(OUT/'protocol.json',protocol)
    gold_single=torch.load(ROOT/single_manifest['golden_path'],map_location='cuda',weights_only=False);assert fingerprint(ROOT/single_manifest['golden_path'])==single_manifest['golden_sha256']
    with np.load(CYCLE/'detection_context_roi_v1/predictions.npz') as a:gold_pair={s:{k:torch.tensor(a[f'seed{s}__{k}'],device='cuda') for k in gold_single[s]} for s in [17,29,43]}
    if args.order_control:
        assert fingerprint(ROOT/reversed_manifest['golden_path'])==reversed_manifest['golden_sha256'];gold_reverse=torch.load(ROOT/reversed_manifest['golden_path'],map_location='cuda',weights_only=False)
        with np.load(CYCLE/'detection_view_context_control_v1/predictions.npz') as a:
            for seed in [17,29,43]:
                for key in gold_single[seed]:
                    gold_single[seed][key]=torch.tensor(a[f'mixed_seed{seed}_single__{key}'],device='cuda');gold_pair[seed][key]=torch.tensor(a[f'mixed_seed{seed}_paired__{key}'],device='cuda')
    if args.full_context_control:
        gold_context={}
        for c,m in context_manifests.items():
            assert fingerprint(ROOT/m['golden_path'])==m['golden_sha256'];gold_context[c]=torch.load(ROOT/m['golden_path'],map_location='cuda',weights_only=False)
        gold_pair=gold_context['shared'];gold_reverse=gold_context['full']
        with np.load(CYCLE/'detection_view_context_control_v1/predictions.npz') as a:
            for seed in gold_single:
                for key in gold_single[seed]:gold_single[seed][key]=torch.tensor(a[f'mixed_seed{seed}_single__{key}'],device='cuda')
    for seed in [17,29,43]:
        schedule_path=CYCLE/f'{"full_context_pose_control_v1" if args.full_context_control else "view_order_control_v1" if args.order_control else "view_context_control_v1"}/seed{seed}_schedule.json';schedule=json.loads(schedule_path.read_text());images=schedule['images'];flags=schedule['mixed_single_flags'];assert len(images)==len(flags)==STEPS and flags==single_context_flags(images,meta,seed+14000)
        if args.order_control:
            reference_contexts=random_reference_contexts(images,flags,meta,seed+18000);assert reference_contexts==json.loads((CYCLE/f'view_order_control_v1/seed{seed}_reference_contexts.json').read_text())['contexts']
        if args.full_context_control:
            full_flags=full_view_contexts(images,flags,meta,seed+18000);assert full_flags==json.loads((CYCLE/f'full_context_pose_control_v1/seed{seed}_full_contexts.json').read_text())['contexts']
        checkpoint=CYCLE/f'detection_context_roi_v1/seed{seed}_latest.pt';cp=torch.load(checkpoint,map_location='cuda',weights_only=False);assert cp['step']==1600 and cp['seed']==seed
        if advanced:
            checkpoint=CYCLE/f'detection_view_context_control_v1/mixed_seed{seed}_latest.pt';cp=torch.load(checkpoint,map_location='cuda',weights_only=False);assert cp['step']==800 and cp['seed']==seed and cp['arm']=='mixed'
        initial_metrics=None
        for arm in arms:
            torch.manual_seed(seed);model=SpatialDetectionReadout().cuda().eval();model.load_state_dict(cp['state'],strict=True);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.001)
            p=predict(model);maxdelta=0.
            for key in p:
                mask=available if args.full_context_control else slice(None);delta=float((p[key][:n][mask]-gold_pair[seed][key][mask]).abs().max());assert delta<2e-5;maxdelta=max(maxdelta,delta)
                delta=float((p[key][n:2*n][available]-gold_single[seed][key][available]).abs().max());assert delta<2e-5;maxdelta=max(maxdelta,delta)
                if advanced:
                    delta=float((p[key][2*n:][available]-gold_reverse[seed][key][available]).abs().max());assert delta<2e-5;maxdelta=max(maxdelta,delta)
            baseline=evaluate(p,include_train=True)
            if initial_metrics is None:initial_metrics=baseline
            else:assert baseline==initial_metrics
            history=[dict(step=0,metrics=baseline)];write_json(OUT/f'{arm}_seed{seed}_initial.json',dict(max_delta=maxdelta,checkpoint_sha256=fingerprint(checkpoint),schedule_sha256=fingerprint(schedule_path),metrics=baseline))
            for step,(step_images,step_flags) in enumerate(zip(images,flags),1):
                if dt.datetime.now(dt.timezone.utc)>=deadline:
                    torch.save(dict(state=model.state_dict(),optimizer=optimizer.state_dict(),step=step-1,seed=seed,arm=arm),OUT/f'{arm}_seed{seed}_deadline.pt');status('deadline_stopped');state.update(stage='training_deadline_reached_wrapup_required',active_job=None,active_pid=None);write_json(CYCLE/'state.json',state);return
                ids=[i+n*int(f and arm=='mixed') for domain in ['real','synthetic'] for i,f in zip(step_images[domain],step_flags[domain])];assert complete[ids].all()
                if args.order_control:
                    ids=[i+n*(contexts.index(c) if arm=='random_ref' else int(c=='single')) for domain in ['real','synthetic'] for i,c in zip(step_images[domain],reference_contexts[step-1][domain])];assert complete[ids].all()
                if args.full_context_control:
                    ids=[i+n*(contexts.index(c) if arm=='full' else int(c=='single')) for domain in ['real','synthetic'] for i,c in zip(step_images[domain],full_flags[step-1][domain])];assert complete[ids].all() and available[[i%n for i in ids]].all()
                lr=1e-5+(1e-4-1e-5)*.5*(1+math.cos(math.pi*(step-1)/STEPS))
                for g in optimizer.param_groups:g['lr']=lr
                optimizer.zero_grad(set_to_none=True);prediction=forward(model,ids);loss,logs=detection_readout_loss(prediction,targets['boxes'][ids],targets['valid'][ids],complete[ids]);retention,retlogs=synthetic_detection_retention(prediction,features['original_boxes'][ids],features['original_objectness'][ids],synth_train[ids]);loss=loss+retention;assert torch.isfinite(loss);loss.backward()
                if step==1:
                    assert all(not v.requires_grad and v.grad is None for v in features.values());write_json(OUT/f'{arm}_seed{seed}_gradient_check.json',dict(input_gradients=False,only_roi_detector_parameters=True,trainable_parameters=sum(p.numel() for p in model.parameters()),pose_encoder_instance_decoder_unmodified=True))
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm);optimizer.step()
                if step%200==0:
                    p=predict(model);metrics=evaluate(p,include_train=step==STEPS);row=dict(step=step,loss=float(loss.detach()),metrics=metrics);history.append(row);write_json(OUT/f'{arm}_seed{seed}_history.json',history);torch.save(dict(state=model.state_dict(),optimizer=optimizer.state_dict(),step=step,seed=seed,arm=arm,initial_sha256=fingerprint(checkpoint)),OUT/f'{arm}_seed{seed}_latest.pt');status('training',arm=arm,seed=seed,step=step);print('DETCTX',arm,seed,step,metrics['single/real_validation'],metrics[contexts[0]+'/synthetic_validation/dual'],flush=True)
            for c,context in enumerate(contexts):
                for key,value in p.items():saved[f'{arm}_seed{seed}_{context}__{key}']=value[c*n:(c+1)*n].cpu().numpy()
            results.append(dict(arm=arm,seed=seed,**history[-1]));write_json(OUT/'partial_results.json',results);np.savez_compressed(OUT/'predictions.npz',**saved)
    write_json(OUT/'results.json',dict(results=results,pose_evaluation_pending=True,no_production_promotion=True));status('complete');state.update(stage='detection_context_control_ready',active_job=None,active_pid=None,updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),next='Review detector paired/mixed800steps3seeds final vs identical initial, real/syntheticdual/context/camera precision-recall-counts. Then frozen mixed800pose re-evaluation with new predictedboxes for ALLqueries, new GTbox association only for metrics. Keep olddetector baseline, no automatic promotion. If no stable detector+pose gain stop similar extensions.');state['completed'].append('detection_context_control3seeds800steps2arms');write_json(CYCLE/'state.json',state)

if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write_json(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()))
        raise
