"""Matched paired-only versus single/paired TRAIN context adaptation."""
import argparse,collections,datetime as dt,fcntl,hashlib,json,math,os,random,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from mousegeoforge.models.decoder_spatial_pose_readout import DecoderSpatialPoseReadout
from mousegeoforge.detection_readout_loss import match_detection
from mousegeoforge.matched_point_loss import matched_image_point_loss
from mousegeoforge.bottom_view_sampling import balance_bottom_real_batches
from mousegeoforge.view_context_sampling import single_context_flags,random_reference_contexts,full_view_contexts
from mousegeoforge.pilot_metrics import box_iou
from run_depth_pose_ablation import fingerprint,write_json
CYCLE=ROOT/'output/core23_spatial_cycle_20261004';OUT=CYCLE/'view_context_control_v1';STEPS=800

def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--order-control',action='store_true');parser.add_argument('--shared-control',action='store_true');parser.add_argument('--full-context-control',action='store_true');args=parser.parse_args()
    if args.full_context_control:
        assert not(args.order_control or args.shared_control);args.shared_control=True
    assert not (args.order_control and args.shared_control);advanced=args.order_control or args.shared_control;alternate='shared' if args.shared_control else 'reversed'
    if args.order_control:OUT=CYCLE/'view_order_control_v1'
    if args.shared_control:OUT=CYCLE/'shared_token_pose_control_v1'
    if args.full_context_control:OUT=CYCLE/'full_context_pose_control_v1'
    plan=json.loads((CYCLE/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'))
    lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not OUT.exists();OUT.mkdir();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=False
    def status(stage,**kw):write_json(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),**kw))
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='training_view_context_control',active_job=OUT.name,active_pid=os.getpid());write_json(CYCLE/'state.json',state);status('loading')
    source=CYCLE/'detection_features_v1/manifest.json';manifest=json.loads(source.read_text());rows=[]
    for rec in manifest['records']:
        assert fingerprint(ROOT/rec['path'])==rec['sha256'];rows.extend(torch.load(ROOT/rec['path'],map_location='cpu',weights_only=False)['observations'])
    meta=[o['metadata'] for o in rows];n=len(rows);assert n==1350
    data={k:torch.stack([o[k] for o in rows]).cuda() for k in ['instance_tokens','boxes','box_valid','instance_valid','gt_xy','point_valid','pixels']}
    complete=torch.tensor([m['detection_complete'] for m in meta],device='cuda');valid=data['box_valid']&data['instance_valid']&complete[:,None]
    kind=['real' if m['record']['kind']=='real' else 'synthetic' for m in meta];role=[m['record']['role'] for m in meta]
    for i,m in enumerate(meta):
        assert m['record'].get('session') not in ['GR2','UR2','UR3']
        if m['expected_instances']>1:data['point_valid'][i,:,7:14]=False
    eligible=[i for i,m in enumerate(meta) if m['detection_complete'] and bool(data['point_valid'][i].any())]
    train=[i for i in eligible if role[i]=='train'];validation=[i for i in eligible if role[i]=='validation']
    buckets={k:collections.defaultdict(list) for k in ['real','synthetic']}
    for i in train:
        r=meta[i]['record'];buckets[kind[i]][r.get('session') or r['scene'].split('/')[2]].append(i)
    memories={'paired':[None]*n,'single':[None]*n};tokens={'paired':data['instance_tokens'],'single':torch.zeros_like(data['instance_tokens'])}
    lookup={(m['record']['key'],m['view_index']):i for i,m in enumerate(meta)}
    native_source=CYCLE/'native_decoder_memory_v1/manifest.json';native=json.loads(native_source.read_text())
    for rec in native['records']:
        assert fingerprint(ROOT/rec['path'])==rec['sha256'];raw=torch.load(ROOT/rec['path'],map_location='cpu',weights_only=False);maps=raw['memory'].cuda()
        for m in raw['metadata']:
            i=lookup[(m['record']['key'],m['view_index'])];assert m==meta[i];memories['paired'][i]=maps[m['view_index']]
    single_det={s:dict(instance_boxes_cxcywh=torch.zeros(n,6,4,device='cuda'),instance_objectness_logits=torch.zeros(n,6,device='cuda')) for s in [17,29,43]}
    manifests={};seen=set()
    for part_role in ['train','validation']:
        path=CYCLE/f'single_view_{part_role}_context_v1/manifest.json';part=json.loads(path.read_text());assert part['role']==part_role and part['source_sha256']==fingerprint(source);manifests[part_role]=fingerprint(path)
        for rec in part['records']:
            i=rec['image_index'];assert role[i]==part_role and i not in seen;seen.add(i)
            assert fingerprint(ROOT/rec['path'])==rec['sha256'];raw=torch.load(ROOT/rec['path'],map_location='cpu',weights_only=False);assert raw['metadata']==meta[i]
            memories['single'][i]=raw['memory'].cuda();tokens['single'][i]=raw['instance_tokens'].cuda()
            for seed in single_det:
                for key in single_det[seed]:single_det[seed][key][i]=raw['adapted'][seed][key].cuda()
    assert seen==set(eligible) and all(memories[c][i] is not None for c in memories for i in eligible)
    reverse_det=None
    if advanced:
        memories[alternate]=[None]*n;tokens[alternate]=torch.zeros_like(data['instance_tokens']);reverse_det={s:{k:torch.zeros_like(v) for k,v in single_det[s].items()} for s in single_det};seen_reverse=set()
        for part_role in ['train','validation']:
            path=CYCLE/f'{alternate}_view_{part_role}_context_v1/manifest.json';part=json.loads(path.read_text());assert part['role']==part_role and part['source_sha256']==fingerprint(source);manifests[alternate+'_'+part_role]=fingerprint(path)
            for rec in part['records']:
                i=rec['image_index'];assert role[i]==part_role and i not in seen_reverse;seen_reverse.add(i);assert fingerprint(ROOT/rec['path'])==rec['sha256'];raw=torch.load(ROOT/rec['path'],map_location='cpu',weights_only=False);assert raw['metadata']==meta[i] and raw['context_view_index']==(meta[i]['view_index'] if args.shared_control else 1-meta[i]['view_index'])
                memories[alternate][i]=raw['memory'].cuda();tokens[alternate][i]=raw['instance_tokens'].cuda()
                for seed in reverse_det:
                    for key in reverse_det[seed]:reverse_det[seed][key][i]=raw['adapted'][seed][key].cuda()
        assert seen_reverse==set(eligible)
    if args.full_context_control:
        memories['full']=[None]*n;tokens['full']=torch.zeros_like(data['instance_tokens']);full_det={s:{k:torch.zeros_like(v) for k,v in single_det[s].items()} for s in single_det};seen_full=set()
        for part_role in ['train','validation']:
            path=CYCLE/f'fullshared_view_{part_role}_context_v1/manifest.json';part=json.loads(path.read_text());assert part['role']==part_role and part['source_sha256']==fingerprint(source) and part['full2view_golden_cases']>0 and part['max_full2view_golden_delta']<2e-5;manifests['full_'+part_role]=fingerprint(path)
            for rec in part['records']:
                i=rec['image_index'];assert role[i]==part_role and i not in seen_full;seen_full.add(i);assert fingerprint(ROOT/rec['path'])==rec['sha256'];raw=torch.load(ROOT/rec['path'],map_location='cpu',weights_only=False);assert raw['metadata']==meta[i] and raw['context_kind']=='fullshared' and raw['context_record']['views'][raw['context_view_index']]==meta[i]['view']
                memories['full'][i]=raw['memory'].cuda();tokens['full'][i]=raw['instance_tokens'].cuda()
                for seed in full_det:
                    for key in full_det[seed]:full_det[seed][key][i]=raw['adapted'][seed][key].cuda()
        assert seen_full==set(eligible)
    del maps,raw,rows
    protocol=dict(seeds=[17,29,43],steps=STEPS,arms=['paired','mixed'],initial='Per-seed bottom_balanced_extended2400_v1 checkpoint, full decoder+readout identical in both arms; fresh optimizer both arms.',
        difference='Context only: paired arm uses existing2view features; mixed uses exactly2of4bottom real,2of4other real,4of8synthetic single-context per step. Same image schedule/point loss/bottom exposure/parameters/steps/seeds. Each single variant is a genuine V=1 encoder run. No view-ID routing at inference.',
        detector='Fixed per-seed ROI model. Predictions and box/objectness Hungarian assignment recomputed for each context, outside model.forward. No GT query count/crop/coordinates enter prediction.',
        optimizer='Fresh AdamWwd.001; readout5e-5to1e-5,decoder1e-5to1e-6 cosine,clip1;800steps. Semantic queryembeddings/coarseMLP/instance decoder/encoder/detector frozen. All138 keypoint queries preserved.',
        loss='synthetic+.25real; same original valid-count mean. No heatmap/curve loss or relabeling. Dualtail8-14disabled. Train/testsource roles immutable.',
        source_sha256=fingerprint(source),native_manifest_sha256=fingerprint(native_source),single_manifests=manifests,script_sha256=fingerprint(Path(__file__)),validation_images=len(validation),train_images=len(train),production_changed=False)
    write_json(OUT/'protocol.json',protocol);results=[];saved={}
    with np.load(CYCLE/'detection_context_roi_v1/predictions.npz') as a:paired_det={s:{k:torch.tensor(a[f'seed{s}__{k}'],device='cuda') for k in single_det[s]} for s in single_det}
    if advanced:
        with np.load(CYCLE/'detection_view_context_control_v1/predictions.npz') as a:
            for seed in single_det:
                for key in single_det[seed]:
                    paired_det[seed][key]=torch.tensor(a[f'mixed_seed{seed}_paired__{key}'],device='cuda');single_det[seed][key]=torch.tensor(a[f'mixed_seed{seed}_single__{key}'],device='cuda')
        protocol.update(arms=['fixed','random_ref'],initial='Same per-seed view_context_control_v1/mixed800 pose checkpoint; fresh optimizer both arms. No special-token edits.',difference='Reference order only: both50percent single+50percent paired contexts. random_ref reverses half paired occurrences (oneof2bottom paired, oneof2side paired, twoof4synthetic paired); fixed keeps originalpairedorder. Same800steps/image schedules/contextcount/bottom exposure/seeds/parameters. Native features actuallyreencoded with reversed2view RGB.',detector='Frozen per-seed detection_view_context_control_v1/mixed800 detector for allcontexts; cached predictions percontext verified. GT assignment outsideforward. No detector/encoder/instance decoder update.',detector_predictions_sha256=fingerprint(CYCLE/'detection_view_context_control_v1/predictions.npz'),single_manifests=manifests,sampler_sha256=fingerprint(ROOT/'mousegeoforge/view_context_sampling.py'))
        write_json(OUT/'protocol.json',protocol)
        if args.shared_control:
            protocol.update(arms=['original','shared'],initial='Identical perseed view_context_control_v1/mixed800 pose; freshoptimizer. Same immutableencoder and fixedmixed800detector weights.',difference='Same originalimageorder, 50percent actualsingle+50percent paired inbotharms. Original uses originalfirst/other tokens; shared uses firsttoken values forallpairedviews. Runtimefeatureextraction only, noencoderparameterupdate. Allnon-single TRAIN occurrences shared in treatment, nothalf orrandomreference. No geometryloss or geometryGT fromsharedpass.',detector='Frozen same mixed800ROI weights, but paired/shared featurecontext yields differentpredictedboxes and GTboxassociation outsideforward; comparebotharms onbothcontexts and initialbaseline toattributeheadadaptation. All138queries; noGTcrop.',geometry='Original multiview geometry mustremain separate atintegration; thiscache/trainer is2Donly.');write_json(OUT/'protocol.json',protocol)
        if args.full_context_control:
            protocol.update(arms=['paired','full'],initial='Identical perseed shared_token_pose_control_v1/shared800 pose andfreshoptimizer. Frozenmixed800ROI detector.',difference='Both50percent single; control other50percent shared-token2view. Treatment25percent shared-token2view+25percent shared-tokenfulleligiblecontext. Same supervisedimages/schedule/bottomexposure/labels/params/steps. Actualencoder re-encoding notmemory stacking. Existingeligible2view syntheticfull isidentical tosharedpair; realfull5-7views.',limitation='Synthetic4/8viewfullRGB remainsuntrained stresscondition. Noadditionalcameras, originalgeometrypreserved separately. Native memory andfrozendetector predictions varywithcontext; GTassociation outsideforward.');write_json(OUT/'protocol.json',protocol)
    def forward_images(model,images,contexts,predictions):
        groups=collections.defaultdict(list);outputs={}
        for i,c in dict.fromkeys(zip(images,contexts)):groups[(c,tuple(memories[c][i].shape[-2:]))].append(i)
        for (c,_),members in groups.items():
            p=model(tokens[c][members],torch.stack([memories[c][i] for i in members]),predictions[c]['instance_boxes_cxcywh'][members])['xy']
            for j,i in enumerate(members):outputs[(i,c)]=p[j]
        return torch.stack([outputs[(i,c)] for i,c in zip(images,contexts)])
    for seed in [17,29,43]:
        detections=dict(paired=paired_det[seed],single=single_det[seed]);mapping={};hit={}
        if advanced:detections[alternate]=reverse_det[seed]
        if args.full_context_control:detections['full']=full_det[seed]
        for context,prediction in detections.items():
            mapping[context]=torch.full((n,2),-1,device='cuda',dtype=torch.long);hit[context]=torch.zeros(n,2,device='cuda',dtype=torch.bool)
            ids=eligible;p={k:v[ids] for k,v in prediction.items()};matches=match_detection(p,data['boxes'][ids],valid[ids])
            for i,(qs,ms) in zip(ids,matches):
                mapping[context][i,ms]=qs
                for q,m in zip(qs.tolist(),ms.tolist()):hit[context][i,m]=bool(prediction['instance_objectness_logits'][i,q].sigmoid()>=.5 and box_iou(prediction['instance_boxes_cxcywh'][i,q:q+1],data['boxes'][i,m:m+1])[0,0]>=.5)
        @torch.no_grad()
        def evaluate(model,include_train=False):
            accum=collections.defaultdict(lambda:dict(errors=[],count=0,pck16=0,pck32=0))
            ids=eligible if include_train else validation
            for context in detections:
                for start in range(0,len(ids),8):
                    group=ids[start:start+8];pred=forward_images(model,group,[context]*len(group),detections)
                    for j,i in enumerate(group):
                        for m in valid[i].nonzero().flatten().tolist():
                            q=int(mapping[context][i,m]);assert q>=0;error=((pred[j,q]-data['gt_xy'][i,m])*data['pixels'][i]).norm(dim=-1).cpu();good=data['point_valid'][i,m].cpu()
                            groups={'all':good,'reliable':good&torch.tensor([k in [0,14,15,16,17] if kind[i]=='real' else k not in [5,6] for k in range(23)]),'back':good&torch.tensor([k in [5,6] for k in range(23)]),'dual' if meta[i]['expected_instances']>1 else 'single':good}
                            if kind[i]=='real':groups[f'cam{chr(65+int(meta[i]["view"]))}/feet']=good&torch.tensor([k in [14,15,16,17] for k in range(23)])
                            for name,g in groups.items():
                                values=error[g].tolist();a=accum[f'{context}/{kind[i]}_{role[i]}/{name}'];a['errors']+=values;a['count']+=len(values)
                                if hit[context][i,m]:a['pck16']+=sum(x<=16 for x in values);a['pck32']+=sum(x<=32 for x in values)
            output={}
            for key,a in accum.items():
                e=a.pop('errors');a.update(error_mean=float(np.mean(e)) if e else None,error_median=float(np.median(e)) if e else None,gated_pck16=a['pck16']/max(a['count'],1),gated_pck32=a['pck32']/max(a['count'],1));output[key]=a
            return output
        def loss(model,ids,flags):
            contexts=[f if isinstance(f,str) else 'single' if f else 'paired' for f in flags];allpred=forward_images(model,ids,contexts,detections);local,ms=valid[ids].nonzero().unbind(1);ims=torch.tensor(ids,device='cuda')[local]
            qs=torch.stack([mapping[contexts[int(j)]][int(i),int(m)] for j,i,m in zip(local,ims,ms)]);assert (qs>=0).all()
            pred=allpred[local,qs]
            return matched_image_point_loss(pred,data['gt_xy'][ims,ms],data['point_valid'][ims,ms],data['pixels'][ims],local,len(ids))
        rng=random.Random(seed+(15000 if advanced else 12000));schedule=[{k:[rng.choice(buckets[k][rng.choice(sorted(buckets[k]))]) for _ in range(8)] for k in buckets} for _ in range(STEPS)]
        schedule=balance_bottom_real_batches(schedule,meta,[i for i in train if kind[i]=='real'],seed+13000);flags=single_context_flags(schedule,meta,seed+14000)
        write_json(OUT/f'seed{seed}_schedule.json',dict(images=schedule,mixed_single_flags=flags,sha256=hashlib.sha256(json.dumps([schedule,flags],sort_keys=True).encode()).hexdigest()))
        if args.full_context_control:
            full_context_flags=full_view_contexts(schedule,flags,meta,seed+18000);write_json(OUT/f'seed{seed}_full_contexts.json',dict(contexts=full_context_flags,sha256=hashlib.sha256(json.dumps(full_context_flags,sort_keys=True).encode()).hexdigest()))
        if args.order_control:
            reverse_contexts=random_reference_contexts(schedule,flags,meta,seed+18000)
            write_json(OUT/f'seed{seed}_reference_contexts.json',dict(contexts=reverse_contexts,sha256=hashlib.sha256(json.dumps(reverse_contexts,sort_keys=True).encode()).hexdigest()))
        checkpoint=CYCLE/f'bottom_balanced_extended2400_v1/balanced_seed{seed}_latest.pt';cp=torch.load(checkpoint,map_location='cuda',weights_only=False);assert cp['seed']==seed and cp['step']==2400
        if advanced:
            checkpoint=CYCLE/f'view_context_control_v1/mixed_seed{seed}_latest.pt';cp=torch.load(checkpoint,map_location='cuda',weights_only=False);assert cp['seed']==seed and cp['step']==800 and cp['arm']=='mixed'
        if args.full_context_control:
            checkpoint=CYCLE/f'shared_token_pose_control_v1/shared_seed{seed}_latest.pt';cp=torch.load(checkpoint,map_location='cuda',weights_only=False);assert cp['seed']==seed and cp['step']==800 and cp['arm']=='shared'
        initial_reference=None
        for arm in (['paired','full'] if args.full_context_control else ['original','shared'] if args.shared_control else ['fixed','random_ref'] if args.order_control else ['paired','mixed']):
            torch.manual_seed(seed);model=DecoderSpatialPoseReadout().cuda().eval();model.load_state_dict(cp['state'],strict=True);model.set_decoder_trainable(True)
            optimizer=torch.optim.AdamW([dict(params=model.spatial_readout.parameters(),peak=5e-5,floor=1e-5),dict(params=model.keypoint_decoder.parameters(),peak=1e-5,floor=1e-6)],lr=5e-5,weight_decay=.001)
            initial=evaluate(model,include_train=True)
            if initial_reference is None:initial_reference=initial
            else:assert initial==initial_reference
            golden_path=CYCLE/('detection_context_pose_audit_v1/predictions.npz' if advanced else 'bottom_balanced_extended2400_v1/predictions.npz')
            if args.full_context_control:golden_path=CYCLE/'shared_token_pose_control_v1/predictions.npz'
            with torch.no_grad(),np.load(golden_path) as a:
                assert a['image_indices'].tolist()==validation
                goldcontext='shared' if args.full_context_control else 'paired';values=torch.cat([forward_images(model,validation[j:j+8],[goldcontext]*len(validation[j:j+8]),detections).cpu() for j in range(0,len(validation),8)])
                goldkey=f'shared_seed{seed}_shared' if args.full_context_control else f'mixed_seed{seed}_paired' if advanced else f'balanced_seed{seed}';delta=float((values-torch.tensor(a[goldkey])).abs().max());assert delta<2e-5
            write_json(OUT/f'{arm}_seed{seed}_initial.json',dict(checkpoint_sha256=fingerprint(checkpoint),paired_golden_delta=delta,metrics=initial));history=[dict(step=0,metrics=initial)]
            for step,(ids,mixed_flags) in enumerate(zip(schedule,flags),1):
                if dt.datetime.now(dt.timezone.utc)>=deadline:
                    torch.save(dict(state=model.state_dict(),optimizer=optimizer.state_dict(),step=step-1,seed=seed,arm=arm),OUT/f'{arm}_seed{seed}_deadline.pt');status('deadline_stopped');state.update(stage='training_deadline_reached_wrapup_required',active_job=None,active_pid=None);write_json(CYCLE/'state.json',state);return
                for g in optimizer.param_groups:g['lr']=g['floor']+(g['peak']-g['floor'])*.5*(1+math.cos(math.pi*(step-1)/STEPS))
                contexts=mixed_flags if arm=='mixed' else {k:[False]*8 for k in ids}
                if args.order_control:contexts=reverse_contexts[step-1] if arm=='random_ref' else mixed_flags
                if args.shared_control:contexts={k:['single' if f else 'shared' if arm=='shared' else 'paired' for f in mixed_flags[k]] for k in ids}
                if args.full_context_control:contexts=full_context_flags[step-1] if arm=='full' else {k:['single' if f else 'shared' for f in mixed_flags[k]] for k in ids}
                optimizer.zero_grad(set_to_none=True);syn=loss(model,ids['synthetic'],contexts['synthetic']);real=loss(model,ids['real'],contexts['real']);total=syn+.25*real;assert torch.isfinite(total);total.backward()
                if step==1:
                    forbidden=[name for name,p in model.named_parameters() if not p.requires_grad and p.grad is not None];assert not forbidden
                    norm=sum(float(p.grad.square().sum()) for p in model.keypoint_decoder.parameters() if p.grad is not None)**.5;assert norm>0
                    write_json(OUT/f'{arm}_seed{seed}_gradient_check.json',dict(forbidden=forbidden,decoder_norm=norm,all138queries=True,input_gradients=False))
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm);optimizer.step()
                if step%200==0:
                    metrics=evaluate(model,include_train=step==STEPS);row=dict(step=step,synthetic_loss=float(syn.detach()),real_loss=float(real.detach()),metrics=metrics);history.append(row)
                    write_json(OUT/f'{arm}_seed{seed}_history.json',history);torch.save(dict(state=model.state_dict(),optimizer=optimizer.state_dict(),step=step,seed=seed,arm=arm,initial_checkpoint_sha256=fingerprint(checkpoint)),OUT/f'{arm}_seed{seed}_latest.pt')
                    status('training',arm=arm,seed=seed,step=step);print('CONTEXT',arm,seed,step,'single_real',metrics['single/real_validation/reliable']['error_mean'],'paired_real',metrics['paired/real_validation/reliable']['error_mean'],flush=True)
            with torch.no_grad():
                for context in detections:saved[f'{arm}_seed{seed}_{context}']=torch.cat([forward_images(model,validation[j:j+8],[context]*len(validation[j:j+8]),detections).cpu() for j in range(0,len(validation),8)]).numpy()
            results.append(dict(arm=arm,seed=seed,**history[-1]));write_json(OUT/'partial_results.json',results);np.savez_compressed(OUT/'predictions.npz',image_indices=np.array(validation),**saved)
    write_json(OUT/'results.json',dict(results=results,no_production_promotion=True));status('complete')
    state.update(stage='view_context_control_results_ready',active_job=None,active_pid=None,updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),next='Compare paired/mixed800steps3seeds at fixed final, percontext/camera/reliable/dual and train vs validation; inspect raw allconfidence query overlays. If benefits credible, fullRGB1/2/all audit from new checkpoint, not only cached contexts. No production promotion or blind step extension.');state['completed'].append('view_context_control_3seeds800steps2arms');write_json(CYCLE/'state.json',state)
    if args.order_control:
        state['completed'][-1]='view_order_control_3seeds800steps2arms';state.update(stage='view_order_control_results_ready',next='Compare fixed vsrandom_ref800steps3seeds acrossoriginalpaired/reversedpaired/single contexts, percam andsyntheticdual, samefrozenmixeddetector. Inspect rawqueries; quantify reduction in ordergap separatelyfrom accuracy. Then fullRGB1/2/all andbothorders usingnewposeweights. No shared-token intervention, no GTbestorderselection, no promotion without multiviewretention.');write_json(CYCLE/'state.json',state)
    if args.shared_control:
        state['completed'][-1]='shared_token_pose_control_3seeds800steps2arms';state.update(stage='shared_token_pose_control_results_ready',next='Review shared-token poseadaptation3seeds800steps original/shared arms ineachsingle/paired/sharedcontext versusinitial, rawqueries, bottomfeet, dualsynthetic. Detectorweightsfixed butcontextpredictionsdiffer. Then defaultoff fullRGBoriginalgeometry plus2Dsharedbranch, same1/2/allbothorders and geometryidentitygate. No directreplacementoforiginalgeometry andnoauto promotion. Followplan.');write_json(CYCLE/'state.json',state)
    if args.full_context_control:
        state['completed'][-1]='full_context_pose_control_3seeds800steps2arms';state.update(stage='full_context_pose_control_results_ready',next='Review paired/fullcontextmatched3seed800steps, percontextreal/camera/dual andtrain/validation. Same1viewexposure, fullcontextfeaturesactualreencoded. No synthetic3plusviewTRAIN coverage. FullRGBshared2D+originalgeometry integration needsnewflag/goldens; followplan, noautopromotion.');write_json(CYCLE/'state.json',state)

if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write_json(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()))
        raise
