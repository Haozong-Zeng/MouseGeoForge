"""Bounded actual-V1 detector background control; no production default changes."""
import collections,datetime as dt,fcntl,json,math,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from mousegeoforge.models.spatial_detection_readout import SpatialDetectionReadout
from mousegeoforge.detection_negative_control import background_control_loss
from mousegeoforge.detection_readout_loss import match_detection,synthetic_detection_retention
from mousegeoforge.pilot_metrics import box_iou
from run_depth_pose_ablation import fingerprint,write_json
OLD=ROOT/'output/core23_spatial_cycle_20261004';CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';OUT=CYCLE/'single_detector_background_control_v1';STEPS=800

def main():
    plan=json.loads((CYCLE/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'))
    lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not OUT.exists() and torch.cuda.is_available();OUT.mkdir();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='training_pure_single_detector_background_control',active_job=OUT.name,active_pid=os.getpid());write_json(CYCLE/'state.json',state)
    def status(stage,**kw):write_json(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),**kw))
    status('loading')
    source=OLD/'detection_features_v1/manifest.json';rows=[]
    for record in json.loads(source.read_text())['records']:
        assert fingerprint(ROOT/record['path'])==record['sha256'];rows.extend(torch.load(ROOT/record['path'],map_location='cpu',weights_only=False)['observations'])
    meta=[o['metadata'] for o in rows];n=len(rows);assert n==1350
    sm=OLD/'single_detection_context_v1/manifest.json';manifest=json.loads(sm.read_text());assert fingerprint(ROOT/manifest['path'])==manifest['sha256'] and fingerprint(source)==manifest['source_sha256']
    raw=torch.load(ROOT/manifest['path'],map_location='cpu',weights_only=False);assert raw['metadata']==meta
    available=raw['available'].cuda();assert int(available.sum())==1335
    features={k:raw[k].float().cuda() for k in ['instance_tokens','roi','global_memory','original_boxes','original_objectness']}
    targets={k:torch.stack([o[k] for o in rows]).cuda() for k in ['boxes','box_valid','instance_valid']};valid=targets['box_valid']&targets['instance_valid']
    complete=torch.tensor([m['detection_complete'] for m in meta],device='cuda');sets=collections.defaultdict(list)
    synth_train=torch.tensor([m['record']['kind']!='real' and m['record']['role']=='train' and m['detection_complete'] for m in meta],device='cuda')
    for i,m in enumerate(meta):
        r=m['record'];assert r.get('session') not in ['GR2','UR2','UR3']
        if not complete[i] or not available[i]:continue
        kind='real' if r['kind']=='real' else 'synthetic';prefix=kind+'_'+r['role'];sets[prefix].append(i);sets[prefix+('/dual' if m['expected_instances']>1 else '/single')].append(i)
        if kind=='real':sets[prefix+'/cam'+chr(65+int(m['view']))].append(i)
    assert len(sets['real_train'])==1028 and len(sets['synthetic_train'])==195 and len(sets['real_validation'])==48 and len(sets['synthetic_validation'])==64
    del rows,raw
    def forward(model,ids):return model(features['instance_tokens'][ids],features['roi'][ids],features['global_memory'][ids],features['original_boxes'][ids],features['original_objectness'][ids])
    @torch.no_grad()
    def predict(model):
        pieces=collections.defaultdict(list)
        for start in range(0,n,128):
            for k,v in forward(model,slice(start,start+128)).items():pieces[k].append(v)
        return {k:torch.cat(v) for k,v in pieces.items()}
    @torch.no_grad()
    def evaluate(p,include_train=False):
        metrics={}
        for group,ids in sets.items():
            if not include_train and '_train' in group:continue
            pp={k:v[ids] for k,v in p.items()};matches=match_detection(pp,targets['boxes'][ids],valid[ids]);tp=gt=pred=dups=0;errors=[];ious=[]
            for j,i in enumerate(ids):
                qs,ms=matches[j];selected=pp['instance_objectness_logits'][j].sigmoid()>=.5;pred+=int(selected.sum());gt+=len(ms);errors.append(abs(int(selected.sum())-len(ms)));matrix=box_iou(pp['instance_boxes_cxcywh'][j],targets['boxes'][i,ms])
                for q,m in zip(qs.tolist(),ms.tolist()):
                    iou=float(matrix[q,ms.tolist().index(m)]);ious.append(iou);tp+=int(iou>=.5 and bool(selected[q]))
                if len(ms):dups+=int(((matrix[selected]>=.5).sum(0)-1).clamp_min(0).sum())
            metrics[group]=dict(images=len(ids),tp=tp,gt=gt,predictions=pred,precision=tp/max(pred,1),recall=tp/max(gt,1),f1=2*tp/max(pred+gt,1),count_absolute_error=float(np.mean(errors)),duplicate_box_hits=dups,matched_iou_mean=float(np.mean(ious)))
        return metrics
    protocol=dict(steps=STEPS,seeds=[17,29,43],arms=dict(control=0.,negative04=.4),inputs='All images genuinely V1-encoded; no paired-token slicing. Encoder/pose/instance decoder frozen. All6 predicted ROIs, notGT.',initial='Same per-seed detection_view_context_control_v1/mixed800, chosen before this experiment.',difference='Only additional unmatched-query negative BCE coefficient .4. Original .1 background, positive1, normalization, boxL1/GIoU and synthetic retention unchanged. No NMS or threshold selection.',schedule='Exact previous full_context_pose_control_v1 image schedules (8real balanced4bottom/4other+8synthetic), all contexts changed to actualsingle for BOTHarms.',optimizer='AdamW fresh,wd.001,cosine1e-4to1e-5,clip1',source_sha256=fingerprint(source),single_manifest_sha256=fingerprint(sm),script_sha256=fingerprint(Path(__file__)),loss_sha256=fingerprint(ROOT/'mousegeoforge/detection_negative_control.py'),validation='48real+64synthetic, same development split, lockedGR2UR2UR3excluded. Complete1350pool1335available; incomplete15notnegatives. Confidence.5/IoU.5, GTassignment evaluationonly, duplicate overlap counts may include neighboringtrue mouse.',production_changed=False)
    write_json(OUT/'protocol.json',protocol);results=[];saved={}
    with np.load(OLD/'detection_view_context_control_v1/predictions.npz') as archive:gold={s:{k:torch.tensor(archive[f'mixed_seed{s}_single__{k}'],device='cuda') for k in ['instance_boxes_cxcywh','instance_objectness_logits']} for s in [17,29,43]}
    for seed in [17,29,43]:
        schedule_path=OLD/f'full_context_pose_control_v1/seed{seed}_schedule.json';schedule=json.loads(schedule_path.read_text())['images'];assert len(schedule)==STEPS
        for batch in schedule:
            for domain,ids in batch.items():
                assert len(ids)==8 and complete[ids].all() and available[ids].all()
                assert all(meta[i]['record']['role']=='train' and ('real' if meta[i]['record']['kind']=='real' else 'synthetic')==domain for i in ids)
            assert sum(int(meta[i]['view'])==0 for i in batch['real'])==4
        write_json(OUT/f'seed{seed}_schedule.json',dict(images=schedule,source_sha256=fingerprint(schedule_path),context='single_only'))
        checkpoint=OLD/f'detection_view_context_control_v1/mixed_seed{seed}_latest.pt';cp=torch.load(checkpoint,map_location='cuda',weights_only=False);assert cp['step']==800 and cp['seed']==seed and cp['arm']=='mixed';initial=None
        for arm,weight in [('control',0.),('negative04',.4)]:
            torch.manual_seed(seed);model=SpatialDetectionReadout().cuda().eval();model.load_state_dict(cp['state'],strict=True);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.001)
            p=predict(model);delta=max(float((p[k][available]-gold[seed][k][available]).abs().max()) for k in p);assert delta<2e-5
            base=evaluate(p,True)
            if initial is None:initial=base
            else:assert initial==base
            history=[dict(step=0,metrics=base)];write_json(OUT/f'{arm}_seed{seed}_initial.json',dict(max_delta=delta,checkpoint_sha256=fingerprint(checkpoint),metrics=base));status('training',seed=seed,arm=arm,step=0)
            for step,batch in enumerate(schedule,1):
                if dt.datetime.now(dt.timezone.utc)>=deadline:
                    torch.save(dict(state=model.state_dict(),step=step-1,seed=seed,arm=arm,optimizer=optimizer.state_dict()),OUT/f'{arm}_seed{seed}_deadline.pt');status('deadline_stopped');state.update(stage='deadline_wrapup',active_job=None,active_pid=None);write_json(CYCLE/'state.json',state);return
                ids=batch['real']+batch['synthetic'];lr=1e-5+9e-5*.5*(1+math.cos(math.pi*(step-1)/STEPS))
                for g in optimizer.param_groups:g['lr']=lr
                optimizer.zero_grad(set_to_none=True);p=forward(model,ids);loss,logs=background_control_loss(p,targets['boxes'][ids],valid[ids],complete[ids],additional_negative_weight=weight);retain,_=synthetic_detection_retention(p,features['original_boxes'][ids],features['original_objectness'][ids],synth_train[ids]);loss=loss+retain;assert torch.isfinite(loss);loss.backward()
                if step==1:
                    assert all(not v.requires_grad and v.grad is None for v in features.values());write_json(OUT/f'{arm}_seed{seed}_gradient_check.json',dict(input_gradients=False,only_ROI_detector_trainable=True,parameters=sum(p.numel() for p in model.parameters())))
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm);optimizer.step()
                if step%200==0:
                    p=predict(model);metrics=evaluate(p,step==STEPS);entry=dict(step=step,loss=float(loss.detach()),additional_negative=float(logs['additional_negative'].detach()),metrics=metrics);history.append(entry);write_json(OUT/f'{arm}_seed{seed}_history.json',history);torch.save(dict(state=model.state_dict(),optimizer=optimizer.state_dict(),step=step,seed=seed,arm=arm,initial_sha256=fingerprint(checkpoint)),OUT/f'{arm}_seed{seed}_latest.pt');status('training',seed=seed,arm=arm,step=step);print('SINGLE_BG',seed,arm,step,metrics['real_validation'],metrics['synthetic_validation/dual'],flush=True)
            for key,value in p.items():saved[f'{arm}_seed{seed}__{key}']=value.cpu().numpy()
            results.append(dict(seed=seed,arm=arm,**history[-1]));write_json(OUT/'partial_results.json',results);np.savez_compressed(OUT/'predictions.npz',**saved)
    write_json(OUT/'results.json',dict(results=results,downstream_pose_audit_pending=True,production_changed=False));status('complete');state.update(stage='single_detector_control_ready',active_job=None,active_pid=None,updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),next='Compare3seed detection control/negative04 against sameinitial, both pureV1. Then frozen full_context_pose_control_v1/full800 pose audit with new predicted boxes, all138queries and posthocGTbox association: conditional/samequery errors, queryswitch, gatedPCK, counts/precision. Do not select from F1 alone. Build rawcamA/camD/dual overlays and fullRGBV1goldens before anycandidate promotion. Then proceed singleviewlocalization andoptional gatedfusion perplan; no automatic head-only reruns.');state['completed'].append('pure_single_detector_background_800x3x2');write_json(CYCLE/'state.json',state)

if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write_json(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()))
        raise
