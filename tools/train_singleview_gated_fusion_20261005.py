"""Matched self versus cross-view residual, single-instance development pilot.

Frozen seed17 baseline, reliable IDs1/15-18 only. Other19? Exactly18 other
Core23 points remain unchanged. No GT query/crop/count enters model.forward.
"""
import collections,datetime as dt,fcntl,json,math,os,random,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from mousegeoforge.models.confidence_gated_view_residual import ConfidenceGatedViewResidual
from mousegeoforge.detection_readout_loss import match_detection
from mousegeoforge.matched_point_loss import matched_image_point_loss
from mousegeoforge.pilot_metrics import box_iou
from run_depth_pose_ablation import fingerprint,write_json
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';SOURCE=CYCLE/'singleview_fusion_inputs_seed17_v1';OUT=CYCLE/'gated_fusion_matched_pilot_v1';KP=[0,14,15,16,17];STEPS=800

def main():
    plan=json.loads((CYCLE/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'));lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not OUT.exists() and torch.cuda.is_available();OUT.mkdir();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='training_matched_single_instance_gated_fusion',active_job=OUT.name,active_pid=os.getpid());write_json(CYCLE/'state.json',state)
    def status(stage,**kw):write_json(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),**kw))
    status('loading');manifest=json.loads((SOURCE/'manifest.json').read_text());raw={};scenes={};scene_groups=json.loads((SOURCE/'scene_groups.json').read_text());cohort={}
    for part in manifest['parts']:
        role=part['role'];assert fingerprint(ROOT/part['path'])==part['sha256'];data=torch.load(ROOT/part['path'],map_location='cpu',weights_only=False)
        for m in data['metadata']:assert m['record']['role']==role and m['record'].get('session') not in ['GR2','UR2','UR3']
        for key,value in data.items():
            if isinstance(value,torch.Tensor):data[key]=value.cuda()
        data['quality']=torch.exp(-data['variance'].sum(-1).clamp_min(0).sqrt()/.05).clamp_min(1e-6);raw[role]=data
        groups=next(g['scenes'] for g in scene_groups if g['role']==role);eligible=[];omitted=[]
        for g in groups:
            ids=g['row_indices'];ms=[data['metadata'][i] for i in ids];expected={m['expected_instances'] for m in ms};assert len(expected)==1 and len(set(g['physical_views']))==len(ids)
            # Explicit experimental cohort selection, not runtime instance count.
            if expected=={1} and len(ids)>=2:eligible.append(dict(g,kind='real' if ms[0]['record']['kind']=='real' else 'synthetic'))
            else:omitted.append(dict(scene=g['scene'],images=len(ids),expected=sorted(expected)))
        scenes[role]=eligible;cohort[role]=dict(scenes=len(eligible),images=sum(len(g['row_indices']) for g in eligible),by_kind=dict(collections.Counter(g['kind'] for g in eligible)),excluded_scenes=omitted)
    write_json(OUT/'cohort.json',cohort)
    def batch(role,groups):
        data=raw[role];b=len(groups);v=max(len(g['row_indices']) for g in groups);ids=torch.zeros(b,v,dtype=torch.long,device='cuda');view_ids=torch.full((b,v),-1,dtype=torch.long,device='cuda');valid=torch.zeros(b,v,dtype=torch.bool,device='cuda')
        for j,g in enumerate(groups):
            count=len(g['row_indices']);ids[j,:count]=torch.tensor(g['row_indices'],device='cuda');view_ids[j,:count]=torch.tensor(g['physical_views'],device='cuda');valid[j,:count]=True
        inputs=dict(xy=data['xy'][ids][:,:,:,KP],tokens=data['tokens'][ids][:,:,:,KP],objectness_logits=data['objectness_logits'][ids],quality=data['quality'][ids][:,:,:,KP],view_ids=view_ids,view_valid=valid)
        return inputs,ids,valid
    def loss_for(role,groups,p,ids,actual):
        data=raw[role];flat=ids[actual];xy=p['xy'][actual];det=dict(instance_boxes_cxcywh=data['boxes'][flat],instance_objectness_logits=data['objectness_logits'][flat]);matches=match_detection(det,data['gt_boxes'][flat],data['box_valid'][flat]&data['instance_valid'][flat]);pred=[];gt=[];mask=[];pixels=[];image_ids=[]
        for j,(qs,ms) in enumerate(matches):
            for q,m in zip(qs.tolist(),ms.tolist()):pred.append(xy[j,q]);gt.append(data['gt_xy'][flat[j],m,KP]);mask.append(data['point_valid'][flat[j],m,KP]);pixels.append(data['pixels'][flat[j]]);image_ids.append(j)
        if not pred:return p['xy'].sum()*0
        return matched_image_point_loss(torch.stack(pred),torch.stack(gt),torch.stack(mask),torch.stack(pixels),torch.tensor(image_ids,device='cuda'),len(flat))
    @torch.no_grad()
    def predict(model,role):
        data=raw[role];xy=data['xy'].clone();active=torch.zeros(*xy.shape[:2],len(KP),device='cuda',dtype=torch.bool);gate=torch.zeros_like(active,dtype=xy.dtype);in_cohort=torch.zeros(len(xy),device='cuda',dtype=torch.bool)
        groups=scenes[role]
        for start in range(0,len(groups),8):
            args,ids,actual=batch(role,groups[start:start+8]);p=model(**args,enabled=True);flat=ids[actual]
            block=xy[flat].clone();block[:,:,KP]=p['xy'][actual];xy[flat]=block;active[flat]=p['active'][actual];gate[flat]=p['gate'][actual];in_cohort[flat]=True
        other=[i for i in range(23) if i not in KP];torch.testing.assert_close(xy[:,:,other],data['xy'][:,:,other],atol=0,rtol=0);torch.testing.assert_close(xy[~in_cohort],data['xy'][~in_cohort],atol=0,rtol=0)
        return dict(xy=xy,active=active,gate=gate,in_cohort=in_cohort)
    @torch.no_grad()
    def evaluate(model,role):
        pred=predict(model,role);data=raw[role];det=dict(instance_boxes_cxcywh=data['boxes'],instance_objectness_logits=data['objectness_logits']);matches=match_detection(det,data['gt_boxes'],data['box_valid']&data['instance_valid']);accum=collections.defaultdict(lambda:dict(errors=[],points=0,pck16=0,pck32=0,active_points=0,active_gates=[]))
        for i,m in enumerate(data['metadata']):
            if not pred['in_cohort'][i]:continue # Duals are outside this experiment, not credited as fusion successes.
            kind='real' if m['record']['kind']=='real' else 'synthetic';cam='cam'+chr(65+int(m['view']))
            for q,t in zip(*matches[i]):
                valid=data['point_valid'][i,t,KP];err=((pred['xy'][i,q,KP]-data['gt_xy'][i,t,KP])*data['pixels'][i]).norm(dim=-1);hit=bool(data['objectness_logits'][i,q].sigmoid()>=.5 and box_iou(data['boxes'][i,q:q+1],data['gt_boxes'][i,t:t+1])[0,0]>=.5)
                for name,select in [(kind+'/reliable',valid),(kind+'/'+cam+'/feet',valid&torch.tensor([False,True,True,True,True],device='cuda'))]:
                    a=accum[name];values=err[select];a['errors']+=values.tolist();a['points']+=len(values);a['active_points']+=int(pred['active'][i,q][select].sum());a['active_gates']+=pred['gate'][i,q][select&pred['active'][i,q]].tolist()
                    if hit:a['pck16']+=int((values<=16).sum());a['pck32']+=int((values<=32).sum())
        result={}
        for name,a in accum.items():
            errors=a.pop('errors');gates=a.pop('active_gates');a.update(error_mean=float(np.mean(errors)) if errors else None,gated_pck16=a['pck16']/max(a['points'],1),gated_pck32=a['pck32']/max(a['points'],1),active_fraction=a['active_points']/max(a['points'],1),active_gate_mean=float(np.mean(gates)) if gates else 0.);result[name]=a
        result['coverage']=dict(cohort_images=int(pred['in_cohort'].sum()),total_cached_images=len(data['metadata']),raw_active_query_points=int(pred['active'].sum()),outside_cohort_unchanged=True,other18points_unchanged=True,boxes_logits_unchanged=True)
        return result,pred
    train_real=[g for g in scenes['train'] if g['kind']=='real'];train_syn=[g for g in scenes['train'] if g['kind']=='synthetic'];rng=random.Random(17005);schedule=[dict(real=[rng.randrange(len(train_real)) for _ in range(4)],synthetic=[rng.randrange(len(train_syn)) for _ in range(4)]) for _ in range(STEPS)];write_json(OUT/'schedule.json',dict(steps=schedule,real_scenes=train_real,synthetic_scenes=train_syn))
    torch.manual_seed(17);initial=ConfidenceGatedViewResidual().state_dict();torch.save(initial,OUT/'initial.pt');write_json(OUT/'protocol.json',dict(steps=STEPS,seed=17,baseline_seed=17,arms=['self','multiview'],initial_sha256=fingerprint(OUT/'initial.pt'),source_manifest_sha256=fingerprint(SOURCE/'manifest.json'),module_sha256=fingerprint(ROOT/'mousegeoforge/models/confidence_gated_view_residual.py'),script_sha256=fingerprint(Path(__file__)),cohort='Known single-instance development scenes only, same cohort botharms; not multi-animal association. Dual scenes excluded from fusion, baseline outputs retained separately and not credited as fusionperformance.',forward='All6queries,5reliablepointsemantic tokens; no GTbox/query/count/visibility enters forward. Prediction-unique-query support only, physicalviewIDs no identity correspondence. Self control preserves supportavailability but uses own semanticfeatures. GTboxmatch afterforward for loss/eval.',loss='Equalweight per actual image, SmoothL1 beta.01 in pixels/768; real.25 + synthetic1. Back/tail/abdomen/other18points unchanged.',quality='exp(-sqrt(sum normalizedXYvariance)/.05), fixed heuristic not calibratedprobability or visibility; no validationlabel fit.',optimizer='AdamW fresh lr3e-4->3e-5 cosine wd.001 clip1. Fourreal+foursynthetic scenes perstep, same list forbotharms. Frozen base.',limitations='Seed17 pilot only; no RGBfusionintegration yet. Single is exactbaselinefallback, no traininggradient when no unique support. Shared frozen encoder passes cached independently, no compute-efficiencyclaim.',production_changed=False))
    results=[];saved={};initial_metrics=None
    for arm in ['self','multiview']:
        model=ConfidenceGatedViewResidual(mode=arm).cuda();model.load_state_dict(initial,strict=True);optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.001);baseline,p=evaluate(model,'validation');torch.testing.assert_close(p['xy'],raw['validation']['xy'],atol=0,rtol=0)
        if initial_metrics is None:initial_metrics=baseline
        else:assert baseline==initial_metrics
        history=[dict(step=0,metrics=baseline)];write_json(OUT/f'{arm}_initial.json',baseline);status('training',arm=arm,step=0)
        for step,selected in enumerate(schedule,1):
            if dt.datetime.now(dt.timezone.utc)>=deadline:
                torch.save(dict(state=model.state_dict(),optimizer=optimizer.state_dict(),step=step-1,arm=arm),OUT/f'{arm}_deadline.pt');status('deadline_stopped');state.update(stage='deadline_wrapup',active_job=None,active_pid=None);write_json(CYCLE/'state.json',state);return
            lr=3e-5+2.7e-4*.5*(1+math.cos(math.pi*(step-1)/STEPS))
            for g in optimizer.param_groups:g['lr']=lr
            optimizer.zero_grad(set_to_none=True);losses={}
            for kind,pool in [('real',train_real),('synthetic',train_syn)]:
                groups=[pool[i] for i in selected[kind]];args,ids,actual=batch('train',groups);p=model(**args,enabled=True);losses[kind]=loss_for('train',groups,p,ids,actual)
            loss=.25*losses['real']+losses['synthetic'];assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm);optimizer.step()
            if step==1:
                assert all(not v.requires_grad and v.grad is None for data in raw.values() for v in data.values() if isinstance(v,torch.Tensor));write_json(OUT/f'{arm}_gradient_check.json',dict(frozen_inputs=True,only_residual_parameters=sum(p.numel() for p in model.parameters()),first_gradient_norm=float(norm)))
            if step%200==0:
                metrics,p=evaluate(model,'validation');entry=dict(step=step,loss=float(loss.detach()),metrics=metrics);history.append(entry);write_json(OUT/f'{arm}_history.json',history);torch.save(dict(state=model.state_dict(),optimizer=optimizer.state_dict(),step=step,arm=arm,seed=17),OUT/f'{arm}_latest.pt');status('training',arm=arm,step=step);print('GATED',arm,step,metrics['real/reliable'],metrics['real/camA/feet'],flush=True)
        train_metrics,_=evaluate(model,'train');args,_,_=batch('validation',scenes['validation'][:4]);invariants={}
        with torch.no_grad():
            base=model(**args,enabled=True);assert model(**args,enabled=False)['xy'] is args['xy'];single={k:v[:,:1] for k,v in args.items()};assert model(**single,enabled=True)['xy'] is single['xy'];reverse={k:v.flip(1) for k,v in args.items()};rev=model(**reverse,enabled=True);delta=float((rev['xy'].flip(1)-base['xy']).abs().max());assert delta<2e-6;invariants['reverse_max_delta']=delta
            dup={k:torch.cat([v,v[:,1:2]],1) for k,v in args.items()};dp=model(**dup,enabled=True);delta=float((dp['xy'][:,:args['xy'].shape[1]]-base['xy']).abs().max());assert delta<2e-6;invariants['duplicate_max_delta']=delta
            blank=dict(args,quality=torch.zeros_like(args['quality']));torch.testing.assert_close(model(**blank,enabled=True)['xy'],args['xy'],atol=0,rtol=0)
        write_json(OUT/f'{arm}_invariants.json',dict(invariants,off_single_no_support_exact=True));saved[f'{arm}_xy']=p['xy'].cpu().numpy();saved[f'{arm}_active']=p['active'].cpu().numpy();saved[f'{arm}_gate']=p['gate'].cpu().numpy();saved['in_cohort']=p['in_cohort'].cpu().numpy();saved['image_indices']=np.array(raw['validation']['image_indices']);np.savez_compressed(OUT/'predictions.npz',**saved);results.append(dict(arm=arm,**history[-1],train_metrics=train_metrics,invariants=invariants));write_json(OUT/'partial_results.json',results)
    write_json(OUT/'results.json',dict(results=results,visual_review_pending=True,RGB_integration_pending=True,production_changed=False));status('complete');state.update(stage='matched_gated_fusion_pilot_ready',active_job=None,active_pid=None,updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),next='Review self vs multiview800 fixedseed17 curves and TRAIN/VAL gap, coverage andcamAfeet; no seedclaim. Renderfixed rawcamA/camD/synthetic single with baseline/self/multiview ALLconfidentqueries. Duals outsideexperiment, never call unchangedbaseline dualfusion. Checkview invariants and actualRGBV1->group cacheparity beforepromoting anybenefit. If pilotbetter, testviewcount/dropout stress and3seeds; otherwise diagnose gatecoverage/overfit orgeometricfusion missing, notblindlymoresteps.');state['completed'].append('matched_single_instance_fusion_pilot_seed17_800x2');write_json(CYCLE/'state.json',state)

if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write_json(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()))
        raise
