"""Matched 800-step unary/geometry candidate ranking pilot, frozen baseline17."""
import collections,datetime as dt,fcntl,json,math,os,random,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from torch.nn import functional as F
from mousegeoforge.models.local_candidate_selector import LocalCandidateSelector
from mousegeoforge.candidate_selector_features import build_selector_inputs,SCALAR_NAMES,GEOMETRY_NAMES,KP
from mousegeoforge.detection_readout_loss import match_detection
from mousegeoforge.pilot_metrics import box_iou
from run_depth_pose_ablation import fingerprint,write_json
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';OUT=CYCLE/'local_candidate_selector_pilot_v1';STEPS=800


def main():
    plan=json.loads((CYCLE/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'));lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not OUT.exists() and torch.cuda.is_available();OUT.mkdir();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='training_local_candidate_selector',active_job=OUT.name,active_pid=os.getpid());write_json(CYCLE/'state.json',state)
    def status(stage,**kw):write_json(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),**kw))
    status('preparing');started=time.monotonic();records=[];source_hashes={}
    for directory in ['cached_local_candidate_inputs_v1','synthetic_full_candidate_context_v1']:
        path=CYCLE/directory/'manifest.json';source_hashes[directory]=fingerprint(path)
        for r in json.loads(path.read_text())['records']:records.append(dict(r,kind=r.get('kind','synthetic')))
    protocol=dict(steps=STEPS,seed=17,arms=['unary','geometry'],optimizer='AdamW',lr_start=.001,lr_end=.0001,weight_decay=.001,gradient_clip=1,
        batch='4 realcamA +4 otherreal +8synthetic TRAIN images, with replacement; identical image schedule and initialweights botharms.',
        loss='Equal-image soft-target cross-entropy. Target weights softmax(-pixel_distance_squared/(2*16^2)) over six prediction-only candidates; labels/matches only in loss. Ignore absent labels, no fake visibility. No validation checkpoint or threshold selection.',
        features=dict(scalars=SCALAR_NAMES,geometry=GEOMETRY_NAMES,sigma_px=16,minimum_distinct_other_support=2),
        frozen='Encoder, instance decoder, originaldet, mixed17ROI, full17pose, NMS candidates. Only newselector trained. Candidate0 originalmean; all6queries kept. Unknownmulti-instance correspondence abstains. Back/tail/detection unchanged.',source_sha256=source_hashes)
    write_json(OUT/'protocol.json',protocol)
    parts=collections.defaultdict(list);roles={}
    for ri,r in enumerate(records):
        assert dt.datetime.now(dt.timezone.utc)<deadline
        assert fingerprint(ROOT/r['path'])==r['sha256'];s=torch.load(ROOT/r['path'],map_location='cpu',weights_only=False);role=s['role'];assert roles.setdefault(s['scene'],role)==role
        assert s['record'].get('session') not in ['GR2','UR2','UR3'];p=s['predictions'];t=s['targets'];kind=r['kind'];v=len(s['physical_views'])
        # Construct all forward inputs before touching supervision arrays.
        features=build_selector_inputs(p,s['cameras']['projections'],s['physical_views'],s['pixels']);features['local']=features['local'].half()
        matches=match_detection(dict(instance_boxes_cxcywh=p['boxes'],instance_objectness_logits=p['objectness_logits']),t['boxes_cxcywh'],t['box_valid']&t['instance_valid'])
        target_prob=torch.zeros(v,6,5,6);train_valid=torch.zeros(v,6,5,dtype=torch.bool)
        for vi,(qs,ts) in enumerate(matches):
            for qq,tt in zip(qs,ts):
                q,inst=int(qq),int(tt);good=t['keypoint_valid'][vi,inst,KP]&features['eligible'][vi,q]
                gt=t['keypoint_xy'][vi,inst,KP];safe=torch.where(good[:,None],gt,0)
                error2=((features['candidate_xy'][vi,q]-safe[:,None])*torch.as_tensor(s['pixels'])).square().sum(-1)
                target_prob[vi,q]=(-error2/(2*16**2)).softmax(-1);train_valid[vi,q]=good
        for vi,physical in enumerate(s['physical_views']):
            parts[role].append(dict(inputs={k:x[vi] for k,x in features.items()},target_prob=target_prob[vi],train_valid=train_valid[vi],base_xy=p['xy'][vi],boxes=p['boxes'][vi],logits=p['objectness_logits'][vi],gt_xy=t['keypoint_xy'][vi],point_valid=t['keypoint_valid'][vi],gt_boxes=t['boxes_cxcywh'][vi],box_valid=t['box_valid'][vi],instance_valid=t['instance_valid'][vi],pixels=torch.as_tensor(s['pixels']),metadata=dict(scene=s['scene'],view=physical,kind=kind,role=role,source_path=r['path'],source_view_index=vi)))
        if (ri+1)%30==0:status('preparing',scenes=ri+1,total=len(records));print('prepared',ri+1,'/',len(records),flush=True)
    assert len(parts['train'])==1489 and len(parts['validation'])==80
    data={}
    for role,items in parts.items():
        packed={k:torch.stack([r[k] for r in items]) for k in items[0] if k not in ['inputs','metadata']};packed['inputs']={k:torch.stack([r['inputs'][k] for r in items]) for k in items[0]['inputs']};packed['metadata']=[r['metadata'] for r in items]
        torch.save(packed,OUT/f'{role}_inputs.pt');data[role]={k:(v.cuda() if isinstance(v,torch.Tensor) else {kk:vv.cuda() for kk,vv in v.items()} if k=='inputs' else v) for k,v in packed.items()}
    del parts,items,packed
    train=data['train'];rng=random.Random(17);bottom=[i for i,m in enumerate(train['metadata']) if m['kind']=='real' and m['view']==0];other=[i for i,m in enumerate(train['metadata']) if m['kind']=='real' and m['view']!=0];synthetic=[i for i,m in enumerate(train['metadata']) if m['kind']=='synthetic'];assert bottom and other and synthetic
    schedule=[rng.choices(bottom,k=4)+rng.choices(other,k=4)+rng.choices(synthetic,k=8) for _ in range(STEPS)];write_json(OUT/'schedule.json',schedule)
    torch.manual_seed(17);model0=LocalCandidateSelector().cuda();initial={k:v.detach().cpu().clone() for k,v in model0.state_dict().items()};torch.save(initial,OUT/'initial.pt');parameters=sum(p.numel() for p in model0.parameters());del model0
    def batch_inputs(role,ids):return {k:v[ids] for k,v in data[role]['inputs'].items()}
    def loss_for(pred,role,ids):
        d=data[role];valid=d['train_valid'][ids]&pred['eligible'];point=-(d['target_prob'][ids]*pred['logits'].log_softmax(-1)).sum(-1)
        image=(point*valid).sum((-1,-2))/valid.sum((-1,-2)).clamp_min(1);return image.mean()
    @torch.no_grad()
    def evaluate(model,role):
        d=data[role];xy=d['base_xy'].clone();active=torch.zeros(len(xy),6,5,dtype=torch.bool,device='cuda');choices=torch.zeros_like(active,dtype=torch.long);losses=[]
        for start in range(0,len(xy),32):
            ids=list(range(start,min(len(xy),start+32)));out=model(**batch_inputs(role,ids),enabled=True);block=xy[ids].clone();block[:,:,KP]=out['xy'];xy[ids]=block;active[ids]=out['active'];choices[ids]=out['choice'];losses.append((len(ids),float(loss_for(out,role,ids))))
        matches=match_detection(dict(instance_boxes_cxcywh=d['boxes'],instance_objectness_logits=d['logits']),d['gt_boxes'],d['box_valid']&d['instance_valid']);acc=collections.defaultdict(lambda:dict(errors=[],hit=[],changed=0));allpoints=[]
        for i,m in enumerate(d['metadata']):
            for qq,tt in zip(*matches[i]):
                q,t=int(qq),int(tt);hit=bool(d['logits'][i,q].sigmoid()>=.5 and box_iou(d['boxes'][i,q:q+1],d['gt_boxes'][i,t:t+1])[0,0]>=.5)
                for ki,kp in enumerate(KP):
                    if not d['point_valid'][i,t,kp]:continue
                    error=float(((xy[i,q,kp]-d['gt_xy'][i,t,kp])*d['pixels'][i]).norm());baseline=float(((d['base_xy'][i,q,kp]-d['gt_xy'][i,t,kp])*d['pixels'][i]).norm());changed=bool(active[i,q,ki]);names=[m['kind']+'/reliable']+([m['kind']+'/view'+str(m['view'])+'/feet'] if kp else [])
                    for name in names:a=acc[name];a['errors'].append(error);a['hit'].append(hit and error<=32);a['changed']+=changed
                    allpoints.append(dict(row=i,point=kp+1,query=q,target=t,error=error,baseline_error=baseline,changed=changed,choice=int(choices[i,q,ki])))
        return dict(loss=sum(n*l for n,l in losses)/len(xy),metrics={name:dict(points=len(a['errors']),mean_error=float(np.mean(a['errors'])),gated_pck32=float(np.mean(a['hit'])),changed_points=a['changed']) for name,a in acc.items()}),xy.cpu().numpy(),active.cpu().numpy(),allpoints
    results=[];predictions={}
    for arm in ['unary','geometry']:
        model=LocalCandidateSelector(arm).cuda();model.load_state_dict(initial,strict=True)
        test=batch_inputs('validation',list(range(8)));out=model(**test,enabled=True);torch.testing.assert_close(out['xy'],test['baseline_xy'],atol=0,rtol=0);assert model(**test)['xy'] is test['baseline_xy']
        if arm=='unary':
            for role in ['train','validation']:
                metrics,xy,active,points=evaluate(model,role);results.append(dict(arm='baseline',role=role,step=0,**metrics));predictions[role+'_baseline']=xy
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001);history=[];model.train()
        for step,ids in enumerate(schedule,1):
            assert dt.datetime.now(dt.timezone.utc)<deadline
            lr=.0001+.5*(.001-.0001)*(1+math.cos(math.pi*(step-1)/STEPS))
            for group in optimizer.param_groups:group['lr']=lr
            optimizer.zero_grad(set_to_none=True);out=model(**batch_inputs('train',ids),enabled=True);loss=loss_for(out,'train',ids);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();history.append(dict(step=step,loss=float(loss),lr=lr))
            if step%100==0:
                write_json(OUT/f'{arm}_history.json',history);torch.save(dict(arm=arm,step=step,state={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict()),OUT/f'{arm}_latest.pt');status('training',arm=arm,step=step,loss=float(loss));print(arm,step,float(loss),flush=True)
        model.eval();off=model(**test);assert off['xy'] is test['baseline_xy'];invalid=dict(test,eligible=torch.zeros_like(test['eligible']));torch.testing.assert_close(model(**invalid,enabled=True)['xy'],test['baseline_xy'],atol=0,rtol=0)
        for role in ['train','validation']:
            metrics,xy,active,points=evaluate(model,role);results.append(dict(arm=arm,role=role,step=STEPS,**metrics));predictions[role+'_'+arm]=xy;predictions[role+'_'+arm+'_active']=active;write_json(OUT/f'{arm}_{role}_points.json',points)
        write_json(OUT/'partial_results.json',dict(results=results));print(arm,'complete',flush=True)
    np.savez_compressed(OUT/'predictions.npz',**predictions);write_json(OUT/'results.json',dict(results=results,parameters=parameters,protocol=protocol,script_sha256=fingerprint(Path(__file__)),feature_builder_sha256=fingerprint(ROOT/'mousegeoforge/candidate_selector_features.py'),model_sha256=fingerprint(ROOT/'mousegeoforge/models/local_candidate_selector.py'),elapsed_seconds=time.monotonic()-started,train_eligible_points=int(train['train_valid'].sum()),input_sha256={role:fingerprint(OUT/f'{role}_inputs.pt') for role in data},production_changed=False))
    status('complete');state.update(stage='candidate_selector_matched_pilot_complete',active_job=None,active_pid=None,next='Inspect local_candidate_selector_pilot_v1 results, convergence and changedpoint harm; raw overlays and viewstress before any expansion/promotion. No VALthresholdsearch, originals unchanged.');state['completed'].append('matched_local_candidate_selector_800x2');write_json(CYCLE/'state.json',state)


if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write_json(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()))
        raise
