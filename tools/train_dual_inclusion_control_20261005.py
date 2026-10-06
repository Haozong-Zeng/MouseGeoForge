"""Fixed-head data inclusion control: replace 2/16 images with dual TRAIN."""
import datetime as dt,fcntl,json,math,os,random,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from mousegeoforge.models.prior_candidate_selector import PriorCandidateSelector
from mousegeoforge.candidate_selector_features import KP
from mousegeoforge.detection_readout_loss import match_detection
from mousegeoforge.pilot_metrics import box_iou
from audit_independent_unary_20261005 import metrics
from run_depth_pose_ablation import fingerprint,write_json
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';SOURCE=CYCLE/'local_candidate_selector_pilot_v1';OUT=CYCLE/'dual_train_inclusion_control_v1'


def prepare_supervision(d):
    # Explicit independent per-query eligibility, no GT-count or camera gate.
    inputs=dict(d['inputs']);inputs['geometry']=torch.zeros_like(inputs['geometry'])
    inputs['eligible']=(d['logits'].sigmoid()>=.5)[...,None].expand(-1,-1,5)
    # Labels and Hungarian assignment below enter loss only, not inputs.
    matches=match_detection(dict(instance_boxes_cxcywh=d['boxes'],instance_objectness_logits=d['logits']),d['gt_boxes'],d['box_valid']&d['instance_valid'])
    target=torch.zeros(*inputs['eligible'].shape,6);valid=torch.zeros_like(inputs['eligible'])
    for i,(qs,ts) in enumerate(matches):
        for qq,tt in zip(qs,ts):
            q,t=int(qq),int(tt);good=d['point_valid'][i,t,KP]&inputs['eligible'][i,q]
            gt=torch.where(good[:,None],d['gt_xy'][i,t,KP],0)
            error2=((inputs['candidate_xy'][i,q]-gt[:,None])*d['pixels'][i]).square().sum(-1)
            target[i,q]=(-error2/(2*16**2)).softmax(-1);valid[i,q]=good
    return dict(d,inputs=inputs,target_prob=target,train_valid=valid)


def main():
    plan=json.loads((CYCLE/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'));lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not OUT.exists();OUT.mkdir();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='dual_train_inclusion_control_running',active_job=OUT.name,active_pid=os.getpid());write_json(CYCLE/'state.json',state)
    def status(stage,**kw):write_json(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),**kw))
    status('preparing');start=time.monotonic()
    paths=dict(train_single=SOURCE/'train_inputs.pt',train_dual=CYCLE/'dual_train_candidate_audit_v1/train_inputs.pt',validation_single=SOURCE/'validation_inputs.pt',validation_dual=CYCLE/'dual_unary_diagnostic_v1/inputs.pt')
    protocol=dict(steps=800,arms=['single_only','dual_mixed'],head='Same PriorCandidateSelector unary bound2,22657params; frozen baseline17 and candidates; no geometry.',initial=fingerprint(SOURCE/'initial.pt'),sources={k:dict(path=p.relative_to(ROOT).as_posix(),sha256=fingerprint(p)) for k,p in paths.items()},schedule='Original seed17 control schedule4realbottom+4otherreal+8syntheticsingle. Mixed first14 identical, last2 replaced by existing dualTRAIN42, independent RNG17042. Exactly12.5% image budget; no extra steps.',loss='Recomputed soft labels sigma16 after frozen box matching; independent per-query predictedscore>=.5 for BOTH arms. Equal image CE averages available matched points (including both instances), no GTcount in forward, no inherited multiview-gated train_valid. Unmatched queries no pose label; detector frozen.',optimizer='AdamW .001 cosine->.0001,wd.001,clip1',selection='Final800 only, no validation selection; reused development datasets not independent-video tests.',model_sha256=fingerprint(ROOT/'mousegeoforge/models/prior_candidate_selector.py'))
    write_json(OUT/'protocol.json',protocol)
    cpu={k:prepare_supervision(torch.load(p,map_location='cpu',weights_only=False)) for k,p in paths.items()}
    assert len(cpu['train_dual']['base_xy'])==42 and len(cpu['validation_dual']['base_xy'])==32
    train_scenes={m['scene'] for tag,d in cpu.items() if tag.startswith('train_') for m in d['metadata']};val_scenes={m['scene'] for tag,d in cpu.items() if tag.startswith('validation_') for m in d['metadata']};assert not train_scenes&val_scenes
    for name,d in cpu.items():
        expected='train' if name.startswith('train_') else 'validation';assert all(m['role']==expected for m in d['metadata'])
    write_json(OUT/'supervision_summary.json',{k:dict(images=len(d['base_xy']),valid_points=int(d['train_valid'].sum()),eligible_predicted_points=int(d['inputs']['eligible'].sum())) for k,d in cpu.items()})
    torch.save({k:dict(target_prob=d['target_prob'],train_valid=d['train_valid'],eligible=d['inputs']['eligible']) for k,d in cpu.items()},OUT/'recomputed_supervision.pt')
    single,dual=cpu['train_single'],cpu['train_dual'];offset=len(single['base_xy'])
    inputs={k:torch.cat([single['inputs'][k],dual['inputs'][k]]).cuda() for k in single['inputs']};target=torch.cat([single['target_prob'],dual['target_prob']]).cuda();valid=torch.cat([single['train_valid'],dual['train_valid']]).cuda()
    control=json.loads((SOURCE/'schedule.json').read_text());rng=random.Random(17042);mixed=[ids[:14]+[offset+j for j in rng.choices(range(42),k=2)] for ids in control]
    assert all(a[:14]==b[:14] and len(b)==16 for a,b in zip(control,mixed));write_json(OUT/'schedules.json',dict(single_only=control,dual_mixed=mixed,dual_offset=offset))
    initial=torch.load(SOURCE/'initial.pt',map_location='cpu',weights_only=True)
    for arm,schedule in [('single_only',control),('dual_mixed',mixed)]:
        model=PriorCandidateSelector('unary').cuda();model.load_state_dict(initial,strict=True);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001);history=[];model.train()
        for step,ids in enumerate(schedule,1):
            assert dt.datetime.now(dt.timezone.utc)<deadline
            lr=.0001+.5*(.001-.0001)*(1+math.cos(math.pi*(step-1)/800))
            for group in optimizer.param_groups:group['lr']=lr
            optimizer.zero_grad(set_to_none=True);out=model(**{k:v[ids] for k,v in inputs.items()},enabled=True);mask=valid[ids]&out['eligible'];point=-(target[ids]*out['logits'].log_softmax(-1)).sum(-1);loss=((point*mask).sum((-1,-2))/mask.sum((-1,-2)).clamp_min(1)).mean();assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();history.append(dict(step=step,loss=float(loss.detach()),lr=lr))
            if step%100==0:
                write_json(OUT/f'{arm}_history.json',history);torch.save(dict(arm=arm,step=step,state={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),residual_bound=2),OUT/f'{arm}_latest.pt');status('training',arm=arm,step=step,loss=history[-1]['loss']);print(arm,step,history[-1]['loss'],flush=True)
    # Both arms trained to predeclared endpoint before development evaluation.
    results=[];arrays={};counts={}
    for tag,d in cpu.items():
        matches=match_detection(dict(instance_boxes_cxcywh=d['boxes'],instance_objectness_logits=d['logits']),d['gt_boxes'],d['box_valid']&d['instance_valid']);tp=0
        for i,(qs,ts) in enumerate(matches):
            for q,t in zip(qs,ts):tp+=int(d['logits'][i,q].sigmoid()>=.5 and box_iou(d['boxes'][i,q:q+1],d['gt_boxes'][i,t:t+1])[0,0]>=.5)
        counts[tag]=dict(images=len(d['base_xy']),gt=int((d['box_valid']&d['instance_valid']).sum()),pred=int((d['logits'].sigmoid()>=.5).sum()),tp=tp)
        for arm in ['baseline','single_only','dual_mixed']:
            model=PriorCandidateSelector('unary').cuda().eval()
            if arm!='baseline':model.load_state_dict(torch.load(OUT/f'{arm}_latest.pt',map_location='cpu',weights_only=False)['state'])
            xy=d['base_xy'].clone();active=torch.zeros_like(d['train_valid']);choice=torch.zeros_like(active,dtype=torch.long);losses=[]
            with torch.no_grad():
                for first in range(0,len(xy),32):
                    ids=list(range(first,min(first+32,len(xy))));b={k:v[ids].cuda() for k,v in d['inputs'].items()};out=model(**b,enabled=arm!='baseline')
                    assert model(**b)['xy'] is b['baseline_xy']
                    block=xy[ids].clone();block[:,:,KP]=out['xy'].cpu();xy[ids]=block;active[ids]=out['active'].cpu()
                    if arm!='baseline':
                        choice[ids]=out['choice'].cpu();p=-(d['target_prob'][ids]*out['logits'].cpu().log_softmax(-1)).sum(-1);m=d['train_valid'][ids]&out['eligible'].cpu();losses.extend(((p*m).sum((-1,-2))/m.sum((-1,-2)).clamp_min(1)).tolist())
            values,points=metrics(d,xy,active,choice)
            for p in points:
                if not tag.endswith('dual'):continue
                i,q,t,k=p['row'],p['query'],p['target'],p['point']-1;other=1-t
                if d['point_valid'][i,other,k]:
                    other_error=float(((xy[i,q,k]-d['gt_xy'][i,other,k])*d['pixels'][i]).norm());old_other=float(((d['base_xy'][i,q,k]-d['gt_xy'][i,other,k])*d['pixels'][i]).norm());p['other_instance_same_point_error']=other_error;p['other_closer_than_matched']=other_error<p['error'];p['new_other_closer']=other_error<p['error'] and not old_other<p['baseline_error']
            fulltag=arm+'_'+tag;arrays[fulltag]=xy.numpy();arrays[fulltag+'_active']=active.numpy();write_json(OUT/f'{fulltag}_points.json',points)
            results.append(dict(arm=arm,dataset=tag,metrics=values,full_image_mean_ce=float(np.mean(losses)) if losses else None,other_nearer=sum(p.get('other_closer_than_matched',False) for p in points),new_other_nearer=sum(p.get('new_other_closer',False) for p in points)))
            print(fulltag,{k:round(v['mean_error'],3) for k,v in values.items() if k in ['real/reliable','real/view0/feet','synthetic/reliable']},flush=True)
    np.savez_compressed(OUT/'predictions.npz',**arrays);write_json(OUT/'results.json',dict(results=results,detector_counts_unchanged=counts,protocol=protocol,elapsed_seconds=time.monotonic()-start,script_sha256=fingerprint(Path(__file__)),production_changed=False));status('complete');state.update(stage='dual_train_inclusion_control_complete',active_job=None,active_pid=None,next='Inspect dual_train_inclusion_control_v1 allgroup outcomes/losses and raw cross-instance overlays. Botharms800 fixedhead andnewper-querygate. No checkpoint promotion; record next based on TRAIN/VAL tradeoff.');state['completed'].append('dual_train_inclusion_control_800x2');write_json(CYCLE/'state.json',state)


if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write_json(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()))
        raise
