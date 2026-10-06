"""Native-V1 full vs structured-union zero-residual decoder control."""
import collections,datetime as dt,fcntl,json,math,os,random,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from mousegeoforge.models.decoder_spatial_pose_readout import DecoderSpatialPoseReadout
from mousegeoforge.models.topology_pose_adapter import TopologyPoseAdapter
from mousegeoforge.detection_readout_loss import match_detection
from mousegeoforge.matched_point_loss import matched_image_point_loss
from mousegeoforge.pilot_metrics import box_iou
from audit_independent_unary_20261005 import metrics
from run_depth_pose_ablation import fingerprint,write_json
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';OLD=ROOT/'output/core23_spatial_cycle_20261004';OUT=CYCLE/'native_topology_control_v2';KP=[0,14,15,16,17];KEEP=[k for k in range(23) if k not in KP]


def main():
    plan=json.loads((CYCLE/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'));lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not OUT.exists();OUT.mkdir();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=False
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='native_topology_control_running',active_job=OUT.name,active_pid=os.getpid());write_json(CYCLE/'state.json',state)
    def status(stage,**kw):write_json(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),**kw))
    status('loading');started=time.monotonic()
    source=CYCLE/'singleview_fusion_inputs_seed17_v1';source_manifest=json.loads((source/'manifest.json').read_text());baseline_path=OLD/'full_context_pose_control_v1/full_seed17_latest.pt';assert fingerprint(baseline_path)==source_manifest['pose_sha256']
    protocol=dict(steps=800,arms=['full','structured_union'],initial='Immutable full_seed17 pose baseline; same copied decoder and zero token projection in both arms. Baseline always frozen. No candidate selector.',input='Existing genuine V1 native FP32 memory1223TRAIN112VAL. No additional encoder pass, no GT box/count/identity in forward. Group native shapes without padding. All6x23 queries retained.',topology='Only new-branch self-attention topology changes: unrestricted vs same-instance OR same-semantic union mask. NOT sequential two-attention GroupPose reproduction. Extra decoder forward is counted.',sampling='Each16:4realbottom4otherreal6syntheticsingle2syntheticdual, random.Random17 with replacement, identical arms.42dualTRAIN. Fullsyntheticextra308candidate-only views not included.',loss='Coordinate-only matched_image_point_loss, pixel/768 SmoothL1beta.01; equal image weighting, only nose and4feet valid, fixed predicted-box Hungarian after frozen detector forward. Train all matched annotated instances independent of detection threshold; inference keepsall6queries. Back/abdomen/tail outputs unchanged.',optimizer='AdamW wd.001,clip1; decoder1e-5cosine->1e-6,residual1e-4cosine->1e-5. Final800 evaluation, no VAL checkpoint selection.',baseline_sha256=fingerprint(baseline_path),module_sha256=fingerprint(ROOT/'mousegeoforge/models/topology_pose_adapter.py'),source_sha256={r['role']:r['sha256'] for r in source_manifest['parts']})
    protocol['preflight_repair']='v1 stopped before initialization/training: synthetic-single bucket is132 synthetic+21 curve-kind synthetic frames, not153 literal synthetic-kind. v1 failure preserved; treat non-real single instances as synthetic, use only nose/fourfeet, no curve labels.'
    write_json(OUT/'protocol.json',protocol)
    cpu={};gpu={};memories={};inst={};mapping={};manifest_hashes={};counts={}
    for role in ['train','validation']:
        assert fingerprint(source/f'{role}.pt')==protocol['source_sha256'][role]
        d=torch.load(source/f'{role}.pt',map_location='cpu',weights_only=False);cpu[role]=d
        mp=OLD/f'single_view_{role}_context_v1/manifest.json';manifest_hashes[role]=fingerprint(mp);records={r['image_index']:r for r in json.loads(mp.read_text())['records']};memories[role]=[];instances=[]
        for i,gid in enumerate(d['image_indices']):
            assert dt.datetime.now(dt.timezone.utc)<deadline
            r=records[int(gid)];assert r['metadata']==d['metadata'][i] and r['metadata']['record']['role']==role and r['metadata']['record'].get('session') not in ['GR2','UR2','UR3']
            assert fingerprint(ROOT/r['path'])==r['sha256'];raw=torch.load(ROOT/r['path'],map_location='cpu',weights_only=False);assert raw['metadata']==d['metadata'][i]
            memories[role].append(raw['memory'].cuda());instances.append(raw['instance_tokens'])
            if (i+1)%200==0:status('loading',role=role,images=i+1,total=len(d['image_indices']));print('loaded',role,i+1,flush=True)
        inst[role]=torch.stack(instances).cuda()
        gpu[role]={k:v.cuda() for k,v in d.items() if isinstance(v,torch.Tensor)}
        matches=match_detection(dict(instance_boxes_cxcywh=d['boxes'],instance_objectness_logits=d['objectness_logits']),d['gt_boxes'],d['box_valid']&d['instance_valid']);mapping[role]=torch.full((len(d['xy']),2),-1,dtype=torch.long,device='cuda');c=collections.defaultdict(lambda:dict(images=0,gt=0,pred=0,tp=0))
        for i,(qs,ts) in enumerate(matches):
            mapping[role][i,ts.cuda()]=qs.cuda();m=d['metadata'][i];kind='real' if m['record']['kind']=='real' else 'synthetic_dual' if m['expected_instances']>1 else 'synthetic_single';a=c[kind];a['images']+=1;a['gt']+=int((d['box_valid'][i]&d['instance_valid'][i]).sum());a['pred']+=int((d['objectness_logits'][i].sigmoid()>=.5).sum())
            for q,t in zip(qs,ts):a['tp']+=int(d['objectness_logits'][i,q].sigmoid()>=.5 and box_iou(d['boxes'][i,q:q+1],d['gt_boxes'][i,t:t+1])[0,0]>=.5)
        counts[role]=dict(c)
    assert len(cpu['train']['xy'])==1223 and len(cpu['validation']['xy'])==112
    assert not ({m['record']['scene'] for m in cpu['train']['metadata']}&{m['record']['scene'] for m in cpu['validation']['metadata']})
    baseline=DecoderSpatialPoseReadout().cuda().eval();baseline.load_state_dict(torch.load(baseline_path,map_location='cpu',weights_only=False)['state'])
    def forward(model,role,ids,enabled):
        groups=collections.defaultdict(list);out=[None]*len(ids)
        for pos,i in enumerate(ids):groups[tuple(memories[role][i].shape)].append((pos,i))
        for members in groups.values():
            positions,images=zip(*members);images=list(images)
            p=model(inst[role][images],torch.stack([memories[role][i] for i in images]),gpu[role]['boxes'][images],enabled=enabled)['xy']
            for j,pos in enumerate(positions):out[pos]=p[j]
        return torch.stack(out)
    def loss_for(pred,role,ids):
        d=gpu[role];valid=d['box_valid'][ids]&d['instance_valid'][ids];local,targets=valid.nonzero().unbind(1);global_ids=torch.tensor(ids,device='cuda')[local];queries=mapping[role][global_ids,targets];assert (queries>=0).all()
        mask=d['point_valid'][global_ids,targets].clone();mask[:,KEEP]=False
        return matched_image_point_loss(pred[local,queries],d['gt_xy'][global_ids,targets],mask,d['pixels'][global_ids],local,len(ids))
    meta=cpu['train']['metadata'];groups=[[i for i,m in enumerate(meta) if m['record']['kind']=='real' and m['view']==0],[i for i,m in enumerate(meta) if m['record']['kind']=='real' and m['view']!=0],[i for i,m in enumerate(meta) if m['record']['kind']!='real' and m['expected_instances']==1],[i for i,m in enumerate(meta) if m['expected_instances']>1]];assert list(map(len,groups))[2:]==[153,42]
    rng=random.Random(17);schedule=[sum([rng.choices(g,k=k) for g,k in zip(groups,[4,4,6,2])],[]) for _ in range(800)];write_json(OUT/'schedule.json',schedule)
    torch.manual_seed(17);template=TopologyPoseAdapter(baseline,'full').cuda().eval();initial={k:v.detach().cpu().clone() for k,v in template.state_dict().items()};torch.save(initial,OUT/'initial.pt');parameters=sum(p.numel() for p in template.parameters() if p.requires_grad);del template
    gates={};arrays={};results=[];arm_compute={}
    for arm in ['full','structured_union']:
        model=TopologyPoseAdapter(baseline,arm).cuda().eval();model.load_state_dict(initial,strict=True);assert sum(p.numel() for p in model.parameters() if p.requires_grad)==parameters
        max_off=max_zero=0.
        with torch.no_grad():
            for role in ['train','validation']:
                for first in range(0,len(cpu[role]['xy']),8):
                    ids=list(range(first,min(first+8,len(cpu[role]['xy']))));off=forward(model,role,ids,False);zero=forward(model,role,ids,True);max_off=max(max_off,float((off-gpu[role]['xy'][ids]).abs().max()));max_zero=max(max_zero,float((zero-off).abs().max()))
        assert max_off<2e-5 and max_zero==0,(max_off,max_zero)
        gates[arm]=dict(initial_vs_source=max_off,initial_residual_difference=max_zero)
        status('smoke',arm=arm);ids=schedule[0];model.train();model.zero_grad(set_to_none=True);loss=loss_for(forward(model,'train',ids,True),'train',ids);assert torch.isfinite(loss);loss.backward();assert model.residual.weight.grad.abs().sum()>0 and all(p.grad is None for p in model.baseline.parameters());model.zero_grad(set_to_none=True)
        optimizer=torch.optim.AdamW([dict(params=model.decoder.parameters(),lr=1e-5),dict(params=model.residual.parameters(),lr=1e-4)],weight_decay=.001);history=[];torch.cuda.synchronize();train_start=time.monotonic();torch.cuda.reset_peak_memory_stats()
        for step,ids in enumerate(schedule,1):
            assert dt.datetime.now(dt.timezone.utc)<deadline
            factor=.1+.45*(1+math.cos(math.pi*(step-1)/800));optimizer.param_groups[0]['lr']=1e-5*factor;optimizer.param_groups[1]['lr']=1e-4*factor
            optimizer.zero_grad(set_to_none=True);pred=forward(model,'train',ids,True);loss=loss_for(pred,'train',ids);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);optimizer.step();history.append(dict(step=step,loss=float(loss.detach()),decoder_lr=optimizer.param_groups[0]['lr'],residual_lr=optimizer.param_groups[1]['lr']))
            if step%100==0:
                write_json(OUT/f'{arm}_history.json',history);checkpoint=dict(arm=arm,step=step,state={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict());torch.save(checkpoint,OUT/f'{arm}_step{step:04d}.pt');status('training',arm=arm,step=step,loss=history[-1]['loss']);print(arm,step,history[-1]['loss'],flush=True)
        torch.cuda.synchronize();arm_compute[arm]=dict(train_seconds=time.monotonic()-train_start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved());model.eval()
        with torch.no_grad():
            for role in ['train','validation']:
                xy=[];losses=[];off_delta=untargeted_delta=0.
                for first in range(0,len(cpu[role]['xy']),8):
                    ids=list(range(first,min(first+8,len(cpu[role]['xy']))));p=forward(model,role,ids,True);off=forward(model,role,ids,False);off_delta=max(off_delta,float((off-gpu[role]['xy'][ids]).abs().max()));untargeted_delta=max(untargeted_delta,float((p[:,:,KEEP]-off[:,:,KEEP]).abs().max()));xy.append(p.cpu());losses.append((len(ids),float(loss_for(p,role,ids))))
                assert off_delta<2e-5 and untargeted_delta==0,(off_delta,untargeted_delta)
                gates[arm][role+'_after_off_delta']=off_delta;gates[arm][role+'_untargeted_delta']=untargeted_delta
                xy=torch.cat(xy);d=cpu[role];evaldata=dict(base_xy=d['xy'],boxes=d['boxes'],logits=d['objectness_logits'],gt_xy=d['gt_xy'],point_valid=d['point_valid'],gt_boxes=d['gt_boxes'],box_valid=d['box_valid'],instance_valid=d['instance_valid'],pixels=d['pixels'],metadata=[dict(scene=m['record']['scene'],view=m['view'],kind='real' if m['record']['kind']=='real' else 'synthetic_dual' if m['expected_instances']>1 else 'synthetic_single',role=role,image_index=int(d['image_indices'][i])) for i,m in enumerate(d['metadata'])])
                if arm=='full':
                    torch.save(evaldata,OUT/f'{role}_eval_inputs.pt');zero=torch.zeros(len(xy),6,5,dtype=torch.bool);bmetrics,bpoints=metrics(evaldata,d['xy'],zero,zero.long());results.append(dict(arm='baseline',role=role,metrics=bmetrics));arrays[role+'_baseline']=d['xy'].numpy()
                active=(xy[:,:,KP]-d['xy'][:,:,KP]).abs().amax(-1)>2e-5;values,points=metrics(evaldata,xy,active,torch.zeros_like(active,dtype=torch.long))
                for point in points:point.pop('choice',None)
                write_json(OUT/f'{arm}_{role}_points.json',points);results.append(dict(arm=arm,role=role,metrics=values,coordinate_loss=sum(n*l for n,l in losses)/len(xy)));arrays[role+'_'+arm]=xy.numpy();print(arm,role,{k:round(v['mean_error'],3) for k,v in values.items() if k in ['real/reliable','real/view0/feet','synthetic_single/reliable','synthetic_dual/reliable']},flush=True)
        write_json(OUT/'partial_results.json',dict(results=results,gates=gates,compute=arm_compute))
        del optimizer,model
    np.savez_compressed(OUT/'predictions.npz',**arrays);write_json(OUT/'results.json',dict(results=results,gates=gates,compute=arm_compute,parameters_trainable=parameters,protocol=protocol,native_manifest_sha256=manifest_hashes,detector_counts_unchanged=counts,elapsed_seconds=time.monotonic()-started,script_sha256=fingerprint(Path(__file__)),production_changed=False));status('complete');state.update(stage='native_topology_control_complete',active_job=None,active_pid=None,next='Inspect native_topology_control_v1 results, convergence/golden/defaultoff/compute. Need raw all-query realbottom/dual overlays; no model promotion. Fixed800 botharms complete.');state['completed'].append('native_topology_control_800x2');write_json(CYCLE/'state.json',state)


if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write_json(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()))
        raise
