"""TRAIN-only capacity sanity check. Not validation or deployment evidence."""
import collections,datetime as dt,fcntl,json,math,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from mousegeoforge.models.decoder_spatial_pose_readout import DecoderSpatialPoseReadout
from mousegeoforge.detection_readout_loss import match_detection
from mousegeoforge.matched_point_loss import matched_image_point_loss
from run_depth_pose_ablation import fingerprint,write_json
CYCLE=ROOT/'output/core23_spatial_cycle_20261004';OUT=CYCLE/'bottom_train_fit_v1'

def main():
    plan=json.loads((CYCLE/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'));lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not OUT.exists();OUT.mkdir();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    def status(stage,**kw):write_json(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),**kw))
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='bottom_train_fit_sanity',active_job=OUT.name,active_pid=os.getpid());write_json(CYCLE/'state.json',state);status('loading')
    rows=[]
    for r in json.loads((CYCLE/'detection_features_v1/manifest.json').read_text())['records']:
        assert fingerprint(ROOT/r['path'])==r['sha256'];rows.extend(torch.load(ROOT/r['path'],map_location='cpu',weights_only=False)['observations'])
    sessions=collections.defaultdict(list)
    for i,o in enumerate(rows):
        m=o['metadata'];r=m['record']
        if r['kind']=='real' and r['role']=='train' and m['view']==0 and m['detection_complete'] and o['point_valid'][0,[0,14,15,16,17]].all():
            assert r['session'] not in ['GR2','UR2','UR3'];sessions[r['session']].append(i)
    ids=[]
    for session,members in sorted(sessions.items()):
        members.sort(key=lambda i:rows[i]['metadata']['record']['frame']);assert len(members)>=2;ids.extend([members[0],members[-1]])
    assert len(ids)>=10 and len(set(ids))==len(ids)
    selected=[rows[i] for i in ids];native_manifest=json.loads((CYCLE/'native_decoder_memory_v1/manifest.json').read_text());native_by_key={r['key']:r for r in native_manifest['records']};memory=[]
    for o in selected:
        m=o['metadata'];rec=native_by_key[m['record']['key']];assert fingerprint(ROOT/rec['path'])==rec['sha256'];part=torch.load(ROOT/rec['path'],map_location='cpu',weights_only=False);memory.append(part['memory'][m['view_index']])
    assert len({tuple(x.shape) for x in memory})==1;memory=torch.stack(memory).cuda();inst=torch.stack([o['instance_tokens'] for o in selected]).cuda();gt=torch.stack([o['gt_xy'][0] for o in selected]).cuda();valid=torch.stack([o['point_valid'][0] for o in selected]).cuda();pixels=torch.stack([o['pixels'] for o in selected]).cuda()
    allowed=torch.zeros_like(valid);allowed[:,[0,14,15,16,17]]=True;valid&=allowed
    with np.load(CYCLE/'detection_context_roi_v1/predictions.npz') as a:p={k:torch.tensor(a['seed17__'+k][ids],device='cuda') for k in ['instance_boxes_cxcywh','instance_objectness_logits']}
    boxes=torch.stack([o['boxes'] for o in selected]).cuda();boxvalid=torch.stack([o['box_valid']&o['instance_valid'] for o in selected]).cuda();matches=match_detection(p,boxes,boxvalid);queries=torch.tensor([int(qs[0]) for qs,ms in matches],device='cuda');im=torch.arange(len(ids),device='cuda')
    decoder_path=CYCLE/'native_decoder_memory_v1/initial_decoder.pt';assert fingerprint(decoder_path)==native_manifest['decoder_initial_sha256'];decoder=torch.load(decoder_path,map_location='cuda',weights_only=True);readout_path=CYCLE/'readout_post4800_control_v1/spatial_seed17_latest.pt';readout=torch.load(readout_path,map_location='cuda',weights_only=False)['state']
    write_json(OUT/'protocol.json',dict(purpose='TRAIN-only capacity sanity; no validation metrics or model selection. Not deployment evidence.',selection='First and last eligible TRAINcamA frame per existing TRAINsession, allnose+4feet observed, complete acceptedmask. Deterministic, no error-based frame selection.',observations=ids,metadata=[o['metadata'] for o in selected],arms=['frozen','unfrozen'],steps=1600,seed=17,batch='All selected frames each step; no augmentation or synthetic in this diagnostic.',optimizer='Fresh AdamWwd0; readoutcosine3e-4to3e-5; decoder3e-5to3e-6 when unfrozen;clip1.',criterion='Report trainmean/feetmean/p90/PCK16 and per-image; mean<10px and PCK16>=.9 indicates fitting capacity only. Fixed1600steps, no validation-directed search.',initial_decoder_sha256=fingerprint(decoder_path),initial_readout_sha256=fingerprint(readout_path),script_sha256=fingerprint(Path(__file__)),production_changed=False))
    results=[];saved={};initial_prediction=None
    for arm in ['frozen','unfrozen']:
        torch.manual_seed(17);model=DecoderSpatialPoseReadout().cuda().eval();loaded=model.load_state_dict(decoder,strict=False);assert not loaded.unexpected_keys and all(k.startswith('spatial_readout.') for k in loaded.missing_keys);model.spatial_readout.load_state_dict(readout);model.set_decoder_trainable(arm=='unfrozen')
        groups=[dict(params=model.spatial_readout.parameters(),lr=3e-4,peak=3e-4,floor=3e-5)]
        if arm=='unfrozen':groups.append(dict(params=model.keypoint_decoder.parameters(),lr=3e-5,peak=3e-5,floor=3e-6))
        opt=torch.optim.AdamW(groups,weight_decay=0.);history=[]
        @torch.no_grad()
        def evaluate():
            allxy=model(inst,memory,p['instance_boxes_cxcywh'])['xy'];xy=allxy[im,queries];err=((xy-gt)*pixels[:,None]).norm(dim=-1);e=err[valid];feet=valid.clone();feet[:,0]=False
            return dict(mean=float(e.mean()),median=float(e.median()),p90=float(torch.quantile(e,.9)),feet_mean=float(err[feet].mean()),pck16=float((e<=16).float().mean()),per_image=[dict(observation=i,mean=float(err[j,valid[j]].mean()),feet_mean=float(err[j,feet[j]].mean())) for j,i in enumerate(ids)]),allxy
        metrics,xy=evaluate()
        if initial_prediction is None:initial_prediction=xy.detach().clone()
        else:torch.testing.assert_close(xy,initial_prediction,atol=2e-5,rtol=0)
        history.append(dict(step=0,metrics=metrics))
        for step in range(1,1601):
            assert dt.datetime.now(dt.timezone.utc)<deadline
            f=.5*(1+math.cos(math.pi*(step-1)/1600))
            for g in opt.param_groups:g['lr']=g['floor']+(g['peak']-g['floor'])*f
            opt.zero_grad(set_to_none=True);xy=model(inst,memory,p['instance_boxes_cxcywh'])['xy'][im,queries];loss=matched_image_point_loss(xy,gt,valid,pixels,im,len(ids));assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm);opt.step()
            if step%200==0:
                metrics,pred=evaluate();history.append(dict(step=step,loss=float(loss.detach()),metrics=metrics));write_json(OUT/f'{arm}_history.json',history);torch.save(dict(state=model.state_dict(),optimizer=opt.state_dict(),step=step,arm=arm,diagnostic_train_only=True),OUT/f'{arm}_latest.pt');status('training',arm=arm,step=step);print('TRAIN_FIT',arm,step,metrics['mean'],metrics['feet_mean'],metrics['pck16'],flush=True)
        results.append(dict(arm=arm,**history[-1]));saved[arm]=pred.cpu().numpy();write_json(OUT/'partial_results.json',results);np.savez_compressed(OUT/'predictions.npz',image_indices=np.array(ids),initial=initial_prediction.cpu().numpy(),**saved)
    write_json(OUT/'results.json',dict(results=results,train_only=True,no_validation_evaluated=True,no_production_promotion=True));status('complete');state.update(stage='bottom_train_fit_ready',active_job=None,active_pid=None,updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),next='Review bottom_train_fit_v1 TRAIN-only curves and all selected frame predictions. This is not generalization. Determine whether tiny subset fits before adding steps to broad decoder experiments; inspect labels and predictedpositions, keep source labels untouched.');state['completed'].append('bottom_train_only_capacity_probe');write_json(CYCLE/'state.json',state)

if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(exist_ok=True);write_json(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()));raise
