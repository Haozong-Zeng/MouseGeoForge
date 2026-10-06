"""Paired warm-start control:8 vs16 dual train poses, unchanged800 updates."""
import argparse
import copy
from collections import Counter
import fcntl
import hashlib
import json
import subprocess
import sys
import time
from torch.nn import functional as F
import train_cylinder_view_balance as base
import train_cylinder_adaptive_temperature as temperature
from run_depth_pose_ablation import ROOT,torch,write_json,fingerprint,tensor_state,state_hash,load_scene
from training.finetune_blendedmvs import median_camera_baseline
from mousegeoforge.adaptive_temperature import restore_adaptive

OUT=ROOT/'output/dual_pose_diversity_seed17_v1'
PARENT=ROOT/'output/cylinder_adaptive_temperature_v1'
CHECKPOINT=PARENT/'adaptive_temperature/final_heads.pt'
ARMS=('diversity_off','diversity_on')
EVAL_CALLS=0
QUALITY=[]
ORIGINAL_LOSS=base.engine.instance_pose_loss
LOSS_ROWS=[]
DEADLINE=1789194663  # 2026-09-12 06:31:03 UTC


def config():
    cfg=copy.deepcopy(json.loads((PARENT/'protocol.json').read_text())['config'])
    cfg.update(arms=list(ARMS),arm_learning_rates={a:3e-6 for a in ARMS},steps=800,
        checkpoint=str(CHECKPOINT.relative_to(ROOT)),max_run_hours=1.0,
        scope='New warm-start paired dual-pose diversity control; same already-used train source; synthetic reused development only',
        data_policy='OFF eight original dual poses; ON adds eight deterministically selected train-only poses; original600 non-dual slots untouched;200 dual slots and total800 updates unchanged',
        checkpoint_policy='Fixed final800 per arm, fresh matched optimizer; no best-development selection; no automatic continuation')
    return cfg


def prepare():
    if (OUT/'protocol.json').exists():raise RuntimeError('Already prepared; inspect, never rewrite')
    if not json.loads((ROOT/'output/training_instance_assignment_v1/report.json').read_text())['no_matching_only_training']:
        raise RuntimeError('Need completed assignment diagnosis first')
    spec=copy.deepcopy(json.loads((PARENT/'protocol.json').read_text()));cfg=config()
    for p,h in spec['code_sha256'].items():assert fingerprint(ROOT/p)==h
    for p,h in spec['source_sha256'].items():assert base.scene_digest(ROOT/p)==h
    manifest=json.loads((ROOT/cfg['manifest']).read_text())
    sequence=spec['arms']['adaptive_temperature'];off=[spec['records'][x[0]] for x in sequence]
    old_dual={r['frame']:r for r in off if r['expected_instances']==2}
    assert len(old_dual)==8
    dev=[spec['records'][e['key']] for e in spec['evaluation']]
    dev_scenes={r['scene'] for r in dev};dev_sources={r['source_group'] for r in dev}
    train_sources={r['source_group'] for r in old_dual.values()}
    pool={r['frame']:r for r in manifest['records'] if r['split']=='train' and r['expected_instances']==2
          and r['source_group'] in train_sources and r['scene'] not in dev_scenes and r['source_group'] not in dev_sources}
    assert len(pool)==200 and not(train_sources&dev_sources)
    # Greedy temporal-gap filling on existing diverse-pose ordinal; no prediction/GT quality ranking.
    selected=[];covered=set(old_dual)
    for _ in range(8):
        frame=min(set(pool)-covered,key=lambda f:(-min(abs(f-x) for x in covered),f))
        selected.append(frame);covered.add(frame)
    replacement={}
    for old_frame,new_frame in zip(sorted(old_dual),sorted(selected)):
        rec=copy.deepcopy(pool[new_frame]);before=old_dual[old_frame]
        loaded=load_scene(ROOT/rec['scene']);order=before['views']
        rigs={r['view_id']:r['orientation'] for r in base.cameras(ROOT/rec['scene'])}
        scale=float(median_camera_baseline(base.np.stack([loaded.frames[v].extrinsic for v in order])))
        revision=base.scene_digest(ROOT/rec['scene'])
        rec.update(views=order,diagnostic_view_order=order,normalization_scale=scale,
                   physical_roles=[rigs[v] for v in order],source_revision=revision)
        for field in ('image_hw','physical_roles','expected_instances','source_group','depth_units'):
            assert rec[field]==before[field],field
        batch=base.oriented_batch(rec,num_views=2)
        assert batch['detection_annotation_complete'].all() and batch['depth_available'].all()
        replacement[old_frame]=rec;spec['source_sha256'][rec['scene']]=revision
    counts=Counter();on=[]
    for rec in off:
        if rec['expected_instances']==2:
            counts[rec['frame']]+=1
            # Keep first occurrence and every odd occurrence, replace every even one.
            rec=replacement[rec['frame']] if counts[rec['frame']]%2==0 else rec
        on.append(rec)
    assert all(v==25 for v in counts.values())
    assert sum(a!=b for a,b in zip(off,on))==96
    assert len({r['scene'] for r in on})==48
    assert len({r['frame'] for r in on if r['expected_instances']==2})==16
    assert all(a==b for a,b in zip(off,on) if a['expected_instances']==1)
    spec['arms']={'diversity_off':sequence,'diversity_on':[]}
    for r in on:
        key=hashlib.sha256(json.dumps(r,sort_keys=True).encode()).hexdigest()
        spec['records'][key]=r;spec['arms']['diversity_on'].append([key])
    spec['config']=cfg;spec['checkpoint_sha256']=fingerprint(CHECKPOINT)
    spec['source_diversity_budget']=dict(old_dual=sorted(old_dual),added_dual=sorted(selected),
        actual_unique_poses_off=40,actual_unique_poses_on=48,dual_slots=200,nondual_slots=600,
        changed_slots=96,steps_per_arm=800,images_per_arm=1800,patch_tokens_per_arm=3540384,
        all_new_inputs_train_only=True,all_evaluation_unchanged=True)
    spec['code_sha256']['tools/train_dual_pose_diversity.py']=fingerprint(ROOT/'tools/train_dual_pose_diversity.py')
    spec['scope']=cfg['scope']
    spec['parent_protocol_sha256']=fingerprint(PARENT/'protocol.json')
    spec['assignment_audit_sha256']=fingerprint(ROOT/'output/training_instance_assignment_v1/report.json')
    spec['acceptance_policy']='Fixed final800. Same historical88/17 gates. Advance seed29 only if all17 ON gates pass, dual detection-aware PCK gain>=.01 versus same-budget OFF, dual box-matched NME not worse than1.02xOFF, and dual TP not lower. Report FP and duplicate-mask views separately. No picking frame296 success alone. If no meaningful gain, stop this mechanism; any extension must be predeclared and paired, maximum1600 additional steps per arm in this cycle, only with defensible held-source trend.'
    spec['freeze_policy']='Encoder frozen; same original shared adapter/depth/instance/temperature training scope and rates both arms, no new architecture/matcher/loss/mask-box postprocessing.'
    spec['production_promoted']=False
    write_json(OUT/'protocol.json',spec)
    write_json(OUT/'preparation.json',dict(passed=True,**spec['source_diversity_budget'],no_model_forward=True))
    write_json(OUT/'status.json',dict(state='prepared',total_steps=1600))
    print(json.dumps(spec['source_diversity_budget'],indent=2))


def build_protocol(cfg):
    spec=json.loads((OUT/'protocol.json').read_text());assert cfg==spec['config']==config()
    assert spec['parent_protocol_sha256']==fingerprint(PARENT/'protocol.json')
    assert fingerprint(CHECKPOINT)==spec['checkpoint_sha256']
    for p,h in spec['code_sha256'].items():assert fingerprint(ROOT/p)==h
    for p,h in spec['source_sha256'].items():assert base.scene_digest(ROOT/p)==h
    assert fingerprint(ROOT/'vggt-omega/vggt_omega_1b_512.pt')==spec['encoder_sha256']
    assert fingerprint(ROOT/cfg['manifest'])==spec['manifest_sha256']
    return spec


def checked_metrics(pred,batch,target):
    global EVAL_CALLS
    metric,match=base.physical_metrics(pred,batch,target)
    idx=EVAL_CALLS%88;tag=('initial',*ARMS)[EVAL_CALLS//88]
    spec=json.loads((OUT/'protocol.json').read_text());r=spec['records'][spec['evaluation'][idx]['key']]
    if EVAL_CALLS<88:
        prior=torch.load(PARENT/'adaptive_temperature'/f'prediction_{idx:03d}.pt',map_location='cpu',weights_only=True)
        for k,v in prior['predictions'].items():torch.testing.assert_close(pred[k].cpu(),v,atol=0,rtol=0)
        assert match==prior['oracle_assignment']
        if idx==87:write_json(OUT/'pretrain_parity.json',dict(passed=True,records=88,prediction_max_difference=0))
    for v in range(len(r['views'])):
        score=pred['instance_objectness_logits'][0,v].detach().cpu().sigmoid()
        masks=pred['instance_mask_logits'][0,v].detach().cpu()>0
        ids=torch.where(target['instance_valid'][0,v])[0]
        gt=F.interpolate(target['masks'][0,v,ids,None].float().cpu(),size=masks.shape[-2:],mode='nearest')[:,0].bool()
        miou=torch.tensor([[float((p&t).sum()/(p|t).sum().clamp_min(1)) for p in masks] for t in gt])
        active=torch.where(score>=.5)[0]
        best=miou[:,active].max(0) if len(active) else None
        identities=best.indices[best.values>=.5].tolist() if best is not None else []
        QUALITY.append(dict(tag=tag,index=idx,frame=r['frame'],group=spec['evaluation'][idx]['group'],view=v,
            role=r['physical_roles'][v],gt_instances=len(ids),active_queries=len(active),
            duplicate_strong_mask=len(identities)>len(set(identities)),scores=score.tolist(),native_mask_iou=miou.tolist()))
    if r['expected_instances']==2 and r['frame'] in (246,296,352,383):
        folder=OUT/tag;folder.mkdir(exist_ok=True)
        torch.save(dict(record=r,mask_logits=pred['instance_mask_logits'].detach().cpu(),
                        gt_masks=target['masks'].detach().cpu()),folder/f'mask_visual_{idx:03d}.pt')
    EVAL_CALLS+=1
    if EVAL_CALLS%88==0:write_json(OUT/f'{tag}_instance_quality.json',[q for q in QUALITY if q['tag']==tag])
    return metric,match


def checked_loss(*args,**kwargs):
    if time.time()>=DEADLINE:raise RuntimeError('User deadline')
    result=ORIGINAL_LOSS(*args,**kwargs)
    LOSS_ROWS.append({k:float(v.detach()) for k,v in result[1].items() if k in ('loss_instance_objectness','loss_instance_mask_bce','loss_instance_mask_dice','loss_instance_keypoint_l1','loss_instance_box_l1')})
    if len(LOSS_ROWS)%25==0:write_json(OUT/'loss_components.json',LOSS_ROWS)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--launch',action='store_true');args=parser.parse_args()
    if args.prepare_only:prepare();return
    if args.launch:
        assert json.loads((OUT/'status.json').read_text())['state']=='prepared'
        with (OUT/'run.log').open('a') as log:
            p=subprocess.Popen([sys.executable,'-u',__file__],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(pid=p.pid,output=str(OUT))));return
    if time.time()+3600>=DEADLINE:raise RuntimeError('Insufficient time before user deadline')
    with (ROOT/'output/cylinder_training.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        base.engine.build_protocol=build_protocol;base.engine.load_batch=base.oriented_batch
        base.engine.DepthPoseModel=temperature.TemperatureModel;base.engine.restore=restore_adaptive
        base.engine.torch.optim.AdamW=temperature.optimizer
        base.engine.evaluate_scene=checked_metrics;base.engine.instance_pose_loss=checked_loss
        base.engine.run(OUT,config(),False)
        spec=build_protocol(config())
        write_json(OUT/'verification.json',dict(passed=True,source_weights_code_unchanged=True,initial_prediction_max_difference=0,
            optimizer_steps_per_arm=800,evaluations=EVAL_CALLS,encoder_gradient_free=True,
            protocol_sha256=fingerprint(OUT/'protocol.json'),production_promoted=False))


if __name__=='__main__':main()
