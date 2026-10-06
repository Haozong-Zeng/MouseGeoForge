"""Bounded image-local center warmup after global-query localization failed.

One shared prerequisite detector, not a claimed matched comparison with the
previous 640-step two-arm pose experiment. Data draws/rotations are its first256.
"""
import os
os.environ['DLClight']='True'
import sys,json,datetime as dt,fcntl,time,traceback,shutil,argparse,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import torch,numpy as np
from scipy.optimize import linear_sum_assignment
from PIL import Image,ImageDraw,ImageFont
from preflight_mixed_local_backbone_20261005 import build_backbone,load_rgb,sha,digest
from train_mixed_local_backbone_20261005 import buffers_digest,atomic_save
from mousegeoforge.models.mixed_local_evidence import MixedLocalEvidence
from mousegeoforge.models.local_point_evidence_probe import point_evidence_loss
from mousegeoforge.models.local_instance_grouping import rotate_mask_centroids
from mousegeoforge.models.quarter_turn_evidence import quarter_turn
from mousegeoforge.models.dense_instance_centers import (
    DenseInstanceCenters,dense_center_loss,decode_centers,center_heatmap_targets,dense_center_gaussian_loss,
)
C=ROOT/'output/local_grouping_cycle_20261005';OLD=ROOT/'output/singleview_fusion_cycle_20261005'
INPUT=OLD/'mixed_local_fulltrain_rgb_inputs_v1';SOURCE=OLD/'mixed_local_backbone800_v1/train_official_backbone_step0800.pt'
OUT=C/'dense_center_warmup256_v1'

def now():return dt.datetime.now(dt.timezone.utc)
def read(p):return torch.load(p,weights_only=False,map_location='cpu')
def write(p,x):
    tmp=p.with_suffix('.json.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def score(decoded,supervision,hw):
    scale=torch.tensor([hw[1],hw[0]]);diag=float(scale.float().norm())
    gt=supervision['mask_centroid'][supervision['instance_valid']]
    centers=decoded['centers'];active=decoded['instance_active'];available=decoded['available']
    active_error=[diag]*len(gt);candidate_error=[diag]*len(gt);tp=0
    for which,keep in [('active',active),('candidate',available)]:
        if not keep.any() or not len(gt):continue
        distance=torch.cdist(centers[keep]*scale,gt*scale)
        rr,cc=linear_sum_assignment(distance.numpy())
        for a,b in zip(rr,cc):
            error=float(distance[a,b])
            if which=='active':active_error[b]=error;tp+=int(error<=64.)
            else:candidate_error[b]=error
    return dict(gt_count=len(gt),predicted_count=int(active.sum()),count_correct=int(int(active.sum())==len(gt)),
                tp64=tp,fp64=int(active.sum())-tp,fn64=len(gt)-tp,error=active_error,
                candidate_oracle=candidate_error,probabilities=decoded['object_probability'].tolist())

def summarize(details):
    answer={}
    for stage in ['fit','calibration']:
        answer[stage]={}
        for kind in ['real','single','dual','real_bottom']:
            dd=[d for d in details if d['metadata']['stage']==stage and
                (d['metadata']['kind']==kind or (kind=='real_bottom' and d['metadata']['kind']=='real' and d['metadata']['physical_bottom']))]
            if not dd:continue
            answer[stage][kind]=dict(images=len(dd),center_loss=float(np.mean([d['center_loss'] for d in dd])),
                local_loss=float(np.mean([d['local_loss'] for d in dd])),count_accuracy=float(np.mean([d['metrics']['count_correct'] for d in dd])),
                tp64=sum(d['metrics']['tp64'] for d in dd),fp64=sum(d['metrics']['fp64'] for d in dd),fn64=sum(d['metrics']['fn64'] for d in dd),
                center_error_px=float(np.mean([x for d in dd for x in d['metrics']['error']])),
                candidate_oracle_px=float(np.mean([x for d in dd for x in d['metrics']['candidate_oracle']])))
    return answer

def main():
    global OUT
    parser=argparse.ArgumentParser();modes=parser.add_mutually_exclusive_group();modes.add_argument('--continue-to1024',action='store_true');modes.add_argument('--overfit-from1024',action='store_true');modes.add_argument('--continue-to4096',action='store_true');modes.add_argument('--budget-arm',choices=['low','high']);modes.add_argument('--loss-arm',choices=['focal','gaussian']);modes.add_argument('--continue-loss-arm',choices=['focal','gaussian']);args=parser.parse_args()
    loss_arm=args.loss_arm or args.continue_loss_arm
    initial_step=256 if args.continue_to1024 else 0;final_step=1024 if args.continue_to1024 else 256
    evaluation_steps=[256,512,768,1024] if args.continue_to1024 else [0,64,128,256]
    prior=C/'dense_center_warmup256_v1'
    if args.continue_to1024:
        assert json.loads((prior/'status.json').read_text())['stage']=='complete'
        OUT=C/'dense_center_continue1024_v1'
    if args.overfit_from1024:
        prior=C/'dense_center_continue1024_v1'
        assert json.loads((prior/'status.json').read_text())['stage']=='complete'
        OUT=C/'dense_center_fit3_v1';initial_step=1024;final_step=1536
        evaluation_steps=[1024,1088,1152,1280,1536]
    if args.continue_to4096:
        prior=C/'dense_center_continue1024_v1'
        assert json.loads((prior/'status.json').read_text())['stage']=='complete'
        fit_result=json.loads((C/'dense_center_fit3_v1/results.json').read_text())
        assert all(v['count_accuracy']==1 and v['center_error_px']<=16 for v in fit_result['summary']['fit'].values())
        fit_details=json.loads((C/'dense_center_fit3_v1/step1536_details.json').read_text())
        assert len(fit_details)==3 and all(d['metrics']['count_correct'] and max(d['metrics']['error'])<=16 for d in fit_details)
        OUT=C/'dense_center_continue4096_v1';initial_step=1024;final_step=4096
        evaluation_steps=[1024,1536,2048,3072,4096]
    if args.budget_arm:
        prior=C/'dense_center_continue4096_v1'
        assert json.loads((prior/'status.json').read_text())['stage']=='complete'
        OUT=C/f'dense_center_lr_{args.budget_arm}_6144_v1';initial_step=4096;final_step=6144
        evaluation_steps=[4096,4608,5120,6144]
    if args.loss_arm:
        prior=C/'dense_center_lr_high_6144_v1'
        assert json.loads((prior/'status.json').read_text())['stage']=='complete'
        assert json.loads((C/'dense_center_lr_point_audit_v1/status.json').read_text())['stage']=='complete'
        OUT=C/f'dense_center_loss_{args.loss_arm}_7168_v1';initial_step=6144;final_step=7168
        evaluation_steps=[6144,6400,6656,7168]
    if args.continue_loss_arm:
        arm=args.continue_loss_arm
        prior=C/f'dense_center_loss_{arm}_7168_v1'
        assert json.loads((prior/'status.json').read_text())['stage']=='complete'
        audit=json.loads((C/'dense_center_loss_point_audit_v1/results.json').read_text())
        assert audit['paired_protocol_exact'] and audit['center_replay_exact'] and audit['weights_bn_input_unchanged']
        assert sha(prior/'center_step7168.pt')==audit['source_hashes'][arm]
        OUT=C/f'dense_center_loss_{arm}_9216_v1';initial_step=7168;final_step=9216
        evaluation_steps=[7168,7680,8192,9216]
    plan=json.loads((C/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'))
    assert now()<deadline
    assert json.loads((C/'grouping_pilot_v1/status.json').read_text())['stage']=='complete'
    lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    OUT.mkdir(exist_ok=False);torch.set_num_threads(4);torch.manual_seed(17);torch.backends.cuda.matmul.allow_tf32=False
    state=json.loads((C/'state.json').read_text());state.update(stage='dense_center_warmup_running',active_job=str(OUT.relative_to(ROOT)),active_pid=os.getpid(),updated_utc=now().isoformat(),next=f'Wait for bounded dense-center{final_step} prerequisite; no second GPUjob.')
    write(C/'state.json',state)
    def status(stage,**kw):write(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=now().isoformat(),**kw))
    status('loading')
    rows={r['index']:r for r in read(INPUT/'evaluation_only.pt')}
    manifest={r['index']:r for r in json.loads((INPUT/'inference_manifest.json').read_text())}
    sup_path=C/'instance_supervision_audit_v1/evaluation_instance_targets.pt';sup={r['index']:r for r in read(sup_path)}
    old_protocol=json.loads((C/'grouping_pilot_v1/protocol.json').read_text())
    schedule=old_protocol['schedule'][:final_step];rotations=old_protocol['rotations'][:final_step]
    if final_step>len(schedule):
        rng=random.Random(31);rot=random.Random(47)
        real={b:[i for i,r in rows.items() if r['metadata']['stage']=='fit' and r['metadata']['kind']=='real' and r['metadata']['physical_bottom']==b] for b in [True,False]}
        synthetic={k:[i for i,r in rows.items() if r['metadata']['stage']=='fit' and r['metadata']['kind']==k] for k in ['single','dual']}
        for j in range(len(schedule),final_step):
            schedule.append([rng.choice(real[j%2==0]),rng.choice(synthetic['single']),rng.choice(synthetic['dual'])]);rotations.append([rot.randrange(4) for _ in range(3)])
    fixed=[next(i for i,r in rows.items() if r['metadata']['stage']=='fit' and r['metadata']['kind']==k) for k in ['real','single','dual']]
    evaluation=fixed+[i for i,r in rows.items() if r['metadata']['stage']=='calibration']
    selected=list(dict.fromkeys(fixed+[next(i for i,r in rows.items() if r['metadata']['stage']=='calibration' and r['metadata']['kind']==k) for k in ['real','single','dual']]+[i for i,r in rows.items() if r['metadata']['stage']=='calibration' and r['metadata']['kind']=='real' and r['metadata']['physical_bottom']]))
    if args.overfit_from1024:
        evaluation=fixed;selected=fixed
        schedule=[list(fixed) for _ in range(final_step)]
        rotations=[[0,0,0] for _ in range(final_step)]
    if args.continue_to4096 or args.budget_arm or loss_arm:
        previous=json.loads((prior/'protocol.json').read_text())
        assert schedule[:initial_step]==previous['schedule'] and rotations[:initial_step]==previous['rotations']
    needed=set(evaluation)|{i for ids in schedule for i in ids}
    for i in needed:
        assert rows[i]['metadata']==sup[i]['metadata']
        assert rows[i]['metadata']['stage'] in ['fit','calibration']
        assert rows[i]['metadata'].get('source') not in ['FR2','AZR2L2','GR2','UR2','UR3']
        assert torch.equal(sup[i]['instance_valid'],sup[i]['centroid_valid'])
        assert bool(sup[i]['detection_annotation_complete'])
    source_hash=sha(SOURCE);sup_hash=sha(sup_path)
    assert source_hash=='601505a17ddd90ea374c1f6c2f6cb98d8692a6a360e268c7eae20134d096d8b7'
    backbone,_=build_backbone();model=MixedLocalEvidence(backbone,adapt_backbone=True).cuda();cp=read(SOURCE)
    model.backbone.load_state_dict(cp['backbone'],strict=True);model.readout.load_state_dict(cp['readout'],strict=True);del cp;model.train()
    detector=DenseInstanceCenters().cuda();bn=buffers_digest(model.backbone)
    assert digest(model.backbone)=='18594fc7cd3a5587ce595ceac8c00c34f99c5572cffa6bcc5fec2a05ca753692'
    opt=torch.optim.AdamW([dict(params=detector.parameters(),lr=3e-4),dict(params=model.readout.parameters(),lr=1e-4),dict(params=model.backbone.parameters(),lr=1e-6)],weight_decay=1e-4)
    resume_hash=None
    if initial_step:
        resume_path=prior/f'center_step{initial_step:04d}.pt'
        resume_hash=sha(resume_path);cp=read(resume_path)
        assert cp['step']==initial_step and cp['source_sha256']==source_hash
        model.backbone.load_state_dict(cp['backbone'],strict=True);model.readout.load_state_dict(cp['readout'],strict=True)
        detector.load_state_dict(cp['center'],strict=True);opt.load_state_dict(cp['optimizer']);del cp
        assert buffers_digest(model.backbone)==bn
    if args.budget_arm:
        assert [g['lr'] for g in opt.param_groups]==[3e-4,1e-4,1e-6]
        opt.param_groups[2]['lr']=1e-5 if args.budget_arm=='high' else 1e-6
    if loss_arm:assert [g['lr'] for g in opt.param_groups]==[3e-4,1e-4,1e-5]
    protocol=dict(steps=final_step,initial_step=initial_step,resume_sha256=resume_hash,evaluation_steps=evaluation_steps,seed=17,schedule=schedule,rotations=rotations,
        source_checkpoint=str(SOURCE.relative_to(ROOT)),source_sha256=source_hash,supervision_sha256=sup_hash,
        source_rule='Fresh commonmixed800 atstep0, continued256withpreservedmodel+optimizer ifrequested; notfailedquery640 model and not3-imageoverfit. Beyond640newdrawseed31/rotation47.',
        architecture='Dense stride4 convolutions on normalized HRNet32 image features; projected visible-mask centroid Gaussian targets, independent sigmoid center peaks. No GT input/count.',
        loss='local heatmap CE + modified focalcenter loss; sigma16px, snappedpositivecell, positiveexponent2/negativeGaussianweight4. Complete instance annotation required fornegativepixels; unknownpointsstillignored.',
        decode='3x3 true regional maxima, capacity2, same8px NMS, fixedsigmoid0.5, missingcentersunavailable. Counts notGT.',
        limitation='Single prerequisite detector, notmatched640 poseperformance. Two subsequent grouping arms will sharethisinitialization ifcenterstagepasses. Center is projectedmasknotanatomy. Existingselected1055poolnotunbiasedbenchmark.',
        lrs=[3e-4,1e-4,1e-6],fixed_bn=True,detector_parameters=sum(p.numel() for p in detector.parameters()),
        deadline=plan['training_stop_utc'],code_sha256=sha(Path(__file__)),module_sha256=sha(ROOT/'mousegeoforge/models/dense_instance_centers.py'),production_changed=False)
    if args.overfit_from1024:
        protocol.update(mode='Three-image FIT-only diagnostic; not generalization or future initialization.',
                        source_rule='Resume mixed1024 weights+optimizer exactly; repeat FIT675/186/1240 for512updates withoutaugmentation. No CAL/DEV/locked images evaluated.',
                        pass_gate='All three counts correct and all four maskcenters within16px at fixed0.5; training sanity only.')
    if args.continue_to4096:
        protocol.update(mode='Bounded training-budget continuation, not new architecture.',
                        source_rule='Exact mixed1024 model+optimizer (NOTfit3); original first1024draws/turns verified; continue same fixed random streams and equal-domain/alternating-real-bottom schedule. All LR/loss/thresholds unchanged.',
                        reason='Three-image diagnostic fits real center in64updates; 1024mixedupdates covered476/821realimages. FITreal contains108bottom/713other; bottom averages4.74draws, other0.72, versus11.91single/17.36dual. Underexposure plausible, not proven.',
                        decision='At4096 inspect same89CAL and fixedFIT, count/FPFN/centerlocalization/positiveprob plusallbottomvisuals; no automaticpromotion or furtherextension.')
    if args.budget_arm:
        protocol.update(mode='Matched backbone-LR diagnostic, NOT independent-vs-grouping poseablation.',
                        arm=args.budget_arm,lrs=[g['lr'] for g in opt.param_groups],
                        source_rule='Both from same mixed4096 model+AdamWstate, same2048furtherupdates/schedule/turns. Only backboneLR differs1e-6vs1e-5; center3e-4/local1e-4 unchanged. Exact4096outputreplay required.',
                        reason='Mixed4096endpoint reviewed beforelaunch. Three-imagefits quickly, mixedrealcenterprob remainslow evenonfixedFIT. Test budget-only vs fasterfeatureadaptation withoutchangingarchitecture/loss/threshold.',
                        decision='Comparefinal6144 onlyafterbothcomplete, includingrealbottomFPFN/centererror/positiveprob, otherdomainsandlocalcandidatecoverage. NoCALthresholdselection orGTforward, no automaticpromotion.')
    if args.loss_arm:
        protocol.update(mode='Matched center-objective diagnostic; noarchitecturechange.',arm=args.loss_arm,lrs=[g['lr'] for g in opt.param_groups],
                        source_rule='BothfromSAMEhighLR6144 model+optimizerstate, notloworfit3. First6144schedule+turns verified; same1024additionalupdates andfixedimage sequence, allLRunchanged.',
                        loss=('Original singlepositive modifiedfocal' if args.loss_arm=='focal' else 'Gaussian-field squared error on sigmoid, sum validcells / clamp_min(sum target_squared,1). Same snappedcenters/sigma16/maxunion; onlycompleteannotation. Energy-normalization ispartofthetestedobjective, NOTexactMMPoseimplementation.'),
                        reason='6144bothLRarmsstillrealTP0; investigate sparsepositive supervision vs fittingwholeGaussianfield. HigherHRNet/MMPose HeatmapLoss usesmaskedMSE; ownsigmoid+energy-normalizedvariant.',
                        decision='Do NOT comparerawlossacrossobjectives. Compare fixed7168count/FPFN/centererror/top1/pointcandidatecoverage andallbottomvisuals afterbothcomplete. Threshold.5 remains heatmap-scoregate, notcalibratedprobability. Noautomaticpromotion.')
    if args.continue_loss_arm:
        protocol.update(mode='Matched objective-pair budget continuation; no new mechanism.',arm=loss_arm,lrs=[g['lr'] for g in opt.param_groups],
                        common_6144_sha256=previous['resume_sha256'],
                        source_rule='Resume each corresponding7168 arm model+AdamW; lineage shares exact6144 initialization. First7168draws/turns verified; same2048additionalupdates to9216, allLR/augmentation/BN/decoder unchanged. Require exact7168output replay.',
                        loss=previous['loss'],
                        reason='Gaussian first1024updates improves realTP0to17/32 and bottom0to7/16; localbottomTop1PCK16 .730to.825 but CALsingle44FP inclapparatus. Gaussian6400/6656/7168 realTP7/5/17, bottom1/1/7: stillchanging, no convergence claim. Test boundedbudget beforeanygrouping ornewhead.',
                        decision='Compare paired9216 endpoints and trajectories; stopbudget-only expansion here unless new concrete evidence warrants a separate protocol. Report all domains/FPFN/localpoints. NMS32 audit reducedduplicates butlost3dualTP, so default8 andgate.5 retained. NoCALthresholdsearch or automaticpromotion.')
    write(OUT/'protocol.json',protocol)
    images={}
    for n,i in enumerate(sorted(needed)):
        if now()>=deadline:raise TimeoutError('Input cutoff')
        images[i]=load_rgb(manifest[i]).cuda()
        if (n+1)%100==0:status('loading',loaded=n+1,total=len(needed))
    curves=[];losses=[];gradients=[];seconds=0.;peak=0;collision_count=0
    def compute(i,k=0):
        r=rows[i];s=sup[i];hw=manifest[i]['image_hw'];x=images[i]
        xy=r['target'].cuda();eligible=r['eligible'].cuda();centers=s['mask_centroid'].cuda();valid=s['centroid_valid'].cuda()
        if k:
            x,xy,hw=quarter_turn(x,xy,eligible,hw,k);centers=rotate_mask_centroids(centers,valid,k)
        evidence=model(x,hw);out=detector(evidence,hw)
        collisions=center_heatmap_targets(out,centers,valid,hw)['collisions']
        objective=dense_center_gaussian_loss if loss_arm=='gaussian' else dense_center_loss
        loss_center=objective(out,centers,valid,hw,annotation_complete=True)
        loss_local=point_evidence_loss(evidence,xy,eligible,hw)
        return out,loss_center,loss_local,collisions
    @torch.no_grad()
    def evaluate(step):
        details=[];pred={}
        for i in evaluation:
            if now()>=deadline:raise TimeoutError('Evalcutoff')
            out,lc,lp,collisions=compute(i);decoded={k:v.cpu() for k,v in decode_centers(out,manifest[i]['image_hw']).items()}
            target=center_heatmap_targets(out,sup[i]['mask_centroid'].cuda(),sup[i]['centroid_valid'].cuda(),manifest[i]['image_hw'])
            logits=out['logits'][0];positive=target['positive'];negative=(~positive)&out['spatial_valid'];prob=logits.sigmoid()
            positive_loss=(-(1-prob).square()*torch.nn.functional.logsigmoid(logits))[positive].sum()/positive.sum().clamp_min(1)
            negative_loss=(-prob.square()*torch.nn.functional.logsigmoid(-logits)*(1-target['heatmap']).pow(4))[negative].sum()/positive.sum().clamp_min(1)
            details.append(dict(index=i,metadata=rows[i]['metadata'],center_loss=float(lc),local_loss=float(lp),collisions=collisions,
                                positive_loss=float(positive_loss),negative_loss=float(negative_loss),positive_probabilities=prob[positive].cpu().tolist(),
                                metrics=score(decoded,sup[i],manifest[i]['image_hw'])))
            pred[i]=dict(decoded=decoded)
            if i in selected:pred[i]['heatmap']=out['logits'].sigmoid().cpu()
        summary=summarize(details);curves.append(dict(step=step,summary=summary));write(OUT/'curves.json',curves)
        write(OUT/f'step{step:04d}_details.json',details);atomic_save(OUT/f'step{step:04d}_predictions.pt',pred)
        return pred,summary
    def checkpoint(step):atomic_save(OUT/f'center_step{step:04d}.pt',dict(step=step,
        backbone={k:v.cpu() for k,v in model.backbone.state_dict().items()},readout={k:v.cpu() for k,v in model.readout.state_dict().items()},
        center={k:v.cpu() for k,v in detector.state_dict().items()},optimizer=opt.state_dict(),source_sha256=source_hash))
    start=time.monotonic();status('evaluating',step=initial_step);pred,summary=evaluate(initial_step)
    if initial_step:
        old_pred=read(prior/f'step{initial_step:04d}_predictions.pt')
        for i in evaluation:
            for key,value in pred[i]['decoded'].items():assert torch.equal(value,old_pred[i]['decoded'][key]),(i,key,'resumeparity')
            if 'heatmap' in pred[i]:assert torch.equal(pred[i]['heatmap'],old_pred[i]['heatmap'])
    for step,ids in enumerate(schedule[initial_step:],initial_step+1):
        if now()>=deadline:checkpoint(step-1);raise TimeoutError('Trainingcutoff')
        opt.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.monotonic();value=[0.,0.]
        for slot,i in enumerate(ids):
            out,lc,lp,collisions=compute(i,rotations[step-1][slot]);loss=(lc+lp)/3
            assert torch.isfinite(loss);loss.backward();value[0]+=float(lc.detach())/3;value[1]+=float(lp.detach())/3;collision_count+=collisions
            del out,lc,lp,loss
        if step in [initial_step+1,initial_step+2,initial_step+32,final_step]:
            norms={name:sum(float(p.grad.square().sum()) for p in m.parameters() if p.grad is not None)**.5 for name,m in [('center',detector),('backbone',model.backbone),('local',model.readout)]}
            assert all(np.isfinite(v) and v>0 for v in norms.values());gradients.append(dict(step=step,norms=norms))
        torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(detector.parameters()),1,error_if_nonfinite=True);opt.step();torch.cuda.synchronize();seconds+=time.monotonic()-t;peak=max(peak,torch.cuda.max_memory_allocated());losses.append(value)
        if step%8==0:status('training',step=step,total=final_step,training_seconds=seconds,recent_loss=np.mean(losses[-8:],axis=0).tolist());print('STEP',step,np.mean(losses[-8:],axis=0).tolist(),flush=True)
        if step in evaluation_steps:
            assert buffers_digest(model.backbone)==bn;checkpoint(step);status('evaluating',step=step);pred,summary=evaluate(step)
    assert sha(SOURCE)==source_hash and sha(sup_path)==sup_hash and buffers_digest(model.backbone)==bn
    assert all(sha(ROOT/manifest[i]['image'])==manifest[i]['sha256'] for i in needed)
    if initial_step:assert sha(resume_path)==resume_hash
    result=dict(summary=summary,steps=final_step,initial_step=initial_step,resume_exact_parity=bool(initial_step),gradients=gradients,training_collisions=collision_count,train_seconds=seconds,
                total_seconds=time.monotonic()-start,peak_gpu_bytes=peak,losses=losses,bn_unchanged=True,sources_unchanged=True,production_changed=False)
    write(OUT/'results.json',result)
    examples=OUT/'examples';examples.mkdir();font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    for i in selected:
        rgb=Image.open(ROOT/manifest[i]['image']).convert('RGB');w,h=rgb.size
        heat=pred[i]['heatmap'];heat=torch.nn.functional.interpolate(heat[None],scale_factor=4,mode='bilinear',align_corners=False)[0,0,:h,:w].numpy()
        arr=np.asarray(rgb).astype(float);color=np.zeros_like(arr);color[...,0]=255;color[...,1]=heat*100
        overlay=Image.fromarray((arr*(1-.6*heat[...,None])+color*.6*heat[...,None]).clip(0,255).astype('uint8'))
        canvas=Image.new('RGB',(2*w,h+40),'white');canvas.paste(rgb,(0,40));canvas.paste(overlay,(w,40));d=ImageDraw.Draw(canvas)
        decoded=pred[i]['decoded'];d.text((4,4),f'index{i}: approved mask centroid GT | predicted center heatmap; count {int(decoded["instance_active"].sum())}',font=font,fill='black')
        for q in sup[i]['instance_valid'].nonzero().flatten().tolist():
            x,y=(sup[i]['mask_centroid'][q]*torch.tensor([w,h])).tolist();d.line((x-8,y+40,x+8,y+40),fill='cyan',width=3);d.line((x,y+32,x,y+48),fill='cyan',width=3)
        for q in decoded['available'].nonzero().flatten().tolist():
            x,y=(decoded['centers'][q]*torch.tensor([w,h])).tolist();x+=w
            color='lime' if decoded['instance_active'][q] else 'orange';d.ellipse((x-5,y+35,x+5,y+45),outline=color,width=3);d.text((x+6,y+40),f'{float(decoded["object_probability"][q]):.3f}',font=font,fill=color)
        canvas.save(examples/f'center_{i}.jpg',quality=93)
    page=[f'<!doctype html><meta charset="utf-8"><title>投影重心热图预训练</title><style>body{{font:18px system-ui;margin:30px}}pre{{white-space:pre-wrap}}img{{max-width:100%}}</style><h1>{final_step}步中心热图预训练：不是姿态模型成绩</h1>',
          '<p>GT仅为loss目标。解码只看图像中心热图，固定0.5阈值、最多2个区域峰。候选oracle仅诊断；本阶段未训练实例分组。</p><pre>'+json.dumps(summary,indent=2)+'</pre>']
    for p in sorted(examples.glob('*.jpg')):page.append(f'<h3>{p.stem}</h3><img src="examples/{p.name}">')
    (OUT/'index.html').write_text('\n'.join(page),encoding='utf-8')
    dest=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-LocalGrouping-20261005')/OUT.name;dest.mkdir(parents=True,exist_ok=True)
    for p in OUT.iterdir():
        if p.suffix in ['.json','.html']:shutil.copy2(p,dest/p.name)
    shutil.copytree(examples,dest/'examples',dirs_exist_ok=True)
    status('complete',step=final_step);state.update(stage='dense_center_warmup_complete',active_job=None,active_pid=None,next='Review center coverage/count and all bottom heatmaps; ifpromising sharethiscenter/backboneinit for matchedpredicted-center grouping arms. Notautomaticpromotion.',updated_utc=now().isoformat());state['completed'].append(f'dense_center_warmup{final_step}');write(C/'state.json',state);print('COMPLETE',json.dumps(summary),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write(OUT/'failure.json',dict(utc=now().isoformat(),traceback=traceback.format_exc()));write(OUT/'status.json',dict(stage='failed',pid=os.getpid(),utc=now().isoformat()))
        raise
