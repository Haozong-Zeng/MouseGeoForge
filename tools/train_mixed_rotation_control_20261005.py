"""Matched continuation: native RGB versus quarter-turn rotation augmentation."""
import os,sys,json,datetime as dt,fcntl,time,random,traceback
os.environ['DLClight']='True'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import torch,numpy as np
from preflight_mixed_local_backbone_20261005 import build_backbone,load_rgb,sha,digest,PREP
from train_mixed_local_backbone_20261005 import score_one,summarize,buffers_digest,atomic_save
from mousegeoforge.models.mixed_local_evidence import MixedLocalEvidence
from mousegeoforge.models.quarter_turn_evidence import quarter_turn
from mousegeoforge.models.local_point_evidence_probe import point_evidence_loss,fixed_top_peaks
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';INPUT=CYCLE/'mixed_local_fulltrain_rgb_inputs_v1';BASEINPUT=CYCLE/'mixed_local_rgb_inputs_v1';BASE=CYCLE/'mixed_local_backbone800_v1';OUT=CYCLE/'mixed_rotation_control_v1'
def read(p):return torch.load(p,weights_only=False,map_location='cpu')
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def now():return dt.datetime.now(dt.timezone.utc)
def main():
    preflight=json.loads((CYCLE/'quarter_turn_preflight_v1/results.json').read_text());assert preflight['tests']==7 and preflight['fit_coordinate_transforms_checked']==3864
    plan=json.loads((CYCLE/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'));assert now()<deadline;lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);OUT.mkdir(exist_ok=False);torch.set_num_threads(4);torch.manual_seed(17);torch.backends.cuda.matmul.allow_tf32=False
    def status(stage,**kw):write(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=now().isoformat(),**kw))
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='mixed_rotation_control_running',active_job=OUT.name,active_pid=os.getpid(),updated_utc=now().isoformat());write(CYCLE/'state.json',state)
    manifest={z['index']:z for z in json.loads((INPUT/'inference_manifest.json').read_text())};rows={z['index']:z for z in read(INPUT/'evaluation_only.pt')};base_rows={z['index']:z for z in read(BASEINPUT/'evaluation_only.pt')};base_manifest={z['index']:z for z in json.loads((BASEINPUT/'inference_manifest.json').read_text())};small=set(json.loads((CYCLE/'highres_preflight_inputs_v1/protocol.json').read_text())['indices']);source=BASE/'train_official_backbone_step0800.pt';source_hash=sha(source);prior_results=json.loads((BASE/'results.json').read_text());prior_predictions=read(BASE/'train_official_backbone_predictions.pt');prior_backbone=prior_results['performance']['train_official_backbone']['backbone_after']
    for i,r in base_rows.items():
        assert manifest[i]['sha256']==base_manifest[i]['sha256'] and rows[i]['metadata']==r['metadata']
        for k in ['target','coordinate_valid','visibility_valid','visible','eligible']:assert torch.equal(rows[i][k],r[k])
    assert all(r['metadata']['stage']=='fit' and r['metadata']['kind']=='real' for i,r in rows.items() if i not in base_rows)
    arms=['native_rgb','quarter_turn'];realpools={a:{b:[i for i,r in rows.items() if r['metadata']['stage']=='fit' and r['metadata']['kind']=='real' and r['metadata']['physical_bottom']==b] for b in [True,False]} for a in arms};synthetic={k:[i for i,r in base_rows.items() if r['metadata']['stage']=='fit' and r['metadata']['kind']==k] for k in ['single','dual']};synrng=random.Random(29);synschedule=[[synrng.choice(synthetic[k]) for k in ['single','dual']] for _ in range(800)];schedules={}
    for arm in arms:
        rng=random.Random(17);schedules[arm]=[[rng.choice(realpools[arm][step%2==0]),*synschedule[step]] for step in range(800)]
    assert all(a[1:]==b[1:] and rows[a[0]]['metadata']['physical_bottom']==rows[b[0]]['metadata']['physical_bottom'] for a,b in zip(schedules[arms[0]],schedules[arms[1]]))
    protocol=dict(initial_step=800,additional_steps=800,final_step=1600,arms=arms,source_checkpoint=str(source.relative_to(ROOT)),source_sha256=source_hash,shared_initial_backbone_digest=prior_backbone,source_optimizer_preserved=True,learning_rate='Hold optimizer last800-step LR constant, no restart orotherrecipechange; samebotharms.',schedules=schedules,real_pool_sizes={a:{str(b):len(p) for b,p in pp.items()} for a,pp in realpools.items()},unique_real_rows_used={a:len({z[0] for z in ss}) for a,ss in schedules.items()},images_available=len(rows),common_evaluation_images=len(base_rows),common_fit=273,common_internal_cal=89,extra_real_fit_images=len(rows)-len(base_rows),evaluation='Same original362 images botharms; fixedendpoint1600, curve800/1000/1200/1600. NewrealTRAIN nottreatedasvalidation. Original42fullmaps forvisual. Candidate coverage notfinalinstancepose/AP.',controlled='Same800checkpoint+optimizer, allbackbone29,305,536andhead19,973trainable, sameBNfixed, same800additionalupdates, one real/single/dual eachstep, exactlysame syntheticdraws andalternatingrealbottom/nonbottomstratum; onlyrealtrainingpool changes128to821. Availablepool!=allimagesseen.',labels='Sameall-instancevisible-synthetic andcoordinate-known-real loss; noGTforward, labelchanges/pseudoGT/validationadaptation.',seed=17,synthetic_schedule_seed=29,deadline=plan['training_stop_utc'],code_sha256=sha(Path(__file__)))
    protocol.update(controlled='Same original800 model+optimizer and expanded821real pool/schedule; only quarter-turn FIT augmentation differs. No mirrors, no point identity swaps, no augmentation at evaluation. 800 additional updates to1600, same LR/BN/mixed domains.', rotations='All FIT domains: uniform fixed seed43 k=0,1,2,3, pixel-exact torch.rot90 without interpolation. Coordinate uses pixel-index/W,H. No camera or3D training here.');rotation_rng=random.Random(43);rotation_schedule=[[rotation_rng.randrange(4) for _ in ids] for ids in schedules['native_rgb']];protocol['rotation_schedule']=rotation_schedule
    assert schedules['native_rgb']==schedules['quarter_turn']
    write(OUT/'protocol.json',protocol);images={}
    # Only actual scheduled images and common evaluation inputs are loaded.
    needed=set(base_rows)|{i for ss in schedules.values() for ids in ss for i in ids}
    for n,i in enumerate(sorted(needed)):
        if now()>=deadline:raise TimeoutError('Loading cutoff')
        images[i]=load_rgb(manifest[i]).cuda()
        if (n+1)%100==0:status('loading_rgb',loaded=n+1,total=len(needed));print('RGB',n+1,flush=True)
    curves=[];summary={};performance={};start_all=time.monotonic()
    @torch.no_grad()
    def evaluate(model):
        pred={};details=[]
        for i,r in base_rows.items():
            if now()>=deadline:raise TimeoutError('Evaluation cutoff')
            out=model(images[i],manifest[i]['image_hw']);p=fixed_top_peaks(out,manifest[i]['image_hw']);loss=float(point_evidence_loss(out,r['target'].cuda(),r['eligible'].cuda(),manifest[i]['image_hw']));details.append(dict(index=i,metadata=r['metadata'],loss=loss,metrics=score_one(p,r,manifest[i]['image_hw'])));pred[i]={k:v.cpu() for k,v in p.items() if k!='probability_map' or i in small}
        return pred,details,summarize(details)
    for arm in arms:
        backbone,info=build_backbone();model=MixedLocalEvidence(backbone,adapt_backbone=True).cuda();cp=read(source);assert cp['step']==800;model.backbone.load_state_dict(cp['backbone'],strict=True);model.readout.load_state_dict(cp['readout'],strict=True);model.train();assert digest(model.backbone)==prior_backbone;bn_before=buffers_digest(model.backbone)
        opt=torch.optim.AdamW([dict(params=model.readout.parameters(),lr=.001,initial_lr=.001),dict(params=model.backbone.parameters(),lr=.00001,initial_lr=.00001)],weight_decay=.0001);opt.load_state_dict(cp['optimizer']);lrs=[g['lr'] for g in opt.param_groups];del cp
        status('initial_parity',arm=arm,step=800);p,d,s=evaluate(model);peak_xy_delta=max(float((p[i]['xy']-prior_predictions[i]['xy']).abs().max()) for i in base_rows);prob_delta=max(float((p[i]['probability_map']-prior_predictions[i]['probability_map']).abs().max()) for i in small);assert peak_xy_delta==0 and prob_delta==0
        curves.append(dict(arm=arm,step=800,metrics=s));write(OUT/'curves.json',curves);seconds=0.;peak=0;losses=[]
        for extra,ids in enumerate(schedules[arm],1):
            step=800+extra
            if now()>=deadline:
                atomic_save(OUT/f'{arm}_deadline_step{step-1:04d}.pt',dict(backbone={k:v.cpu() for k,v in model.backbone.state_dict().items()},readout={k:v.cpu() for k,v in model.readout.state_dict().items()},step=step-1,optimizer=opt.state_dict()));raise TimeoutError('Training cutoff')
            opt.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic();value=0.
            for slot,i in enumerate(ids):
                r=rows[i];x=images[i];target=r['target'].cuda();eligible=r['eligible'].cuda();hw=manifest[i]['image_hw']
                if arm=='quarter_turn':x,target,hw=quarter_turn(x,target,eligible,hw,rotation_schedule[extra-1][slot])
                out=model(x,hw);loss=point_evidence_loss(out,target,eligible,hw)/3;assert torch.isfinite(loss);loss.backward();value+=float(loss.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);opt.step();torch.cuda.synchronize();seconds+=time.monotonic()-start;peak=max(peak,torch.cuda.max_memory_allocated());losses.append(value)
            if extra%25==0:status('training',arm=arm,step=step,total=1600,training_seconds=seconds);print(arm,step,float(np.mean(losses[-25:])),seconds,flush=True)
            if extra in [200,400,800]:
                assert [g['lr'] for g in opt.param_groups]==lrs;assert buffers_digest(model.backbone)==bn_before;atomic_save(OUT/f'{arm}_step{step:04d}.pt',dict(backbone={k:v.cpu() for k,v in model.backbone.state_dict().items()},readout={k:v.cpu() for k,v in model.readout.state_dict().items()},step=step,optimizer=opt.state_dict() if extra==800 else None,source_sha256=source_hash));status('evaluating',arm=arm,step=step);p,d,s=evaluate(model);curves.append(dict(arm=arm,step=step,metrics=s,loss_recent100=float(np.mean(losses[-100:]))));write(OUT/'curves.json',curves);write(OUT/f'{arm}_step{step:04d}_details.json',d)
        assert buffers_digest(model.backbone)==bn_before and sha(source)==source_hash;performance[arm]=dict(train_seconds=seconds,peak_gpu_bytes=peak,constant_learning_rates=lrs,bn_digest=bn_before,bn_buffers_unchanged=True,initial_peak_xy_delta=peak_xy_delta,initial_probability_delta=prob_delta,backbone_after=digest(model.backbone),losses=losses);summary[arm]=s;atomic_save(OUT/f'{arm}_predictions.pt',p);write(OUT/'results_partial.json',dict(summary=summary,performance=performance));del model,backbone,opt;torch.cuda.empty_cache()
    assert all(sha(ROOT/manifest[i]['image'])==manifest[i]['sha256'] for i in needed);assert sha(source)==source_hash;write(OUT/'results.json',dict(summary=summary,performance=performance,total_seconds=time.monotonic()-start_all,production_changed=False));status('complete');state.update(stage='mixed_rotation_control_complete',active_job=None,active_pid=None,next='Review native versus quarter-turn800additional paired RGB training, original evaluation only, realbottom/LR andsyntheticretention. Noautomaticpromotion.',updated_utc=now().isoformat());state['completed'].append('mixed_real_pool_continuation800x2');write(CYCLE/'state.json',state);print('COMPLETE',json.dumps(summary),flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write(OUT/'failure.json',dict(traceback=traceback.format_exc(),utc=now().isoformat()));write(OUT/'status.json',dict(stage='failed',pid=os.getpid(),utc=now().isoformat()))
        raise
