"""Fixed matched learning-curve extension, 1600 -> 4000, preserving each arm trajectory."""
import os,sys,json,datetime as dt,fcntl,time,random,traceback
os.environ['DLClight']='True'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import torch,numpy as np
from preflight_mixed_local_backbone_20261005 import build_backbone,load_rgb,sha,digest
from train_mixed_local_backbone_20261005 import score_one,summarize,buffers_digest,atomic_save
from mousegeoforge.models.mixed_local_evidence import MixedLocalEvidence
from mousegeoforge.models.local_point_evidence_probe import point_evidence_loss,fixed_top_peaks
from mousegeoforge.models.quarter_turn_evidence import quarter_turn
C=ROOT/'output/singleview_fusion_cycle_20261005';INPUT=C/'mixed_local_fulltrain_rgb_inputs_v1';BASEINPUT=C/'mixed_local_rgb_inputs_v1';BASE=C/'mixed_rotation_control_v1';OUT=C/'mixed_rotation4000_v1'
def read(p):return torch.load(p,weights_only=False,map_location='cpu')
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def now():return dt.datetime.now(dt.timezone.utc)
def main():
    plan=json.loads((C/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'));assert now()<deadline;lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);OUT.mkdir(exist_ok=False);torch.set_num_threads(4);torch.manual_seed(17);torch.backends.cuda.matmul.allow_tf32=False
    def status(stage,**kw):write(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=now().isoformat(),**kw))
    state=json.loads((C/'state.json').read_text());state.update(stage='mixed_rotation4000_running',active_job=OUT.name,active_pid=os.getpid(),next='Fixed matched2400extra perarm; no other GPUjob or mid-run changes.',updated_utc=now().isoformat());write(C/'state.json',state)
    manifest={r['index']:r for r in json.loads((INPUT/'inference_manifest.json').read_text())};rows={r['index']:r for r in read(INPUT/'evaluation_only.pt')};evalrows={r['index']:r for r in read(BASEINPUT/'evaluation_only.pt')};small=set(json.loads((C/'highres_preflight_inputs_v1/protocol.json').read_text())['indices']);oldresults=json.loads((BASE/'results.json').read_text());oldprotocol=json.loads((BASE/'protocol.json').read_text());arms=['native_rgb','quarter_turn'];sources={a:BASE/f'{a}_step1600.pt' for a in arms};hashes={a:sha(p) for a,p in sources.items()}
    real={b:[i for i,r in rows.items() if r['metadata']['stage']=='fit' and r['metadata']['kind']=='real' and r['metadata']['physical_bottom']==b] for b in [True,False]};syn={k:[i for i,r in evalrows.items() if r['metadata']['stage']=='fit' and r['metadata']['kind']==k] for k in ['single','dual']};rr=random.Random(53);ss=random.Random(61);rot=random.Random(71);schedule=[[rr.choice(real[j%2==0]),ss.choice(syn['single']),ss.choice(syn['dual'])] for j in range(2400)];turns=[[rot.randrange(4) for _ in ids] for ids in schedule]
    protocol=dict(arms=arms,initial_step=1600,additional_steps=2400,final_step=4000,common_ancestor=oldprotocol['source_checkpoint'],common_ancestor_sha256=oldprotocol['source_sha256'],sources={a:str(p.relative_to(ROOT)) for a,p in sources.items()},source_sha256=hashes,optimizer_states_preserved=True,learning_rates='Each arm preserves identical previous optimizer LR; no restart, no scheduler change.',schedule=schedule,rotation_schedule=turns,real_sampling_seed=53,synthetic_sampling_seed=61,rotation_seed=71,unique_real_seen=len({z[0] for z in schedule}),real_available=821,real_strata=dict(bottom_draws=1200,other_draws=1200),evaluation='Same original362, unrotated, no new CAL/VAL. Fixed endpoints1600/2400/3200/4000; no best-step selection. Cumulative3200additional updates since common800 ancestor perarm.',hypothesis='Short rotation pilot was mixed: bottomTop1 improves113.84to107.50 butotherreal/singlehurt. CE stilldecreasing, so match longer optimization before concluding. No other architecture/data/LR changes.',protected='No GT box/count/label/domain toforward; unknownlabelsnotinvisible; nooriginalchanges, pseudoGT, specialistinitialization, productionpromotion. No lockeddata.',deadline=plan['training_stop_utc'],code_sha256=sha(Path(__file__)))
    write(OUT/'protocol.json',protocol);needed=set(evalrows)|{i for ids in schedule for i in ids};images={}
    for n,i in enumerate(sorted(needed)):
        if now()>=deadline:raise TimeoutError('Input cutoff')
        images[i]=load_rgb(manifest[i]).cuda()
        if (n+1)%100==0:status('loading_rgb',loaded=n+1,total=len(needed));print('RGB',n+1,flush=True)
    curves=[];summary={};performance={};allstart=time.monotonic()
    @torch.no_grad()
    def evaluate(model):
        pred={};details=[]
        for i,r in evalrows.items():
            if now()>=deadline:raise TimeoutError('Evaluation cutoff')
            out=model(images[i],manifest[i]['image_hw']);p=fixed_top_peaks(out,manifest[i]['image_hw']);loss=float(point_evidence_loss(out,r['target'].cuda(),r['eligible'].cuda(),manifest[i]['image_hw']));details.append(dict(index=i,metadata=r['metadata'],loss=loss,metrics=score_one(p,r,manifest[i]['image_hw'])));pred[i]={k:v.cpu() for k,v in p.items() if k!='probability_map' or i in small}
        return pred,details,summarize(details)
    for arm in arms:
        backbone,_=build_backbone();model=MixedLocalEvidence(backbone,adapt_backbone=True).cuda();cp=read(sources[arm]);assert cp['step']==1600;model.backbone.load_state_dict(cp['backbone'],strict=True);model.readout.load_state_dict(cp['readout'],strict=True);model.train();assert digest(model.backbone)==oldresults['performance'][arm]['backbone_after'];bn=buffers_digest(model.backbone);opt=torch.optim.AdamW([dict(params=model.readout.parameters(),lr=.001,initial_lr=.001),dict(params=model.backbone.parameters(),lr=.00001,initial_lr=.00001)],weight_decay=.0001);opt.load_state_dict(cp['optimizer']);lrs=[g['lr'] for g in opt.param_groups];del cp
        assert lrs==oldresults['performance'][arm]['constant_learning_rates'];status('initial_parity',arm=arm,step=1600);p,d,s=evaluate(model);oldp=read(BASE/(arm+'_predictions.pt'));peakdelta=max(float((p[i]['xy']-oldp[i]['xy']).abs().max()) for i in p);probdelta=max(float((p[i]['probability_map']-oldp[i]['probability_map']).abs().max()) for i in small);assert peakdelta==0 and probdelta==0;del oldp;curves.append(dict(arm=arm,step=1600,metrics=s));write(OUT/'curves.json',curves);losses=[];seconds=0.;peak=0
        def checkpoint(step):atomic_save(OUT/f'{arm}_step{step:04d}.pt',dict(backbone={k:v.cpu() for k,v in model.backbone.state_dict().items()},readout={k:v.cpu() for k,v in model.readout.state_dict().items()},step=step,optimizer=opt.state_dict(),source_sha256=hashes[arm]))
        for extra,ids in enumerate(schedule,1):
            step=1600+extra
            if now()>=deadline:checkpoint(step-1);raise TimeoutError('Training cutoff')
            opt.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic();value=0.
            for slot,i in enumerate(ids):
                r=rows[i];x=images[i];target=r['target'].cuda();eligible=r['eligible'].cuda();hw=manifest[i]['image_hw']
                if arm=='quarter_turn':x,target,hw=quarter_turn(x,target,eligible,hw,turns[extra-1][slot])
                out=model(x,hw);loss=point_evidence_loss(out,target,eligible,hw)/3;assert torch.isfinite(loss);loss.backward();value+=float(loss.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);opt.step();torch.cuda.synchronize();seconds+=time.monotonic()-start;peak=max(peak,torch.cuda.max_memory_allocated());losses.append(value)
            if extra%50==0:status('training',arm=arm,step=step,total=4000,training_seconds=seconds);print(arm,step,float(np.mean(losses[-50:])),seconds,flush=True)
            if extra%800==0:
                assert buffers_digest(model.backbone)==bn and [g['lr'] for g in opt.param_groups]==lrs;checkpoint(step);status('evaluating',arm=arm,step=step);p,d,s=evaluate(model);curves.append(dict(arm=arm,step=step,metrics=s,loss_recent100=float(np.mean(losses[-100:]))));write(OUT/'curves.json',curves);write(OUT/f'{arm}_step{step:04d}_details.json',d)
        assert sha(sources[arm])==hashes[arm];performance[arm]=dict(train_seconds=seconds,peak_gpu_bytes=peak,bn_buffers_unchanged=True,bn_digest=bn,constant_learning_rates=lrs,initial_peak_delta=peakdelta,initial_probability_delta=probdelta,backbone_after=digest(model.backbone),losses=losses);summary[arm]=s;atomic_save(OUT/(arm+'_predictions.pt'),p);write(OUT/'results_partial.json',dict(summary=summary,performance=performance));del model,backbone,opt;torch.cuda.empty_cache()
    assert all(sha(ROOT/manifest[i]['image'])==manifest[i]['sha256'] for i in needed);assert all(sha(p)==hashes[a] for a,p in sources.items());write(OUT/'results.json',dict(summary=summary,performance=performance,total_seconds=time.monotonic()-allstart,production_changed=False));status('complete');state.update(stage='mixed_rotation4000_complete',active_job=None,active_pid=None,next='Review fixed4000matchedendpoints, allcurves andbottomfeetRGB; noautomaticextension orpromotion.',updated_utc=now().isoformat());state['completed'].append('mixed_rotation4000_matched_extension');write(C/'state.json',state);print('COMPLETE',json.dumps(summary),flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write(OUT/'failure.json',dict(traceback=traceback.format_exc(),utc=now().isoformat()));write(OUT/'status.json',dict(stage='failed',pid=os.getpid(),utc=now().isoformat()))
        raise
