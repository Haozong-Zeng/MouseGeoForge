"""Matched frozen-feature local evidence diagnostic, not an integrated pose model."""
import sys,json,datetime as dt,fcntl,os,time,math,hashlib,itertools,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np
from mousegeoforge.models.local_point_evidence_probe import LocalPointEvidenceProbe,point_evidence_loss,fixed_top_peaks
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';INPUT=CYCLE/'highres_preflight_inputs_v1';OUT=CYCLE/'local_evidence_probe_v1'
def read(p):return torch.load(p,weights_only=False,map_location='cpu')
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return dt.datetime.now(dt.timezone.utc)
def verify_completed_outputs():
    # Recovery for the final path-join typo after both300-step results were saved.
    # Do not rerun/overwrite training or discard the original failure record.
    plan=json.loads((CYCLE/'plan.json').read_text());lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    result=json.loads((OUT/'results.json').read_text());prediction=read(OUT/'predictions.pt');curves=json.loads((OUT/'curves.json').read_text());verified=[]
    for arm,folder in [('finetuned','highres_preflight_features_v1'),('official','official_quadruped_features_v2')]:
        checkpoint=read(OUT/f'{arm}_step0300.pt');assert checkpoint['step']==300 and len(prediction[arm])==42
        assert [z['step'] for z in curves if z['arm']==arm]==[0,50,100,200,300]
        assert curves[[z['arm'] for z in curves].index(arm)]['step']==0
        assert result['summary'][arm]==[z for z in curves if z['arm']==arm and z['step']==300][0]['metrics']
        for z in json.loads((CYCLE/folder/'manifest.json').read_text()):assert digest(ROOT/z['path'])==z['sha256'];verified.append(z['path'])
        for p in prediction[arm].values():assert torch.isfinite(p['probability_map']).all() and torch.allclose(p['probability_map'].sum((-1,-2)),torch.ones(5),atol=2e-5)
    write(OUT/'completion_verification.json',dict(utc=now().isoformat(),verified_cache_files=len(verified),recovered='Both300-step arms/results/curves/predictions existed; final ROOT[path] path-join typo corrected to ROOT/path. Independent post-completion cache hashes and outputs verified. Original failure retained; no retraining.'))
    write(OUT/'status.json',dict(stage='complete_verified_after_finalization_error',pid=None,utc=now().isoformat()))
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='local_evidence_probe_complete',active_job=None,active_pid=None,next='Review local evidence curves and RGB; completion guard recovered without retraining.',updated_utc=now().isoformat());state['completed'].append('local_evidence_query_independent300x2');write(CYCLE/'state.json',state)
    for arm,s in result['summary'].items():
        for stage,kinds in s.items():
            for kind,z in kinds.items():print(arm,stage,kind,z['eligible'])
def main():
    plan=json.loads((CYCLE/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'));assert now()<deadline
    lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);OUT.mkdir(exist_ok=False);torch.set_num_threads(2);torch.manual_seed(17);torch.backends.cuda.matmul.allow_tf32=False
    def status(stage,**kw):write(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=now().isoformat(),**kw))
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='local_evidence_probe_running',active_job=OUT.name,active_pid=os.getpid(),updated_utc=now().isoformat());write(CYCLE/'state.json',state)
    ev={r['index']:r for r in read(INPUT/'evaluation_only.pt')};visibility=read(CYCLE/'highres_visibility_audit_v1/evaluation_visibility.pt');features={};manifests={};rows={}
    for arm,folder in [('finetuned','highres_preflight_features_v1'),('official','official_quadruped_features_v2')]:
        manifests[arm]=json.loads((CYCLE/folder/'manifest.json').read_text());features[arm]={}
        for z in manifests[arm]:
            assert digest(ROOT/z['path'])==z['sha256'];cache=read(ROOT/z['path']);i=z['index'];features[arm][i]=cache['features'].cuda();assert ev[i]['metadata']['stage'] in ['fit','calibration']
            eligible=ev[i]['valid'].clone() if ev[i]['metadata']['kind']=='real' else visibility[i]['masks']['visible'].clone()
            rows[i]=dict(**ev[i],eligible=eligible,image_hw=cache['image_hw'])
        assert set(features[arm])==set(ev)
    schedule=json.loads((CYCLE/'highres_preflight_pilot_v1/protocol.json').read_text())['schedule'];assert len(schedule)==300 and all(rows[i]['metadata']['stage']=='fit' for ids in schedule for i in ids)
    model=LocalPointEvidenceProbe();initial={k:v.clone() for k,v in model.state_dict().items()};torch.save(initial,OUT/'initial.pt')
    protocol=dict(seed=17,steps=300,images=42,fit=24,internal_cal=18,development=0,arms=list(features),parameters=sum(p.numel() for p in model.parameters()),tests=11,schedule=schedule,initial_sha256=digest(OUT/'initial.pt'),code_sha256=digest(ROOT/'mousegeoforge/models/local_point_evidence_probe.py'),objective='Class-wise spatial CE against equal-weight normalized Gaussian mixtures sigma8inputpx. Synthetic explicit-visible coordinates only; real known coordinates with visibility unknown, no relabeling. Qualified subset unchanged. No main-model loss changes.',optimizer='AdamW1e-3 cosine floor1e-4 wd1e-4 clip1,3images/update same oneeachkind schedule as priorreadout.',forward='Frozen32channel+absoluteXY conv3x3 64 GELU conv1x1 5. No tokens/boxes/GT/count/domain routing. Same normalization and sharedarchitecture across all3kinds.',decode='Fixedtop6 perpointtype, greedy spatial suppression radius8inputpx, noGTcount. Paddedcells excluded.',evaluation='Nearest-top6 is optimistic candidate coverage; one-to-one candidate matching withinpointclass is posthoc evaluation only. Singleinstance top1 separately. No multiinstance pose association/AP claim. 42small internalcohort notindependenttest; pretrainingseen data possible. Only real visibility is unknown, notinvisible.',decision='Compare local evidence quality/calibration across feature sources before deciding instance-conditioning versus representation bottleneck. No integration or automaticcheckpointpromotion.',deadline=plan['training_stop_utc'])
    write(OUT/'protocol.json',protocol)
    def forward(m,arm,i):return m(features[arm][i],rows[i]['image_hw'])
    @torch.no_grad()
    def evaluate(m,arm):
        predictions={};details=[]
        for i,r in rows.items():
            if now()>=deadline:raise TimeoutError('Training/evaluation cutoff')
            out=forward(m,arm,i);p=fixed_top_peaks(out,r['image_hw']);predictions[i]={k:v.cpu() for k,v in p.items()};loss=float(point_evidence_loss(out,r['gt'].cuda(),r['eligible'].cuda(),r['image_hw']))
            xy=p['xy'].cpu();available=p['available'].cpu();grid=out['grid'].cpu();prob=p['probability_map'].cpu();group_metrics={}
            for maskname,mask in [('eligible',r['eligible']),('all_known',r['valid']),('explicit_occluded',visibility[i]['masks']['occluded'])]:
                nearest=[];assigned=[];top1=[];mass=[];uniform_mass=[];point_info=[]
                for k in range(5):
                    queries=mask[:,k].nonzero().flatten();target=r['gt'][queries,k]
                    if not len(target):continue
                    dist=torch.cdist(target*r['pixels'],xy[k]*r['pixels']).masked_fill(~available[k],torch.inf);nearest+=dist.min(1).values.tolist()
                    perms=torch.tensor(list(itertools.permutations(range(6),len(target))));cost=dist[torch.arange(len(target))[None],perms].sum(1);choice=perms[cost.argmin()];assigned+=dist[torch.arange(len(target)),choice].tolist()
                    if r['metadata']['kind']!='dual':top1+=dist[:,0].tolist()
                    for j,(q,t) in enumerate(zip(queries.tolist(),target)):
                        near=((grid-t)*r['pixels']).square().sum(-1)<=8.**2;mass.append(float(prob[k][near].sum()));uniform_mass.append(float((near&out['spatial_valid'].cpu()).sum()/out['spatial_valid'].sum().cpu()));point_info.append(dict(query=q,point=k,nearest=float(dist[j].min()),assigned=float(dist[j,choice[j]]),top1=float(dist[j,0]),mass=mass[-1]))
                group_metrics[maskname]=dict(nearest=nearest,assigned=assigned,top1_single=top1,mass8=mass,uniform_mass8=uniform_mass,points=point_info)
            details.append(dict(index=i,metadata=r['metadata'],loss=loss,metrics=group_metrics))
        summary={}
        for stage in ['fit','calibration']:
            summary[stage]={}
            for kind in ['real','single','dual']:
                selected=[z for z in details if z['metadata']['stage']==stage and z['metadata']['kind']==kind];summary[stage][kind]={}
                for group in ['eligible','all_known','explicit_occluded']:
                    def values(key):return np.asarray(sum([z['metrics'][group][key] for z in selected],[]))
                    a=values('nearest');b=values('assigned');c=values('top1_single');mass=values('mass8');uniform=values('uniform_mass8')
                    summary[stage][kind][group]=dict(n=len(a),nearest_top6_mean=float(a.mean()) if len(a) else None,nearest_top6_pck32=float((a<=32).mean()) if len(a) else None,unique_top6_mean=float(b.mean()) if len(b) else None,top1_single_mean=float(c.mean()) if len(c) else None,mass8_mean=float(mass.mean()) if len(mass) else None,uniform_mass8_mean=float(uniform.mean()) if len(uniform) else None,loss_mean=float(np.mean([z['loss'] for z in selected])))
        return predictions,details,summary
    curves=[];results={};predictions={};performance={}
    for arm in features:
        model=LocalPointEvidenceProbe().cuda();model.load_state_dict(initial);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);p,d,s=evaluate(model,arm);curves.append(dict(arm=arm,step=0,metrics=s));seconds=0.;peak=0;losses=[]
        for step,ids in enumerate(schedule,1):
            if now()>=deadline:raise TimeoutError('Training cutoff')
            opt.zero_grad(set_to_none=True);factor=.1+.9*.5*(1+math.cos(math.pi*(step-1)/300))
            for group in opt.param_groups:group['lr']=.001*factor
            torch.cuda.synchronize();start=time.monotonic();torch.cuda.reset_peak_memory_stats();value=0.
            for i in ids:
                r=rows[i];out=forward(model,arm,i);loss=point_evidence_loss(out,r['gt'].cuda(),r['eligible'].cuda(),r['image_hw'])/3;assert torch.isfinite(loss);loss.backward();value+=float(loss.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);opt.step();torch.cuda.synchronize();seconds+=time.monotonic()-start;peak=max(peak,torch.cuda.max_memory_allocated());losses.append(value)
            if step%25==0:status('training',arm=arm,step=step);print(arm,step,float(np.mean(losses[-25:])),flush=True)
            if step in [50,100,200,300]:
                p,d,s=evaluate(model,arm);curves.append(dict(arm=arm,step=step,metrics=s,loss_recent=float(np.mean(losses[-50:]))));write(OUT/'curves.json',curves);torch.save(dict(state={k:v.cpu() for k,v in model.state_dict().items()},step=step,optimizer=opt.state_dict()),OUT/f'{arm}_step{step:04d}.pt')
        predictions[arm]=p;results[arm]=s;write(OUT/f'{arm}_details.json',d);performance[arm]=dict(training_seconds=seconds,peak_gpu_bytes=peak,losses=losses);del model,opt
    torch.save(predictions,OUT/'predictions.pt');write(OUT/'results.json',dict(summary=results,performance=performance,production_changed=False))
    for manifest in manifests.values():assert all(digest(ROOT/z['path'])==z['sha256'] for z in manifest)
    status('complete');state.update(stage='local_evidence_probe_complete',active_job=None,active_pid=None,next='Review local evidence curves/candidate coverage and rawRGB; candidate coverage is not final pose accuracy.',updated_utc=now().isoformat());state['completed'].append('local_evidence_query_independent300x2');write(CYCLE/'state.json',state);print(json.dumps(results),flush=True)
if __name__=='__main__':
    try:verify_completed_outputs() if '--verify-completed' in sys.argv else main()
    except Exception:
        if OUT.exists():write(OUT/'failure.json',dict(traceback=traceback.format_exc(),utc=now().isoformat(),pid=os.getpid()))
        raise
