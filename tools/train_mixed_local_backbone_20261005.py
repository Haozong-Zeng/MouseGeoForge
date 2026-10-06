"""Actual-RGB frozen/adapted official backbone control on mixed TRAIN sources."""
import os,sys,json,datetime as dt,fcntl,math,time,itertools,traceback,hashlib
os.environ['DLClight']='True'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import torch,numpy as np
from preflight_mixed_local_backbone_20261005 import build_backbone,load_rgb,sha,digest,PREP
from mousegeoforge.models.mixed_local_evidence import MixedLocalEvidence
from mousegeoforge.models.local_point_evidence_probe import point_evidence_loss,fixed_top_peaks
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';INPUT=CYCLE/'mixed_local_rgb_inputs_v1';OUT=CYCLE/'mixed_local_backbone800_v1'
def now():return dt.datetime.now(dt.timezone.utc)
def read(p):return torch.load(p,weights_only=False,map_location='cpu')
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def atomic_save(path,value):
    temp=path.with_suffix('.tmp');torch.save(value,temp);temp.replace(path)
def buffers_digest(model):
    h=hashlib.sha256()
    for k,v in model.named_buffers():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def score_one(peaks,row,image_hw):
    xy=peaks['xy'].cpu();available=peaks['available'].cpu();prob=peaks['probability_map'].cpu();ih,iw=image_hw;hh,ww=prob.shape[-2:];yy,xx=torch.meshgrid((torch.arange(hh)+.5)*4,(torch.arange(ww)+.5)*4,indexing='ij');grid=torch.stack([xx/iw,yy/ih],-1);masks={'eligible':row['eligible'],'all_known':row['coordinate_valid'],'explicit_occluded':row['coordinate_valid']&row['visibility_valid']&~row['visible']};result={}
    for group,mask in masks.items():
        near=[];assigned=[];first=[];mass=[]
        for k in range(5):
            target=row['target'][mask[:,k],k]
            if not len(target):continue
            cost=torch.cdist(target*row['pixels'],xy[k]*row['pixels']).masked_fill(~available[k],torch.inf);near+=cost.min(1).values.tolist();perms=torch.tensor(list(itertools.permutations(range(6),len(target))));sums=cost[torch.arange(len(target))[None],perms].sum(1);assignment=perms[sums.argmin()];assigned+=cost[torch.arange(len(target)),assignment].tolist()
            if row['metadata']['kind']!='dual':first+=cost[:,0].tolist()
            for point in target:
                around=((grid-point)*row['pixels']).square().sum(-1)<=64;mass.append(float(prob[k][around].sum()))
        result[group]=dict(nearest=near,assigned=assigned,top1_single=first,mass8=mass)
    return result

def summarize(details):
    summary={}
    for stage in ['fit','calibration']:
        summary[stage]={}
        for kind in ['real','single','dual','real_bottom_feet']:
            selected=[r for r in details if r['metadata']['stage']==stage and (r['metadata']['kind']==kind if kind!='real_bottom_feet' else r['metadata']['kind']=='real' and r['metadata']['physical_bottom'])]
            # real_bottom here includes nose; avoid falsely presenting a feet-only result.
            if kind=='real_bottom_feet':continue
            summary[stage][kind]={}
            for group in ['eligible','all_known','explicit_occluded']:
                def values(key):return np.asarray(sum([z['metrics'][group][key] for z in selected],[]))
                a=values('nearest');b=values('assigned');c=values('top1_single');m=values('mass8')
                summary[stage][kind][group]=dict(n=len(a),nearest_top6_mean=float(a.mean()) if len(a) else None,nearest_top6_pck32=float((a<=32).mean()) if len(a) else None,unique_top6_mean=float(b.mean()) if len(b) else None,top1_single_mean=float(c.mean()) if len(c) else None,mass8_mean=float(m.mean()) if len(m) else None,loss_mean=float(np.mean([z['loss'] for z in selected])))
    return summary

def main():
    plan=json.loads((CYCLE/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'));assert now()<deadline;lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);OUT.mkdir(exist_ok=False);torch.set_num_threads(4);torch.manual_seed(17);torch.backends.cuda.matmul.allow_tf32=False
    def status(stage,**kw):write(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=now().isoformat(),**kw))
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='mixed_local_backbone800_running',active_job=OUT.name,active_pid=os.getpid(),updated_utc=now().isoformat());write(CYCLE/'state.json',state)
    preflight=json.loads((CYCLE/'mixed_local_preflight_v1/results.json').read_text());assert preflight['max_feature_difference']==0 and preflight['smoke_weights_discarded'] and preflight['bn_buffers_unchanged'];initial=read(CYCLE/'mixed_local_preflight_v1/initial_readout.pt');manifest={z['index']:z for z in json.loads((INPUT/'inference_manifest.json').read_text())};rows={z['index']:z for z in read(INPUT/'evaluation_only.pt')};small=set(json.loads((CYCLE/'highres_preflight_inputs_v1/protocol.json').read_text())['indices']);protocol=json.loads((INPUT/'protocol.json').read_text());schedule=protocol['schedule'];assert len(schedule)==800
    images={}
    for n,(i,z) in enumerate(manifest.items()):
        if now()>=deadline:raise TimeoutError('Loading cutoff')
        images[i]=load_rgb(z).cuda()
        if (n+1)%50==0:status('loading_rgb',loaded=n+1,total=len(manifest));print('RGB',n+1,flush=True)
    protocol.update(torch_version=torch.__version__,parameter_count_head=19973,parameter_count_adapt_backbone=preflight['trainable_backbone_parameters'],initial_readout_sha256=sha(CYCLE/'mixed_local_preflight_v1/initial_readout.pt'),initial_backbone_digest=preflight['initial_backbone_digest'],scope='Query-independent local evidence diagnostic, notfinalmultiinstancepose. Frozenversusadapted BOTHactualRGB sameDLCtorch2.7.1 environment andsameinitial/readout/schedule800. Moreparams/compute inadaptarm explicitlyreported. Oldsmall-cohort results are notmatched controls.',evaluation='ALL273FIT/89internalcal at0/200/400/800; fixedendpoint; fixedtop6 candidates/class, optimisticnearest andone-to-oneposthoccoverage; singleinstanceTop1. NoGTmodelinput. Fullmaps retainedonlyoriginal42 forvisualization.',code_sha256=sha(Path(__file__)),deadline=plan['training_stop_utc']);write(OUT/'protocol.json',protocol)
    curves=[];summary={};performance={};predictions={};runtime_start=time.monotonic()
    def forward(model,i):return model(images[i],manifest[i]['image_hw'])
    @torch.no_grad()
    def evaluate(model):
        details=[];pred={}
        for i,r in rows.items():
            if now()>=deadline:raise TimeoutError('Evaluation cutoff')
            out=forward(model,i);loss=float(point_evidence_loss(out,r['target'].cuda(),r['eligible'].cuda(),manifest[i]['image_hw']));p=fixed_top_peaks(out,manifest[i]['image_hw']);details.append(dict(index=i,metadata=r['metadata'],loss=loss,metrics=score_one(p,r,manifest[i]['image_hw'])));pred[i]={k:v.cpu() for k,v in p.items() if k!='probability_map' or i in small}
        return pred,details,summarize(details)
    for arm in ['frozen_official_backbone','train_official_backbone']:
        adapt=arm=='train_official_backbone';backbone,info=build_backbone();model=MixedLocalEvidence(backbone,adapt_backbone=adapt).cuda();model.readout.load_state_dict(initial);model.train();before=digest(model.backbone);assert before==preflight['initial_backbone_digest'];bn_before=buffers_digest(model.backbone)
        params=[dict(params=model.readout.parameters(),lr=.001,initial_lr=.001)]
        if adapt:params.append(dict(params=model.backbone.parameters(),lr=.00001,initial_lr=.00001))
        opt=torch.optim.AdamW(params,weight_decay=.0001);status('evaluating',arm=arm,step=0);p,d,s=evaluate(model);curves.append(dict(arm=arm,step=0,metrics=s));write(OUT/'curves.json',curves);seconds=0.;peak=0;losses=[];grad=[]
        for step,ids in enumerate(schedule,1):
            if now()>=deadline:
                atomic_save(OUT/f'{arm}_deadline_step{step-1:04d}.pt',dict(backbone={k:v.cpu() for k,v in model.backbone.state_dict().items()} if adapt else None,readout={k:v.cpu() for k,v in model.readout.state_dict().items()},step=step-1,optimizer=opt.state_dict()));raise TimeoutError('Training cutoff checkpoint saved')
            opt.zero_grad(set_to_none=True);factor=.1+.9*.5*(1+math.cos(math.pi*(step-1)/800))
            for g in opt.param_groups:g['lr']=g['initial_lr']*factor
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic();value=0.
            for i in ids:
                r=rows[i];out=forward(model,i);loss=point_evidence_loss(out,r['target'].cuda(),r['eligible'].cuda(),manifest[i]['image_hw'])/3;assert torch.isfinite(loss);loss.backward();value+=float(loss.detach())
            if step in [1,2,200,400,800]:
                norm=sum(float(p.grad.square().sum()) for p in model.backbone.parameters() if p.grad is not None)**.5;assert np.isfinite(norm);grad.append(dict(step=step,norm=norm))
                if adapt and step>1:assert norm>0
                if not adapt:assert norm==0
            torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);opt.step();torch.cuda.synchronize();seconds+=time.monotonic()-start;peak=max(peak,torch.cuda.max_memory_allocated());losses.append(value)
            if step%25==0:status('training',arm=arm,step=step,total=800,training_seconds=seconds);print(arm,step,float(np.mean(losses[-25:])),'seconds',seconds,flush=True)
            if step in [200,400,800]:
                atomic_save(OUT/f'{arm}_step{step:04d}.pt',dict(backbone={k:v.detach().cpu() for k,v in model.backbone.state_dict().items()} if adapt else None,readout={k:v.detach().cpu() for k,v in model.readout.state_dict().items()},step=step,arm=arm,optimizer=opt.state_dict() if step==800 else None,source_sha256=info['sha256']))
                assert buffers_digest(model.backbone)==bn_before;status('evaluating',arm=arm,step=step);p,d,s=evaluate(model);curves.append(dict(arm=arm,step=step,metrics=s,loss_recent100=float(np.mean(losses[-100:]))));write(OUT/'curves.json',curves);write(OUT/f'{arm}_step{step:04d}_details.json',d)
        after=digest(model.backbone);assert buffers_digest(model.backbone)==bn_before;assert (after!=before) if adapt else (after==before);assert sha(PREP/'superanimal_quadruped_hrnet_w32.pt')==info['sha256'];summary[arm]=s;predictions[arm]=p;performance[arm]=dict(train_seconds=seconds,peak_gpu_bytes=peak,gradient_norms=grad,backbone_before=before,backbone_after=after,bn_digest=bn_before,bn_buffers_unchanged=True,losses=losses);write(OUT/'performance.json',performance);atomic_save(OUT/f'{arm}_predictions.pt',p);write(OUT/'results_partial.json',dict(summary=summary,performance=performance));del model,backbone,opt;torch.cuda.empty_cache()
    for z in manifest.values():assert sha(ROOT/z['image'])==z['sha256']
    write(OUT/'results.json',dict(summary=summary,performance=performance,total_seconds=time.monotonic()-runtime_start,images=len(rows),production_changed=False));status('complete');state.update(stage='mixed_local_backbone800_complete',active_job=None,active_pid=None,next='Review mixed800 frozen/adapted convergence, all3kind and real-bottomRGB before any extension/integration; these are local evidence notinstanceposes.',updated_utc=now().isoformat());state['completed'].append('mixed_actualRGB_backbone800x2');write(CYCLE/'state.json',state);print('COMPLETE',json.dumps(summary),flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write(OUT/'failure.json',dict(traceback=traceback.format_exc(),utc=now().isoformat(),pid=os.getpid()));write(OUT/'status.json',dict(stage='failed',pid=os.getpid(),utc=now().isoformat()))
        raise
