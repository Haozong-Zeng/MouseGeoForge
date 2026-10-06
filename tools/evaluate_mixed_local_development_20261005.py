"""Final-only fixed checkpoint comparison on reused development sources; no updates."""
import os,sys,json,datetime as dt,fcntl,time,traceback
os.environ['DLClight']='True'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import torch,numpy as np
from preflight_mixed_local_backbone_20261005 import build_backbone,load_rgb,sha,digest
from train_mixed_local_backbone_20261005 import score_one,atomic_save,buffers_digest
from mousegeoforge.models.mixed_local_evidence import MixedLocalEvidence
from mousegeoforge.models.local_point_evidence_probe import point_evidence_loss,fixed_top_peaks
C=ROOT/'output/singleview_fusion_cycle_20261005';INPUT=C/'mixed_local_development_inputs_v1';OUT=C/'mixed_local_development_eval_v1';TRAIN=C/'mixed_rotation4000_v1'
def now():return dt.datetime.now(dt.timezone.utc)
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def read(p):return torch.load(p,weights_only=False,map_location='cpu')
def aggregate(details):
    result={}
    selectors={'real':lambda m:m['kind']=='real','single':lambda m:m['kind']=='single','dual':lambda m:m['kind']=='dual','real_bottom':lambda m:m['kind']=='real' and m['physical_bottom'],'real_other':lambda m:m['kind']=='real' and not m['physical_bottom'],'real_bottom_feet':lambda m:m['kind']=='real' and m['physical_bottom']}
    for source in ['FR2','AZR2L2']:selectors[source]=lambda m,s=source:m['kind']=='real' and m['source']==s
    for name,selector in selectors.items():
        chosen=[r for r in details if selector(r['metadata'])];result[name]={}
        for group in ['eligible','all_known','explicit_occluded']:
            field='feet_metrics' if name.endswith('_feet') else 'metrics';values=lambda key:np.asarray(sum([r[field][group][key] for r in chosen],[]));a=values('nearest');b=values('assigned');c=values('top1_single');mass=values('mass8');losses=[r['loss'] for r in chosen if r['eligible_count']>0]
            result[name][group]=dict(images=len(chosen),images_without_eligible=sum(r['eligible_count']==0 for r in chosen),n=len(a),nearest_top6_mean=float(a.mean()) if len(a) else None,unique_top6_mean=float(b.mean()) if len(b) else None,top1_single_mean=float(c.mean()) if len(c) else None,top1_single_pck16=float((c<=16).mean()) if len(c) else None,top1_single_p90=float(np.percentile(c,90)) if len(c) else None,nearest_top6_pck32=float((a<=32).mean()) if len(a) else None,mass8_mean=float(mass.mean()) if len(mass) else None,loss_mean_nonempty=float(np.mean(losses)) if losses and not name.endswith('_feet') else None)
    return result
@torch.no_grad()
def main():
    assert (TRAIN/'results.json').exists(),'Both4000arms must finish before final evaluation';plan=json.loads((C/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['hard_deadline_utc'].replace('Z','+00:00'))-dt.timedelta(minutes=5);assert now()<deadline;lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);OUT.mkdir(exist_ok=False);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    def status(stage,**kw):write(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=now().isoformat(),**kw))
    state=json.loads((C/'state.json').read_text());state.update(stage='final_development_evaluation',active_job=OUT.name,active_pid=os.getpid(),next='Read-only fixed800/native4000/rotated4000 development check; no post-development training thiscycle.',updated_utc=now().isoformat());write(C/'state.json',state)
    manifest={r['index']:r for r in json.loads((INPUT/'inference_manifest.json').read_text())};rows={r['index']:r for r in read(INPUT/'evaluation_only.pt')};assert len(rows)==144;selected=[]
    for bottom in [True,False]:selected.append(next(i for i,r in rows.items() if r['metadata']['kind']=='real' and r['metadata']['physical_bottom']==bottom))
    for kind in ['single','dual']:selected.extend([i for i,r in rows.items() if r['metadata']['kind']==kind][:2])
    sources={'baseline800':C/'mixed_local_backbone800_v1/train_official_backbone_step0800.pt','native4000':TRAIN/'native_rgb_step4000.pt','rotated4000':TRAIN/'quarter_turn_step4000.pt'};hashes={a:sha(p) for a,p in sources.items()};protocol=json.loads((INPUT/'protocol.json').read_text());protocol.update(checkpoints={a:str(p.relative_to(ROOT)) for a,p in sources.items()},checkpoint_sha256=hashes,visual_indices=selected,inference='FP32 nativewholeRGB, sameAlbNormalize/pad32/frozenBN, 5pointqueryindependenthead, noGTforward. FixedTop6 andinputpixelerrors. Empty-label image retained; lossmean excludes it.',new_training=False,deadline=deadline.isoformat(),code_sha256=sha(Path(__file__)));write(OUT/'protocol.json',protocol)
    summary={};performance={}
    for arm,path in sources.items():
        backbone,_=build_backbone();model=MixedLocalEvidence(backbone,adapt_backbone=False).cuda();cp=read(path);assert cp['step']==(800 if arm=='baseline800' else 4000);model.backbone.load_state_dict(cp['backbone'],strict=True);model.readout.load_state_dict(cp['readout'],strict=True);model.eval();del cp;before=digest(model.backbone);bn=buffers_digest(model.backbone);headbefore={k:v.clone() for k,v in model.readout.state_dict().items()};torch.cuda.reset_peak_memory_stats();pred={};details=[];start=time.monotonic()
        for n,(i,r) in enumerate(rows.items()):
            if now()>=deadline:raise TimeoutError('Final evaluation cutoff')
            x=load_rgb(manifest[i]).cuda();hw=manifest[i]['image_hw'];out=model(x,hw);p=fixed_top_peaks(out,hw);loss=float(point_evidence_loss(out,r['target'].cuda(),r['eligible'].cuda(),hw));feet=dict(r)
            for key in ['eligible','coordinate_valid']:feet[key]=r[key].clone();feet[key][:,0]=False
            details.append(dict(index=i,metadata=r['metadata'],loss=loss,eligible_count=int(r['eligible'].sum()),metrics=score_one(p,r,hw),feet_metrics=score_one(p,feet,hw)));pred[i]={k:v.cpu() for k,v in p.items() if k!='probability_map' or i in selected}
            if (n+1)%24==0:status('evaluating',arm=arm,images=n+1,total=144);print(arm,n+1,flush=True)
        assert digest(model.backbone)==before and buffers_digest(model.backbone)==bn;assert all(torch.equal(v,headbefore[k]) for k,v in model.readout.state_dict().items());assert sha(path)==hashes[arm];summary[arm]=aggregate(details);performance[arm]=dict(seconds=time.monotonic()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(),weights_and_bn_unchanged=True,backbone_digest=before);atomic_save(OUT/(arm+'_predictions.pt'),pred);write(OUT/(arm+'_details.json'),details);write(OUT/'results_partial.json',dict(summary=summary,performance=performance));del model,backbone,pred;torch.cuda.empty_cache()
    assert all(sha(ROOT/z['image'])==z['sha256'] for z in manifest.values());write(OUT/'results.json',dict(summary=summary,performance=performance,new_training=False,production_changed=False));status('complete');state.update(stage='final_development_evaluated',active_job=None,active_pid=None,next='No moretraining afterdevelopment results. Visuallyreview fixedendpoint outputs andwritefinalevidence/limits; pauseexistingheartbeatby18:41:41UTC.',updated_utc=now().isoformat());state['completed'].append('mixed_fixed_endpoints_development144');write(C/'state.json',state);print('COMPLETE',json.dumps(summary),flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write(OUT/'failure.json',dict(traceback=traceback.format_exc(),utc=now().isoformat()));write(OUT/'status.json',dict(stage='failed',pid=os.getpid(),utc=now().isoformat()))
        raise
