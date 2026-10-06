"""Frozen all-query specialist heatmap ROI diagnostic; inference inputs have no GT."""
import datetime as dt,fcntl,hashlib,json,os,sys,time
os.environ['DLClight']='True'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from PIL import Image
import albumentations as alb
from mousegeoforge.models.specialist_roi_readout import specialist_roi_readout,CORE_INDICES
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';OLD=ROOT/'output/bottom_anchor_cycle_20261002';SRC=CYCLE/'specialist_roi_inputs_v1';OUT=CYCLE/'specialist_roi_frozen_v1'
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def digest(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def main():
    plan=json.loads((CYCLE/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'));assert dt.datetime.now(dt.timezone.utc)<deadline
    lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);OUT.mkdir(exist_ok=False);(OUT/'heatmaps').mkdir()
    def status(stage,**kw):write(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),**kw))
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='specialist_roi_frozen_running',active_job=OUT.name,active_pid=os.getpid());write(CYCLE/'state.json',state);status('loading')
    from deeplabcut.pose_estimation_pytorch.models import PoseModel
    torch.set_num_threads(4);records=json.loads((SRC/'inference_manifest.json').read_text());assert len(records)==144
    forbidden={'gt','valid','kind','session','expected_instances','metadata'};assert all(not forbidden.intersection(r) for r in records)
    info=json.loads((OLD/'quadruped_epoch110_review_v1/summary.json').read_text());cp=Path(info['checkpoint']);assert hashlib.sha256(cp.read_bytes()).hexdigest()==info['checkpoint_sha256']
    cfg=json.loads((OLD/'superanimal_quadruped_preparation_v1/model_config.json').read_text());model=PoseModel.build(cfg,pretrained_backbone=False);weights=torch.load(cp,map_location='cpu',weights_only=False);model.load_state_dict(weights['model'],strict=True);del weights;model.cuda().eval().requires_grad_(False);before=digest(model)
    normalize=alb.Normalize(mean=[.485,.456,.406],std=[.229,.224,.225]);prior=np.load(CYCLE/'teacher_current_rgb_replay_v1/predictions.npz');gold={int(i):p for i,p in zip(prior['indices'],prior['current'])};parity=[];native_errors=[];outputs=[];supports=[];scores=[];globals_=[];indices=[];times=[];peak=0
    for j,r in enumerate(records):
        assert dt.datetime.now(dt.timezone.utc)<deadline
        path=ROOT/r['image'];assert hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256'];rgb=np.asarray(Image.open(path).convert('RGB'));h,w=r['image_hw'];assert rgb.shape==(h,w,3);ph,pw=r['pad_hw'];padded=np.pad(rgb,((0,ph),(0,pw),(0,0)),constant_values=0);assert padded.shape[0]%32==padded.shape[1]%32==0
        x=torch.from_numpy(normalize(image=padded)['image'].copy()).permute(2,0,1)[None].cuda();box=torch.tensor(r['boxes'],device='cuda');base=torch.tensor(r['baseline_xy'],device='cuda');assert box.shape==(6,4) and base.shape==(6,23,2)
        torch.cuda.synchronize();start=time.monotonic();torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            raw=model(x);hm=raw['bodypart']['heatmap'][0];assert tuple(hm.shape)==(7,(h+ph)//4,(w+pw)//4);native=model.get_predictions(raw)['bodypart']['poses'].reshape(7,3)
            chosen=hm[[0,3,4,5,6]];out=specialist_roi_readout(chosen,box,base,(h,w),enabled=True)
            off=specialist_roi_readout(chosen,box,base,(h,w));assert torch.equal(off['xy'],base)
            full=specialist_roi_readout(chosen,torch.tensor([[.5,.5,1.,1.]],device='cuda'),base[:1],(h,w),enabled=True)
            non=[k for k in range(23) if k not in CORE_INDICES];assert torch.equal(out['xy'][:,non],base[:,non]);assert torch.isfinite(out['xy']).all()
            if ph==pw==0:
                delta=float((full['xy'][0,CORE_INDICES]*torch.tensor([w,h],device='cuda')-native[[0,3,4,5,6],:2]).abs().max());assert delta<1e-4;native_errors.append(delta)
            if r['index'] in gold:
                delta=float(np.max(np.abs(native.cpu().numpy()[:,:2]-gold[r['index']][:,:2])));assert delta<1e-4;parity.append(dict(index=r['index'],max_px=delta))
        torch.cuda.synchronize();times.append(time.monotonic()-start);peak=max(peak,torch.cuda.max_memory_allocated());outputs.append(out['xy'].cpu().numpy());supports.append(out['supported'].cpu().numpy());scores.append(out['scores'].cpu().numpy());globals_.append(full['xy'][0,CORE_INDICES].cpu().numpy());indices.append(r['index'])
        np.savez_compressed(OUT/'heatmaps'/f'{r["index"]:04d}.npz',heatmaps=chosen.cpu().numpy(),image_hw=np.array([h,w]),pad_hw=np.array([ph,pw]));status('inference',completed=j+1,total=144)
    after=digest(model);assert before==after and len(parity)==48
    np.savez_compressed(OUT/'predictions.npz',indices=np.array(indices),xy=np.stack(outputs),supported=np.stack(supports),raw_scores=np.stack(scores),global_valid_xy=np.stack(globals_))
    report=dict(images=144,checkpoint_sha256=info['checkpoint_sha256'],model_before_sha256=before,model_after_sha256=after,real48_replay_max_px=max(p['max_px'] for p in parity),native_predictor_parity_max_px=max(native_errors),compute_seconds=sum(times),median_seconds=float(np.median(times)),peak_gpu_bytes=peak,defaultoff_exact_all144=True,non_target_exact_all144=True,empty_query_point_fallbacks=int((~np.stack(supports)).sum()),inference='All6predictedROIs hardmasknativeHRNetstride4centers,5points only, noconfidencefilter/nocount/datasetkind/GTinputs. Rightbottompad32withoutresize, paddedcells excluded. Untrainedreadout, no learnedgate ornewlabels.',tests=7,production_changed=False)
    write(OUT/'results.json',report);status('complete');state.update(stage='specialist_roi_frozen_complete',active_job=None,active_pid=None,next='Evaluate fixed144cohort conditionalpoints; real/single/dual, rawallqueries, wronganimal/duplicateROI, heatmaps. No datasetkindgate, no promotion.');state['completed'].append('specialist_roi144_frozen_inference');write(CYCLE/'state.json',state);print(json.dumps(report),flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()))
        raise
