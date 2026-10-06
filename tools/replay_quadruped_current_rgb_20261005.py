"""Frozen HRNet replay on current real images; no old-cycle state or deadlines."""
import datetime as dt,fcntl,hashlib,json,os,time
os.environ['DLClight']='True'
from pathlib import Path
import numpy as np,torch
from PIL import Image
import albumentations as alb
ROOT=Path(__file__).resolve().parents[1];CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';OLD=ROOT/'output/bottom_anchor_cycle_20261002';OUT=CYCLE/'teacher_current_rgb_replay_v1'
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def digest(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def main():
    plan=json.loads((CYCLE/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'));assert dt.datetime.now(dt.timezone.utc)<deadline
    lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);OUT.mkdir(exist_ok=False)
    def status(stage,**kw):write(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),**kw))
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='teacher_current_rgb_replay_running',active_job=OUT.name,active_pid=os.getpid());write(CYCLE/'state.json',state);status('loading')
    from deeplabcut.pose_estimation_pytorch.models import PoseModel
    torch.set_num_threads(4);records=json.loads((CYCLE/'teacher_current_cohort_cache_audit_v3/image_parity.json').read_text());assert len(records)==48 and all(not r['teacher_was_train'] for r in records)
    info=json.loads((OLD/'quadruped_epoch110_review_v1/summary.json').read_text());cp=Path(info['checkpoint']);assert hashlib.sha256(cp.read_bytes()).hexdigest()==info['checkpoint_sha256']
    cfg=json.loads((OLD/'superanimal_quadruped_preparation_v1/model_config.json').read_text());model=PoseModel.build(cfg,pretrained_backbone=False);weights=torch.load(cp,map_location='cpu',weights_only=False);assert weights['metadata']['epoch']==110;model.load_state_dict(weights['model'],strict=True);del weights
    model.cuda().eval().requires_grad_(False);before=digest(model);normalize=alb.Normalize(mean=[.485,.456,.406],std=[.229,.224,.225]);cached=np.load(OLD/'quadruped_epoch110_review_v1/predictions_source_pixels.npz');dm=json.loads((OLD/'superanimal_quadruped_hrnet_v1/dataset_manifest.json').read_text());A=np.array(dm['affine_source_to_input']);allpred=[];oldpred=[];indices=[];parity=[];seconds=0.
    @torch.inference_mode()
    def infer(path):
        nonlocal seconds
        assert dt.datetime.now(dt.timezone.utc)<deadline
        rgb=np.asarray(Image.open(path).convert('RGB'));assert rgb.shape==(768,704,3)
        x=torch.from_numpy(normalize(image=rgb)['image'].copy()).permute(2,0,1)[None].cuda();torch.cuda.synchronize();start=time.monotonic();pred=model.get_predictions(model(x))['bodypart']['poses'].cpu().numpy().reshape(7,3);torch.cuda.synchronize();seconds+=time.monotonic()-start;assert np.isfinite(pred).all();return pred
    for n,r in enumerate(records):
        path=ROOT/r['current_image'];assert hashlib.sha256(path.read_bytes()).hexdigest()==r['current_image_sha256'];p=infer(path);allpred.append(p);indices.append(r['index'])
        old=np.full((7,3),np.nan)
        if r['old_cache_index'] is not None:
            old=infer(ROOT/r['entry']['derived']);j=r['old_cache_index'];expected=cached['uv'][j]*np.diag(A)[:2]+A[:2,2];d=np.linalg.norm(old[:,:2]-expected,axis=-1);current_delta=np.linalg.norm(old[:,:2]-p[:,:2],axis=-1)
            parity.append(dict(index=r['index'],old_cache_max_px=float(d.max()),old_cache_nosefeet_max_px=float(d[[0,3,4,5,6]].max()),interpolation_max_px=float(current_delta.max()),interpolation_nosefeet_max_px=float(current_delta[[0,3,4,5,6]].max())))
        oldpred.append(old);status('inference',completed=n+1,total=48)
    after=digest(model);assert before==after;np.savez_compressed(OUT/'predictions.npz',indices=np.array(indices),current=np.stack(allpred),old_input=np.stack(oldpred),ids=np.array(dm['ids_1based']))
    report=dict(images=48,old_replay_images=len(parity),checkpoint=str(cp),checkpoint_sha256=info['checkpoint_sha256'],before_sha256=before,after_sha256=after,normalization='Native Albumentations ImageNet, single-image FP32 GPU, model.eval, no updates/crops/detector.',compute_seconds=seconds,torch=torch.__version__,cuda=torch.version.cuda,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,parity=parity,labels_updated=False,old_cycle_state_changed=False)
    write(OUT/'results.json',report);status('complete');state.update(stage='teacher_current_rgb_replay_complete',active_job=None,active_pid=None,next='Run audit_teacher_current_cohort_20261005.py --fresh and inspect same-currentRGB/currentlabels outputs; compare full48 and old41 separately, no architecture-only or multi-mouse claims.');state['completed'].append('teacher_current_rgb48_frozen_replay');write(CYCLE/'state.json',state);print(json.dumps(report),flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()))
        raise
