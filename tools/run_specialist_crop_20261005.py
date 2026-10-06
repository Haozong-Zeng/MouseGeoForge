"""One prespecified, frozen predicted-crop control; no GT read during inference."""
import sys,datetime as dt,fcntl,hashlib,json,os,time
os.environ['DLClight']='True'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import numpy as np,torch
from PIL import Image
import albumentations as alb
from run_specialist_roi_20261005 import digest,write
from mousegeoforge.models.specialist_crop import make_specialist_crop,decode_specialist_crop,CORE_INDICES
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';OLD=ROOT/'output/bottom_anchor_cycle_20261002';SRC=CYCLE/'specialist_roi_inputs_v1';OUT=CYCLE/'specialist_crop_frozen_v1'

def main():
    plan=json.loads((CYCLE/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'));assert dt.datetime.now(dt.timezone.utc)<deadline
    lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);OUT.mkdir(exist_ok=False)
    def status(stage,**kw):write(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),**kw))
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='specialist_crop_running',active_job=OUT.name,active_pid=os.getpid());write(CYCLE/'state.json',state);status('loading')
    from deeplabcut.pose_estimation_pytorch.models import PoseModel
    torch.set_num_threads(4);records=json.loads((SRC/'inference_manifest.json').read_text());assert len(records)==144
    assert all(not {'gt','valid','kind','session','expected_instances','metadata'}.intersection(r) for r in records)
    info=json.loads((OLD/'quadruped_epoch110_review_v1/summary.json').read_text());cp=Path(info['checkpoint']);assert hashlib.sha256(cp.read_bytes()).hexdigest()==info['checkpoint_sha256']
    cfg=json.loads((OLD/'superanimal_quadruped_preparation_v1/model_config.json').read_text());model=PoseModel.build(cfg,pretrained_backbone=False);weights=torch.load(cp,map_location='cpu',weights_only=False);model.load_state_dict(weights['model'],strict=True);del weights;model.cuda().eval().requires_grad_(False);before=digest(model)
    normalize=alb.Normalize(mean=[.485,.456,.406],std=[.229,.224,.225]);outputs=[];geometry=[];times=[];fallbacks=0;peak=0;indices=[]
    for j,r in enumerate(records):
        assert dt.datetime.now(dt.timezone.utc)<deadline
        path=ROOT/r['image'];assert hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256'];im=Image.open(path).convert('RGB');h,w=r['image_hw'];assert im.size==(w,h)
        base=torch.tensor(r['baseline_xy'],device='cuda');pred=base.clone();assert base.shape==(6,23,2)
        for q,box in enumerate(r['boxes']):
            cropped=make_specialist_crop(im,box,expansion=1.25,canvas_hw=(768,704))
            if cropped is None:fallbacks+=1;geometry.append(dict(index=r['index'],query=q,invalid=True));continue
            canvas,meta=cropped;geometry.append(dict(index=r['index'],query=q,**meta))
            x=torch.from_numpy(normalize(image=np.asarray(canvas))['image'].copy()).permute(2,0,1)[None].cuda();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic()
            with torch.inference_mode():
                hm=model(x)['bodypart']['heatmap'][0,[0,3,4,5,6]];assert hm.shape==(5,192,176)
                pred[q]=decode_specialist_crop(hm,base[q],meta,enabled=True)
                assert torch.equal(decode_specialist_crop(hm,base[q],meta),base[q])
            torch.cuda.synchronize();times.append(time.monotonic()-start);peak=max(peak,torch.cuda.max_memory_allocated())
        non=[k for k in range(23) if k not in CORE_INDICES];assert torch.equal(pred[:,non],base[:,non]);assert torch.isfinite(pred).all()
        outputs.append(pred.cpu().numpy());indices.append(r['index']);status('inference',completed=j+1,total=144,forward_calls=len(times))
    after=digest(model);assert before==after
    np.savez_compressed(OUT/'predictions.npz',indices=np.array(indices),xy=np.stack(outputs));write(OUT/'geometry.json',geometry)
    result=dict(images=144,queries=864,forward_calls=len(times),invalid_crop_fallbacks=fallbacks,expansion=1.25,canvas_hw=[768,704],interpolation='PIL Bicubic; round integer resized shape; black out-of-source and letterbox padding excluded from decoding',checkpoint_sha256=info['checkpoint_sha256'],model_before_sha256=before,model_after_sha256=after,compute_seconds=sum(times),peak_gpu_bytes=peak,tests=8,defaultoff_exact=True,other18_exact=True,production_changed=False,gt_used_in_forward=False)
    write(OUT/'results.json',result);status('complete');state.update(stage='specialist_crop_complete',active_job=None,active_pid=None,next='Evaluate frozen crop vs baseline/fullframeROI; all144 samequalifiedGT afterforward. Inspect rawRGB. No promotion or margin tuning.');state['completed'].append('specialist_crop144_frozen_inference');write(CYCLE/'state.json',state);print(json.dumps(result),flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()))
        raise
