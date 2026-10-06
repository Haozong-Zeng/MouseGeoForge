"""42-image exact FP32 shared HRNet feature cache; inference does not read GT."""
import sys,datetime as dt,fcntl,json,os,time,hashlib
os.environ['DLClight']='True'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import numpy as np,torch
from PIL import Image
import albumentations as alb
from run_specialist_roi_20261005 import digest,write
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';OLD=ROOT/'output/bottom_anchor_cycle_20261002';SRC=CYCLE/'highres_preflight_inputs_v1';OUT=CYCLE/'highres_preflight_features_v1'

def main():
    global OUT
    pre_relu='--pre-relu' in sys.argv
    if pre_relu:OUT=CYCLE/'highres_pre_relu_features_v1'
    plan=json.loads((CYCLE/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'));assert dt.datetime.now(dt.timezone.utc)<deadline
    lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);OUT.mkdir(exist_ok=False)
    def status(stage,**kw):write(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),**kw))
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='highres_preflight_caching',active_job=OUT.name,active_pid=os.getpid());write(CYCLE/'state.json',state)
    from deeplabcut.pose_estimation_pytorch.models import PoseModel
    torch.set_num_threads(4);records=json.loads((SRC/'inference_manifest.json').read_text());assert len(records)==42
    info=json.loads((OLD/'quadruped_epoch110_review_v1/summary.json').read_text());cp=Path(info['checkpoint']);assert hashlib.sha256(cp.read_bytes()).hexdigest()==info['checkpoint_sha256']
    cfg=json.loads((OLD/'superanimal_quadruped_preparation_v1/model_config.json').read_text());model=PoseModel.build(cfg,pretrained_backbone=False);weights=torch.load(cp,map_location='cpu',weights_only=False);model.load_state_dict(weights['model'],strict=True);del weights;model.cuda().eval().requires_grad_(False);before=digest(model);assert model.neck is None;model.output_features=True
    captured=[];hook=None
    if pre_relu:
        def save_pre(module,args):
            if args[0].shape[1]==32:captured.append(args[0].detach().clone())
        hook=model.backbone.model.stage4[-1].fuse_act.register_forward_pre_hook(save_pre)
    normalize=alb.Normalize(mean=[.485,.456,.406],std=[.229,.224,.225]);manifest=[];seconds=0.;parity=0.
    for j,r in enumerate(records):
        assert dt.datetime.now(dt.timezone.utc)<deadline
        path=ROOT/r['image'];assert hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256'];rgb=np.asarray(Image.open(path).convert('RGB'));h,w=r['image_hw'];assert rgb.shape==(h,w,3);ph,pw=(-h)%32,(-w)%32;rgb=np.pad(rgb,((0,ph),(0,pw),(0,0)))
        x=torch.from_numpy(normalize(image=rgb)['image'].copy()).permute(2,0,1)[None].cuda();torch.cuda.synchronize();start=time.monotonic()
        with torch.inference_mode():
            captured.clear();raw=model(x);f=raw['backbone']['features'];assert tuple(f.shape)==(1,32,(h+ph)//4,(w+pw)//4)
            direct=model.heads['bodypart'](f)['heatmap'];delta=float((direct-raw['bodypart']['heatmap']).abs().max());assert delta==0;parity=max(parity,delta)
            if pre_relu:
                assert len(captured)==1 and torch.equal(captured[0].relu(),f);f=captured[0]
        torch.cuda.synchronize();seconds+=time.monotonic()-start
        dest=OUT/f'{r["index"]:04d}.pt';torch.save(dict(features=f[0].cpu().clone(),image_hw=[h,w],pad_hw=[ph,pw]),dest);restored=torch.load(dest,weights_only=False,map_location='cpu');assert torch.equal(restored['features'],f[0].cpu())
        manifest.append(dict(index=r['index'],path=str(dest.relative_to(ROOT)),sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),shape=list(f.shape[1:]),source_sha256=r['sha256']));status('caching',completed=j+1,total=42)
    if hook is not None:hook.remove()
    after=digest(model);assert before==after;write(OUT/'manifest.json',manifest);write(OUT/'results.json',dict(images=42,bytes=sum((ROOT/r['path']).stat().st_size for r in manifest),compute_seconds=seconds,head_replay_max_abs=parity,cache_fp32_exact=True,before=before,after=after,checkpoint_sha256=info['checkpoint_sha256'],no_gt_read=True,pre_relu=pre_relu,relu_replay_exact=pre_relu));status('complete');state.update(stage='highres_preflight_cached',active_job=None,active_pid=None);state['completed'].append('highres_pre_relu42_parity' if pre_relu else 'highres_feature42_parity');write(CYCLE/'state.json',state);print(json.dumps(json.loads((OUT/'results.json').read_text())),flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()))
        raise
