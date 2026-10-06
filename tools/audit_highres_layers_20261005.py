"""Frozen stage-wise TRAIN-only activation audit, RGB versus fixed grayscale."""
import sys,datetime as dt,fcntl,json,os,time,hashlib
os.environ['DLClight']='True'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import numpy as np,torch
from PIL import Image
import albumentations as alb
from run_specialist_roi_20261005 import digest,write
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';OLD=ROOT/'output/bottom_anchor_cycle_20261002';SRC=CYCLE/'highres_preflight_inputs_v1';OUT=CYCLE/'highres_layer_audit_v2'

def main():
    plan=json.loads((CYCLE/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'));assert dt.datetime.now(dt.timezone.utc)<deadline
    lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);OUT.mkdir(exist_ok=False);(OUT/'energy').mkdir()
    def status(stage,**kw):write(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),**kw))
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='highres_layer_audit_running',active_job=OUT.name,active_pid=os.getpid());write(CYCLE/'state.json',state)
    from deeplabcut.pose_estimation_pytorch.models import PoseModel
    torch.set_num_threads(4);protocol=json.loads((SRC/'protocol.json').read_text());ids=protocol['indices'][:protocol['fit']];assert len(ids)==24
    records=[r for r in json.loads((SRC/'inference_manifest.json').read_text()) if r['index'] in ids];inf={r['index']:r for r in torch.load(SRC/'inference.pt',map_location='cpu',weights_only=False)}
    info=json.loads((OLD/'quadruped_epoch110_review_v1/summary.json').read_text());cp=Path(info['checkpoint']);assert hashlib.sha256(cp.read_bytes()).hexdigest()==info['checkpoint_sha256']
    cfg=json.loads((OLD/'superanimal_quadruped_preparation_v1/model_config.json').read_text());model=PoseModel.build(cfg,pretrained_backbone=False);weights=torch.load(cp,map_location='cpu',weights_only=False);model.load_state_dict(weights['model'],strict=True);del weights;model.cuda().eval().requires_grad_(False);before=digest(model);model.output_features=True;net=model.backbone.model
    captured={};hooks=[]
    def hook(name):
        def save(module,args,out):captured[name]=(out[0] if isinstance(out,list) else out).detach().clone()
        return save
    for name in ['act2','layer1','stage2','stage3','stage4']:hooks.append(getattr(net,name).register_forward_hook(hook(name)))
    for k,mod in enumerate(net.stage4):hooks.append(mod.register_forward_hook(hook(f'stage4_unit{k}')))
    def pre(module,args):
        if args[0].shape[1]==32:captured['stage4_pre_relu']=args[0].detach().clone()
    hooks.append(net.stage4[-1].fuse_act.register_forward_pre_hook(pre))
    normalize=alb.Normalize(mean=[.485,.456,.406],std=[.229,.224,.225]);stats=[];seconds=0.;max_parity=0.;shapes={}
    try:
        for j,r in enumerate(records):
            assert dt.datetime.now(dt.timezone.utc)<deadline
            path=ROOT/r['image'];assert hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256'];im=Image.open(path).convert('RGB');h,w=r['image_hw'];assert im.size==(w,h);ph,pw=(-h)%32,(-w)%32;boxes=inf[r['index']]['boxes'].cuda()
            for variant in ['rgb','gray']:
                rgb=np.asarray(im if variant=='rgb' else im.convert('L').convert('RGB'));padded=np.pad(rgb,((0,ph),(0,pw),(0,0)));x=torch.from_numpy(normalize(image=padded)['image'].copy()).permute(2,0,1)[None].cuda();captured.clear();torch.cuda.synchronize();start=time.monotonic()
                with torch.inference_mode():raw=model(x)
                torch.cuda.synchronize();seconds+=time.monotonic()-start;f=raw['backbone']['features'];assert torch.equal(f,captured['stage4'])
                if variant=='rgb':
                    old=torch.load(CYCLE/'highres_preflight_features_v1'/f'{r["index"]:04d}.pt',weights_only=False,map_location='cpu')['features'];delta=float((f[0].cpu()-old).abs().max());assert delta==0;max_parity=max(max_parity,delta)
                energies={}
                for name,out in captured.items():
                    z=out[0];c,fh,fw=z.shape;assert (fh,fw)==((h+ph)//4,(w+pw)//4);shapes[name]=c
                    yy,xx=torch.meshgrid((torch.arange(fh,device='cuda')+.5)*4,(torch.arange(fw,device='cuda')+.5)*4,indexing='ij');grid=torch.stack([xx/w,yy/h],-1);valid=(xx<w)&(yy<h);energy=z.abs().amax(0);energies[name]=energy.cpu().numpy();boxstats=[]
                    for q,b in enumerate(boxes):
                        mask=((grid>=b[:2]-b[2:]/2)&(grid<=b[:2]+b[2:]/2)).all(-1)&valid
                        if not mask.any():boxstats.append(dict(query=q,cells=0));continue
                        values=z[:,mask];n=int(mask.sum());zero=int((values==0).all(0).sum());nonpositive=int((values<=0).all(0).sum());boxstats.append(dict(query=q,cells=n,allchannels_zero_cells=zero,allchannels_nonpositive_cells=nonpositive,allchannels_zero_fraction=zero/n,allchannels_nonpositive_fraction=nonpositive/n,absmax=float(values.abs().max()),absmean=float(values.abs().mean()),rms=float(values.square().mean().sqrt()),spatialstd=float(values.std(-1,unbiased=False).mean())))
                    stats.append(dict(index=r['index'],variant=variant,layer=name,shape=list(z.shape),boxes=boxstats))
                np.savez_compressed(OUT/'energy'/f'{r["index"]:04d}_{variant}.npz',**energies);status('audit',completed=j+1,total=24,variant=variant)
    finally:
        for h0 in hooks:h0.remove()
    after=digest(model);assert before==after;write(OUT/'statistics.json',stats);report=dict(images=24,variants=['rgb','gray'],forward_calls=48,channels=shapes,all_stride=4,final_rgb_cache_max_abs=max_parity,before=before,after=after,compute_seconds=seconds,gt_read=False,normalization='PIL L replicated toRGB as one fixed variant before sameImageNetnormalize;pad32; no resizing',no_bn_updates=True,no_training=True);write(OUT/'results.json',report);status('complete');state.update(stage='highres_layer_audit_complete',active_job=None,active_pid=None,next='Posthoc stagewise stats grouping/feature visual review, no diagnosis of BN cause without evidence.');state['completed'].append('highres_layers_rgb_gray24fit');write(CYCLE/'state.json',state);print(json.dumps(report),flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()))
        raise
