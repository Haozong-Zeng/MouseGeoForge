"""Local official SuperAnimal backbone provenance/control; no decoder mapping."""
import sys,datetime as dt,fcntl,json,os,hashlib,time
os.environ['DLClight']='True'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import numpy as np,torch
from PIL import Image
import albumentations as alb
from run_specialist_roi_20261005 import digest,write
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';PREP=ROOT/'output/bottom_anchor_cycle_20261002/superanimal_quadruped_preparation_v1';INPUT=CYCLE/'highres_preflight_inputs_v1';OUT=CYCLE/'official_quadruped_features_v2'
def main():
    plan=json.loads((CYCLE/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'));assert dt.datetime.now(dt.timezone.utc)<deadline
    lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);OUT.mkdir(exist_ok=False);(OUT/'energy').mkdir()
    def status(stage,**kw):write(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),**kw))
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='official_quadruped_feature_audit',active_job=OUT.name,active_pid=os.getpid());write(CYCLE/'state.json',state)
    from deeplabcut.pose_estimation_pytorch.models import PoseModel
    torch.set_num_threads(4);torch.manual_seed(17);info=json.loads((PREP/'preflight.json').read_text());path=PREP/'superanimal_quadruped_hrnet_w32.pt';assert path.stat().st_size==info['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==info['sha256'];cp=torch.load(path,map_location='cpu',weights_only=True);source={k[len('backbone.'):]:v for k,v in cp['model'].items() if k.startswith('backbone.')};assert len(source)==1830
    cfg=json.loads((PREP/'model_config.json').read_text());model=PoseModel.build(cfg,pretrained_backbone=False);backbone=model.backbone;del model;backbone.load_state_dict(source,strict=True);assert all(torch.equal(backbone.state_dict()[k],v) for k,v in source.items());del source,cp
    backbone.cuda();backbone.eval();backbone.requires_grad_(False);before=digest(backbone);records=json.loads((INPUT/'inference_manifest.json').read_text());inference={r['index']:r for r in torch.load(INPUT/'inference.pt',map_location='cpu',weights_only=False)};assert len(records)==42
    captured={}
    def pre(module,args):
        if args[0].shape[1]==32:captured['pre_relu']=args[0].detach().clone()[0]
    def stage(module,args,out):captured['stage3']=out[0].detach().clone()[0]
    hooks=[backbone.model.stage4[-1].fuse_act.register_forward_pre_hook(pre),backbone.model.stage3.register_forward_hook(stage)];normalize=alb.Normalize(mean=[.485,.456,.406],std=[.229,.224,.225]);manifest=[];stats=[];seconds=0.
    try:
        for j,r in enumerate(records):
            assert dt.datetime.now(dt.timezone.utc)<deadline
            path0=ROOT/r['image'];assert hashlib.sha256(path0.read_bytes()).hexdigest()==r['sha256'];rgb=np.asarray(Image.open(path0).convert('RGB'));h,w=r['image_hw'];ph,pw=(-h)%32,(-w)%32;x=torch.from_numpy(normalize(image=np.pad(rgb,((0,ph),(0,pw),(0,0))))['image'].copy()).permute(2,0,1)[None].cuda();captured.clear();torch.cuda.synchronize();start=time.monotonic()
            with torch.inference_mode():f=backbone(x)[0]
            torch.cuda.synchronize();seconds+=time.monotonic()-start;assert f.shape==(32,(h+ph)//4,(w+pw)//4) and torch.equal(captured['pre_relu'].relu(),f);captured['post_relu']=f
            dest=OUT/f'{r["index"]:04d}.pt';torch.save(dict(features=f.cpu().clone(),image_hw=[h,w],pad_hw=[ph,pw]),dest);assert torch.equal(torch.load(dest,map_location='cpu',weights_only=False)['features'],f.cpu());manifest.append(dict(index=r['index'],path=str(dest.relative_to(ROOT)),sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),shape=list(f.shape),source_sha256=r['sha256']))
            energies={};boxes=inference[r['index']]['boxes'].cuda()
            for name,z in captured.items():
                c,fh,fw=z.shape;assert c==32;yy,xx=torch.meshgrid((torch.arange(fh,device='cuda')+.5)*4,(torch.arange(fw,device='cuda')+.5)*4,indexing='ij');grid=torch.stack([xx/w,yy/h],-1);valid=(xx<w)&(yy<h);bs=[];energies[name]=z.abs().amax(0).cpu().numpy()
                for q,b in enumerate(boxes):
                    mask=((grid>=b[:2]-b[2:]/2)&(grid<=b[:2]+b[2:]/2)).all(-1)&valid;n=int(mask.sum())
                    if not n:bs.append(dict(query=q,cells=0));continue
                    v=z[:,mask];bs.append(dict(query=q,cells=n,zero_cells=int((v==0).all(0).sum()),nonpositive_cells=int((v<=0).all(0).sum()),rms=float(v.square().mean().sqrt()),spatialstd=float(v.std(-1,unbiased=False).mean())))
                stats.append(dict(index=r['index'],layer=name,boxes=bs))
            np.savez_compressed(OUT/'energy'/f'{r["index"]:04d}.npz',**energies);status('audit',completed=j+1,total=42)
    finally:
        for hook in hooks:hook.remove()
    after=digest(backbone);assert before==after;write(OUT/'manifest.json',manifest);write(OUT/'statistics.json',stats);result=dict(images=42,bytes=sum((ROOT/r['path']).stat().st_size for r in manifest),backbone_tensors_verified=1830,checkpoint_sha256=info['sha256'],before=before,after=after,compute_seconds=seconds,cache_fp32_exact=True,relu_replay_exact=True,head_used=False,no_gt_read=True,no_download=True,provenance=dict(repo=info['repo'],revision=info['revision']));write(OUT/'results.json',result);status('complete');state.update(stage='official_quadruped_feature_audit_complete',active_job=None,active_pid=None,next='Compare genericpretrained vsreal-onlyfinetuned featureson fixed42 afterinference; no accuracy claim from activations alone.');state['completed'].append('official_quadruped_backbone42_frozen_audit');write(CYCLE/'state.json',state);print(json.dumps(result),flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()))
        raise
