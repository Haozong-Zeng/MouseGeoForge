"""Exact native last-unit input cache and isolated-copy gradient preflight."""
import sys,datetime as dt,fcntl,json,os,time,hashlib
os.environ['DLClight']='True'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import numpy as np,torch
from PIL import Image
import albumentations as alb
from run_specialist_roi_20261005 import digest,write
from mousegeoforge.models.highres_fusion_adapter import HighresFusionAdapter,buffer_digest
from mousegeoforge.models.highres_instance_readout import highres_localization_loss,KP
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';OLD=ROOT/'output/bottom_anchor_cycle_20261002';SRC=CYCLE/'highres_preflight_inputs_v1';OUT=CYCLE/'highres_unit_inputs_v1'

def main():
    plan=json.loads((CYCLE/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'));assert dt.datetime.now(dt.timezone.utc)<deadline
    lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);OUT.mkdir(exist_ok=False)
    def status(stage,**kw):write(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),**kw))
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='highres_unit_caching',active_job=OUT.name,active_pid=os.getpid());write(CYCLE/'state.json',state)
    from deeplabcut.pose_estimation_pytorch.models import PoseModel
    torch.set_num_threads(4);records=json.loads((SRC/'inference_manifest.json').read_text());assert len(records)==42
    info=json.loads((OLD/'quadruped_epoch110_review_v1/summary.json').read_text());cp=Path(info['checkpoint']);assert hashlib.sha256(cp.read_bytes()).hexdigest()==info['checkpoint_sha256']
    cfg=json.loads((OLD/'superanimal_quadruped_preparation_v1/model_config.json').read_text());model=PoseModel.build(cfg,pretrained_backbone=False);weights=torch.load(cp,map_location='cpu',weights_only=False);model.load_state_dict(weights['model'],strict=True);del weights;model.cuda().eval().requires_grad_(False);before=digest(model)
    native=model.backbone.model.stage4[-1];adapter=HighresFusionAdapter(native,adapt=True).cuda().eval();adapter_before=digest(adapter.unit);bn_before=buffer_digest(adapter.unit);captured=[]
    def capture(module,args):captured.append([v.detach().clone() for v in args[0]])
    hook=native.register_forward_pre_hook(capture);normalize=alb.Normalize(mean=[.485,.456,.406],std=[.229,.224,.225]);manifest=[];seconds=0.;parity=[]
    for j,r in enumerate(records):
        assert dt.datetime.now(dt.timezone.utc)<deadline
        path=ROOT/r['image'];assert hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256'];rgb=np.asarray(Image.open(path).convert('RGB'));h,w=r['image_hw'];ph,pw=(-h)%32,(-w)%32;x=torch.from_numpy(normalize(image=np.pad(rgb,((0,ph),(0,pw),(0,0))))['image'].copy()).permute(2,0,1)[None].cuda();captured.clear();torch.cuda.synchronize();start=time.monotonic()
        with torch.no_grad():
            _=model(x);assert len(captured)==1;branches=captured[0];assert len(branches)==4 and [v.shape[1] for v in branches]==[32,64,128,256]
            saved=[v.clone() for v in branches];f=adapter.features(branches);old=torch.load(CYCLE/'highres_pre_relu_features_v1'/f'{r["index"]:04d}.pt',weights_only=False,map_location='cpu')['features'];delta=float((f.cpu()-old).abs().max());assert delta==0,delta;assert all(torch.equal(a,b) for a,b in zip(branches,saved));parity.append(delta)
        torch.cuda.synchronize();seconds+=time.monotonic()-start;dest=OUT/f'{r["index"]:04d}.pt';torch.save(dict(branches=[v.cpu() for v in branches],image_hw=[h,w],pad_hw=[ph,pw]),dest);manifest.append(dict(index=r['index'],path=str(dest.relative_to(ROOT)),sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),shapes=[list(v.shape) for v in branches]));status('caching',completed=j+1,total=42)
    hook.remove();assert adapter_before==digest(adapter.unit) and before==digest(model)
    # Integration smoke uses deterministic fake supervision, not any evaluation labels.
    torch.manual_seed(17);first=records[0];saved=torch.load(OUT/f'{first["index"]:04d}.pt',weights_only=False,map_location='cpu');branches=[v.cuda() for v in saved['branches']];frozen_inputs=[v.clone() for v in branches];inf=next(z for z in torch.load(SRC/'inference.pt',weights_only=False,map_location='cpu') if z['index']==first['index']);tokens=inf['tokens'].cuda().float();boxes=inf['boxes'].cuda().float();base=inf['baseline_xy'].cuda().float();hw=saved['image_hw']
    assert adapter(None,None,None,base,None)['xy'] is base;initial=adapter(branches,tokens,boxes,base,hw,enabled=True);assert torch.equal(initial['xy'],base)
    target=(base[:,KP]+.02).clamp(.05,.95);known=torch.ones(6,5,dtype=torch.bool,device='cuda');opt=torch.optim.SGD(adapter.parameters(),lr=.001);gradnorms=[]
    adapter.train();assert all(not m.training for m in adapter.unit.modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm))
    for step in range(3):
        opt.zero_grad(set_to_none=True);out=adapter(branches,tokens,boxes,base,hw,enabled=True);loss=highres_localization_loss(out,target,known,hw);assert torch.isfinite(loss);loss.backward();gg=[p.grad for p in adapter.unit.parameters() if p.grad is not None];assert all(torch.isfinite(g).all() for g in gg);norm=sum(float(g.norm()) for g in gg);gradnorms.append(norm);torch.nn.utils.clip_grad_norm_(adapter.parameters(),1,error_if_nonfinite=True);opt.step()
    assert gradnorms[-1]>0 and adapter_before!=digest(adapter.unit);assert bn_before==buffer_digest(adapter.unit);assert all(torch.equal(a,b) for a,b in zip(branches,frozen_inputs));assert before==digest(model)
    with torch.no_grad():
        out=adapter(branches,tokens,boxes,base,hw,enabled=True)['xy'];p=torch.tensor([5,3,1,4,2,0],device='cuda');permuted=adapter(branches,tokens[p],boxes[p],base[p],hw,enabled=True)['xy'];assert torch.allclose(permuted,out[p],atol=1e-6);non=[k for k in range(23) if k not in KP];assert torch.equal(out[:,non],base[:,non])
    # Smoke-updated copy is deliberately discarded; only pristine native state is exported.
    torch.save(native.cpu().state_dict(),OUT/'native_initial.pt');write(OUT/'manifest.json',manifest);report=dict(images=42,bytes=sum((ROOT/r['path']).stat().st_size for r in manifest),native_pre_relu_parity_max=max(parity),compute_seconds=seconds,original_before=before,original_after=digest(model),bn_buffers_unchanged=True,unit_parameters=sum(p.numel() for p in adapter.unit.parameters()),readout_parameters=sum(p.numel() for p in adapter.readout.parameters()),smoke_unit_gradient_norms=gradnorms,smoke_weights_discarded=True,defaultoff_and_zero_exact=True,query_permutation=True,non_target18_exact=True,cache_inputs_unchanged=True,torch=torch.__version__,checkpoint_sha256=info['checkpoint_sha256']);assert report['original_before']==report['original_after'];write(OUT/'results.json',report);status('complete');state.update(stage='highres_unit_preflight_complete',active_job=None,active_pid=None,next='If parity/gradient/BN/input guards pass, matchedDLCruntime frozen-vs-adapted300 smallTRAIN control only.');state['completed'].append('highres_unit42_cache_gradient_parity');write(CYCLE/'state.json',state);print(json.dumps(report),flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()))
        raise
