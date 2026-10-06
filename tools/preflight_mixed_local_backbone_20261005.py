"""Actual-RGB native parity and discarded full-backbone gradient smoke."""
import os,sys,json,datetime as dt,fcntl,hashlib,time
os.environ['DLClight']='True'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import torch,numpy as np,albumentations as alb
from PIL import Image
from mousegeoforge.models.mixed_local_evidence import MixedLocalEvidence
from mousegeoforge.models.local_point_evidence_probe import point_evidence_loss
from run_specialist_roi_20261005 import digest
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';INPUT=CYCLE/'mixed_local_rgb_inputs_v1';PREP=ROOT/'output/bottom_anchor_cycle_20261002/superanimal_quadruped_preparation_v1';OUT=CYCLE/'mixed_local_preflight_v1'
def read(p):return torch.load(p,weights_only=False,map_location='cpu')
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def build_backbone():
    from deeplabcut.pose_estimation_pytorch.models import PoseModel
    info=json.loads((PREP/'preflight.json').read_text());path=PREP/'superanimal_quadruped_hrnet_w32.pt';assert sha(path)==info['sha256'];cfg=json.loads((PREP/'model_config.json').read_text());pose=PoseModel.build(cfg,pretrained_backbone=False);backbone=pose.backbone;del pose
    source={k[len('backbone.'):]:v for k,v in read(path)['model'].items() if k.startswith('backbone.')};assert len(source)==1830;backbone.load_state_dict(source,strict=True);return backbone,info
def load_rgb(row):
    path=ROOT/row['image'];assert sha(path)==row['sha256'];rgb=np.asarray(Image.open(path).convert('RGB'));h,w=row['image_hw'];assert rgb.shape==(h,w,3);padded=np.pad(rgb,((0,(-h)%32),(0,(-w)%32),(0,0)));normalized=alb.Normalize(mean=[.485,.456,.406],std=[.229,.224,.225])(image=padded)['image'];return torch.from_numpy(normalized.copy()).permute(2,0,1)[None]
def main():
    plan=json.loads((CYCLE/'plan.json').read_text());assert dt.datetime.now(dt.timezone.utc)<dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'));lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);OUT.mkdir(exist_ok=False);torch.set_num_threads(4);torch.manual_seed(17)
    rows={z['index']:z for z in json.loads((INPUT/'inference_manifest.json').read_text())};ev={z['index']:z for z in read(INPUT/'evaluation_only.pt')};old=json.loads((CYCLE/'official_quadruped_features_v2/manifest.json').read_text());backbone,info=build_backbone();model=MixedLocalEvidence(backbone,adapt_backbone=True).cuda();model.train();initial=digest(model.backbone);buffers={k:v.clone() for k,v in model.backbone.named_buffers()};deltas=[];seconds=0.
    torch.save({k:v.detach().cpu().clone() for k,v in model.readout.state_dict().items()},OUT/'initial_readout.pt')
    for z in old:
        x=load_rgb(rows[z['index']]).cuda();torch.cuda.synchronize();start=time.monotonic()
        with torch.no_grad():out=model(x,rows[z['index']]['image_hw'])
        torch.cuda.synchronize();seconds+=time.monotonic()-start;reference=read(ROOT/z['path'])['features'];delta=float((out['features'].cpu()-reference).abs().max());assert delta==0;deltas.append(delta)
    smoke=[next(i for i,r in ev.items() if r['metadata']['stage']=='fit' and r['metadata']['kind']==kind) for kind in ['real','single','dual']];optimizer=torch.optim.AdamW([dict(params=model.backbone.parameters(),lr=1e-5),dict(params=model.readout.parameters(),lr=1e-3)],weight_decay=1e-4);gradients=[];timings=[];peak=0
    for step in range(3):
        optimizer.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic()
        for i in smoke:
            r=ev[i];out=model(load_rgb(rows[i]).cuda(),rows[i]['image_hw']);loss=point_evidence_loss(out,r['target'].cuda(),r['eligible'].cuda(),rows[i]['image_hw'])/3;assert torch.isfinite(loss);loss.backward()
        norm=sum(float(p.grad.square().sum()) for p in model.backbone.parameters() if p.grad is not None)**.5;assert np.isfinite(norm);gradients.append(norm);torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);optimizer.step();torch.cuda.synchronize();timings.append(time.monotonic()-start);peak=max(peak,torch.cuda.max_memory_allocated())
    assert gradients[1]>0 and gradients[2]>0;assert all(torch.equal(v,buffers[k]) for k,v in model.backbone.named_buffers());assert sha(PREP/'superanimal_quadruped_hrnet_w32.pt')==info['sha256']
    # No smoke weight checkpoint is saved. Reloading verifies pristine source restore.
    del optimizer,model,backbone;torch.cuda.empty_cache();pristine,_=build_backbone();assert digest(pristine)==initial
    result=dict(images_native_parity=42,max_feature_difference=max(deltas),initial_backbone_digest=initial,source_sha256=info['sha256'],native_seconds=seconds,smoke_indices=smoke,smoke_gradient_norms=gradients,smoke_step_seconds=timings,peak_gpu_bytes=peak,bn_buffers_unchanged=True,smoke_weights_discarded=True,source_reload_exact=True,trainable_backbone_parameters=sum(p.numel() for p in pristine.parameters()),readout_parameters=19973,torch_version=torch.__version__,unit_tests=16)
    write(OUT/'results.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
