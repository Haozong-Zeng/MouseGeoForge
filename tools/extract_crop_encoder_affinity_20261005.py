"""Encode predicted RGB/gray crops with current frozen Omega image backbone."""
import datetime as dt,fcntl,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from PIL import Image
from vggt_omega.models.aggregator import _build_patch_embed
from mousegeoforge.predicted_rgb_crops import predicted_rgb_crops,MEAN,STD
from run_surface_curve_pilot_20261003 import load_item
from run_depth_pose_ablation import fingerprint,write_json
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';OUT=CYCLE/'predicted_crop_encoder_v1'
SOURCES=dict(train='dual_train_allview_affinity_context_v1',validation='dual_allscene_association_audit_v1')


@torch.no_grad()
def main():
    plan=json.loads((CYCLE/'plan.json').read_text());cutoff=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'))
    assert dt.datetime.now(dt.timezone.utc)<cutoff
    lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not OUT.exists();OUT.mkdir();(OUT/'crops').mkdir();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='predicted_crop_encoder_running',active_job=OUT.name,active_pid=os.getpid());write_json(CYCLE/'state.json',state)
    checkpoint=ROOT/'vggt-omega/vggt_omega_1b_512.pt';sha=fingerprint(checkpoint);assert sha=='c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934'
    write_json(OUT/'protocol.json',dict(checkpoint_sha256=sha,size=224,scope='Current Omega image-backbone ONLY; no new re-ID model/training. Exact preprocessed source RGB -> predicted confident boxes only -> clipped integer floor/ceil crop -> aspect-resize/bilinear antialias -> fixed meanRGB squarepadding -> normalization. RGB and grayscale-active-content variants; gray padding remains same meanRGB. Chroma ablation does NOT remove brightness or prove same-color generalization.',
        readouts=['rgb_cls','rgb_patch_mean','gray_cls','gray_patch_mean'],selection='original predicted score>=.5, no GTcount/crop/IoU selection; all original6query indices retained, invalid descriptor slots NaN+validmask. Main768detector/pose unchanged, extra cropencoder cost reported.'))
    started=time.monotonic();model=_build_patch_embed(16,1024)
    raw=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True);prefix='aggregator.patch_embed.';weights={k[len(prefix):]:v for k,v in raw.items() if k.startswith(prefix)}
    model.load_state_dict(weights,strict=True);del raw,weights;model=model.cuda().eval().requires_grad_(False)
    mean=torch.tensor(MEAN,device='cuda')[None,:,None,None];std=torch.tensor(STD,device='cuda')[None,:,None,None]
    def encode(x):
        with torch.autocast('cuda',dtype=torch.bfloat16):o=model((x.cuda()-mean)/std)
        return dict(cls=o['x_norm_clstoken'].float().cpu(),patch_mean=o['x_norm_patchtokens'].float().mean(1).cpu())
    saved=[];records=[];hashes={};crop_count=0;calls=0;tests={}
    for role,folder in SOURCES.items():
        source=CYCLE/folder/'groups.pt';hashes[role]=fingerprint(source);groups=torch.load(source,map_location='cpu',weights_only=False)
        for gi,g in enumerate(groups):
            item=load_item(g['record']);all_desc=[];valids=[];cropinfo=[]
            for vi,view in enumerate(g['record']['views']):
                assert dt.datetime.now(dt.timezone.utc)<cutoff
                image=item['batch']['images'][0,vi];boxes=g['predictions']['boxes'][vi];qs=torch.where(g['predictions']['objectness_logits'][vi].sigmoid()>=.5)[0]
                rgb=predicted_rgb_crops(image,boxes);gray=predicted_rgb_crops(image,boxes,grayscale=True);valid=rgb['valid'].clone();valid &= g['predictions']['objectness_logits'][vi].sigmoid()>=.5;qs=qs[valid[qs]]
                vectors={name:torch.full((6,1024),float('nan')) for name in ['rgb_cls','rgb_patch_mean','gray_cls','gray_patch_mean']}
                if len(qs):
                    inp=torch.cat([rgb['images'][qs],gray['images'][qs]])
                    torch.cuda.synchronize();t0=time.monotonic();out=encode(inp);torch.cuda.synchronize();elapsed=time.monotonic()-t0;calls+=1
                    for key in out:
                        vectors['rgb_'+key][qs]=out[key][:len(qs)];vectors['gray_'+key][qs]=out[key][len(qs):]
                    if not tests:
                        rev=encode(inp.flip(0));single=encode(inp[:1]);calls+=2
                        tests['reverse_max']=max(float((rev[k].flip(0)-out[k]).abs().max()) for k in out)
                        tests['singleton_max']=max(float((single[k]-out[k][:1]).abs().max()) for k in out)
                        # bf16 batch-shape kernel differences are recorded, not hidden.
                        tests['singleton_cosine_min']=min(float(torch.nn.functional.cosine_similarity(single[k],out[k][:1]).min()) for k in out)
                        assert tests['reverse_max']<1e-4 and tests['singleton_cosine_min']>.999,tests
                    crop_count+=2*len(qs)
                    if gi==0:
                        for q in qs.tolist():
                            for name,obj in [('rgb',rgb),('gray',gray)]:
                                path=f'crops/{role}_view{view}_q{q}_{name}.png';Image.fromarray((obj['images'][q].permute(1,2,0).numpy().clip(0,1)*255).round().astype(np.uint8)).save(OUT/path)
                else:elapsed=0.
                all_desc.append(vectors);valids.append(valid);cropinfo.append(dict(view=view,queries=qs.tolist(),bounds=rgb['bounds'].tolist(),content_bounds=rgb['content_bounds'].tolist()))
                records.append(dict(role=role,group=gi,view=view,queries=qs.tolist(),seconds=elapsed));write_json(OUT/'status.json',dict(stage='encoding',pid=os.getpid(),images=len(records),crops=crop_count))
            saved.append(dict(role=role,group=gi,scene=g['record']['scene'],views=g['record']['views'],descriptors={k:torch.stack([x[k] for x in all_desc]) for k in all_desc[0]},valid=torch.stack(valids),cropinfo=cropinfo))
            print(role,gi+1,'crops',crop_count,flush=True)
    torch.save(saved,OUT/'descriptors.pt');write_json(OUT/'results.json',dict(images=len(records),crop_examples=crop_count,encoder_batch_calls=calls,
        tests=tests,source_sha256=hashes,records=records,checkpoint_sha256=sha,script_sha256=fingerprint(Path(__file__)),module_sha256=fingerprint(ROOT/'mousegeoforge/predicted_rgb_crops.py'),elapsed_seconds=time.monotonic()-started,
        production_changed=False,training=False))
    write_json(OUT/'status.json',dict(stage='complete',pid=os.getpid()))
    state.update(stage='predicted_crop_encoder_ready',active_job=None,active_pid=None,next='Evaluate four fixed crop readouts and RGB/gray controls, inspect cropinputs; no VAL tuning.');state['completed'].append('predicted_crop_encoder_rgb_gray');write_json(CYCLE/'state.json',state)


if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write_json(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()))
        raise
