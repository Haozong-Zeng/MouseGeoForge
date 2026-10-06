"""Frozen image-backbone versus projected-memory predicted-ROI descriptors."""
import datetime as dt,fcntl,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import torch
from singleview_candidate_runtime import FrozenSingleViewCandidateRuntime
from run_surface_curve_pilot_20261003 import load_item
from mousegeoforge.predicted_roi_descriptor import pool_predicted_roi
from run_depth_pose_ablation import fingerprint,write_json
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';OUT=CYCLE/'prequery_roi_descriptors_v1'
SOURCES=dict(train='dual_train_allview_affinity_context_v1',validation='dual_allscene_association_audit_v1')


@torch.no_grad()
def main():
    plan=json.loads((CYCLE/'plan.json').read_text());cutoff=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'))
    assert dt.datetime.now(dt.timezone.utc)<cutoff
    lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not OUT.exists();OUT.mkdir();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='prequery_roi_extract_running',active_job=OUT.name,active_pid=os.getpid());write_json(CYCLE/'state.json',state)
    write_json(OUT/'protocol.json',dict(scope='No training. Frozen VGGT-Omega image backbone x_norm_patchtokens before instance queries versus projected decoder-memory map. Pool5x5 within fresh predicted box extent (not2x), mean valid in-image samples; all6queries, no GTboxes/count/mask. Existing21TRAIN/16VALframes all4views. Labels never used in extraction.',
        hypothesis='Earlier pose-token/local-point cosine tracks query slot rather than animal. This separate probe removes query embeddings from descriptor and samples predicted boxes, not inaccurate foot positions. Not an independently pretrained/re-ID tuned encoder. No new model download.'))
    started=time.monotonic();runtime=FrozenSingleViewCandidateRuntime();captured=[];records=[];saved=[];hashes={};count=0
    hook=runtime.model.aggregator.patch_embed.register_forward_hook(lambda module,args,out:captured.append(out['x_norm_patchtokens'].detach()))
    try:
        for role,folder in SOURCES.items():
            source=CYCLE/folder/'groups.pt';hashes[role]=fingerprint(source);groups=torch.load(source,map_location='cpu',weights_only=False)
            for gi,g in enumerate(groups):
                assert g['record']['role']==role;item=load_item(g['record']);rgb=item['batch']['images'];views=g['record']['views'];descriptors=[]
                for vi,view in enumerate(views):
                    assert dt.datetime.now(dt.timezone.utc)<cutoff
                    captured.clear();torch.cuda.synchronize();t0=time.monotonic();raw=runtime.from_images(rgb[:,vi:vi+1]);torch.cuda.synchronize();elapsed=time.monotonic()-t0
                    assert len(captured)==1;h,w=rgb.shape[-2:];features=captured[0].transpose(1,2).reshape(1,1024,h//16,w//16)
                    boxes=raw['boxes'][None];a=pool_predicted_roi(features,boxes);b=pool_predicted_roi(raw['maps'],boxes)
                    delta=max(float((raw[k].cpu()-g['predictions'][k][vi]).abs().max()) for k in ['xy','boxes','objectness_logits']);assert delta<2e-5,delta
                    descriptors.append(dict(image_backbone=a['descriptor'][0].cpu(),projected_memory=b['descriptor'][0].cpu(),valid=a['valid'][0].cpu(),sampled_fraction=a['sampled_fraction'][0].cpu()))
                    count+=1;records.append(dict(role=role,group=gi,view=view,delta=delta,seconds=elapsed))
                    write_json(OUT/'status.json',dict(stage='extracting',pid=os.getpid(),images=count,total=148));del raw,features
                saved.append(dict(role=role,group=gi,scene=g['record']['scene'],views=views,descriptors={k:torch.stack([x[k] for x in descriptors]) for k in descriptors[0]}))
                print(role,gi+1,'images',count,flush=True)
    finally:hook.remove()
    assert count==148
    torch.save(saved,OUT/'descriptors.pt');write_json(OUT/'results.json',dict(images=count,records=records,source_sha256=hashes,runtime_sha256=runtime.hashes,
        descriptor_sha256=fingerprint(OUT/'descriptors.pt'),script_sha256=fingerprint(Path(__file__)),module_sha256=fingerprint(ROOT/'mousegeoforge/predicted_roi_descriptor.py'),elapsed_seconds=time.monotonic()-started))
    write_json(OUT/'status.json',dict(stage='complete',pid=os.getpid()))
    state.update(stage='prequery_roi_descriptors_ready',active_job=None,active_pid=None,next='Evaluate new ROI descriptor cosine separately TRAIN/VAL against prior geometry/semantic controls, no VAL tuning; record partial matcher zero coverage and descriptor outcome.')
    state['completed'].append('prequery_roi_descriptors148');write_json(CYCLE/'state.json',state)


if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write_json(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()))
        raise
