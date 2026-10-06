"""Frozen two-seed inference on existing exact lossless development clips."""
import datetime as dt
import fcntl
import hashlib
import json
import os
os.environ['DLClight']='True'
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from PIL import Image
from bottom_anchor_common_20261002 import ROOT, OUT, PRIOR, budget, write_state, exclusions


def main():
    plan, deadline, check=budget(1200)
    lock=(ROOT/plan['gpu_lock']).open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=json.loads((OUT/'state.json').read_text());assert state.get('pid') is None
    dest=OUT/'quadruped_temporal_frozen_v1';dest.mkdir(exist_ok=False)
    clips=json.loads((PRIOR/'png_exact_clips_v1/manifest.json').read_text())
    dm=json.loads((OUT/'superanimal_quadruped_hrnet_v1/dataset_manifest.json').read_text())
    entries=[dm['entries'][i] for i in dm['validation_indices']]
    lookup={(e['session'],e['camera'],int(e['frame'])):i for i,e in enumerate(entries)}
    excluded=exclusions(plan)
    for clip in clips:
        assert clip['session'] in plan['development_sessions'] and clip['train_allowed'] is False
        assert clip['fps']==52 and clip['frames']==208
        assert not any((clip['session'],f) in excluded for f in range(clip['source_start'],clip['source_end']+1))
    settings=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),clips=clips,seeds=[17,29],
        selection='Reuse all4 previously fixed PNG-exact windows, not selected for new teacher errors; same development videos, NOT independent test.',
        input_wh=[704,768],source_wh=[1080,1200],fps=52,batch_size=8,
        normalization='CPU float32 RGB/255 then ImageNet normalization, same as immutable checkpoint audits',
        training=False,adaptation=False,geometry_correction=False,pseudo_gt_released=False,
        missing_label_policy='Unknown; all7 raw outputs cached, only existing labels evaluated. No visibility negatives.')
    (dest/'settings.json').write_text(json.dumps(settings,indent=2))
    write_state(stage='quadruped_frozen_temporal_inference',pid=os.getpid(),active_run=dest.name,mode='GPU_inference',
        job_deadline_utc=dt.datetime.fromtimestamp(deadline,dt.timezone.utc).isoformat(),error=None)
    records=[]
    try:
        from deeplabcut.pose_estimation_pytorch.models import PoseModel
        torch.set_num_threads(6);assert torch.cuda.is_available()
        cfg=json.loads((OUT/'superanimal_quadruped_preparation_v1/model_config.json').read_text())
        A=np.array(dm['affine_source_to_input']);scale=np.diag(A)[:2]
        norm_mean=torch.tensor([.485,.456,.406]).view(1,3,1,1)
        norm_std=torch.tensor([.229,.224,.225]).view(1,3,1,1)
        for seed,epoch,run_name,review in [(17,110,'superanimal_quadruped_hrnet_v1','quadruped_epoch110_review_v1'),
                                           (29,85,'superanimal_quadruped_hrnet_seed29_v1','quadruped_seed29_epoch085_review_v1')]:
            check();paths=list((OUT/run_name/'project/dlc-models-pytorch').rglob(f'snapshot-best-{epoch:03d}.pt'));assert len(paths)==1
            snapshot=paths[0];sha=hashlib.sha256(snapshot.read_bytes()).hexdigest()
            weights=torch.load(snapshot,map_location='cpu',weights_only=False)
            assert weights['metadata']['epoch']==epoch
            model=PoseModel.build(cfg,pretrained_backbone=False);model.load_state_dict(weights['model'],strict=True)
            del weights
            model.cuda().eval();assert not any(m.training for m in model.modules())
            with np.load(OUT/review/'predictions_source_pixels.npz') as z: sparse={k:z[k].copy() for k in z.files}
            for clip in clips:
                check();cam_predictions=[];parities=[];began=time.monotonic()
                for camera in clip['cameras']:
                    paths=list(Path(clip['video_folder']).glob(f'{clip["session"]}_cam{camera}_*.avi'));assert len(paths)==1
                    cap=cv2.VideoCapture(str(paths[0]));assert cap.isOpened()
                    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==208 and cap.get(cv2.CAP_PROP_FPS)==52
                    pending=[];pred=[]
                    for fi in range(208):
                        check();ok,bgr=cap.read();assert ok and bgr.shape==(768,704,3)
                        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                        key=(clip['session'],camera,clip['source_start']+fi)
                        if key in lookup:
                            idx=lookup[key];still=np.asarray(Image.open(ROOT/entries[idx]['derived']).convert('RGB'))
                            assert np.array_equal(rgb,still),f'Image parity failure: {key}'
                        pending.append(torch.from_numpy(rgb.copy()).permute(2,0,1).float()/255.)
                        if len(pending)==8 or fi==207:
                            x=(torch.stack(pending)-norm_mean)/norm_std
                            with torch.inference_mode():
                                poses=model.get_predictions(model(x.cuda()))['bodypart']['poses'].detach().cpu().numpy().reshape(len(pending),7,3)
                            pred.extend(poses);pending=[]
                    cap.release();pred=np.stack(pred);assert pred.shape==(208,7,3) and np.isfinite(pred).all()
                    uv=(pred[:,:,:2]-A[:2,2])/scale
                    for fi in range(208):
                        key=(clip['session'],camera,clip['source_start']+fi)
                        if key in lookup:
                            idx=lookup[key];distance=np.linalg.norm(uv[fi]-sparse['uv'][idx],axis=-1)
                            # Argmax ties can change under batching; report all outputs, evaluate observed labels too.
                            valid=np.isfinite(sparse['gt'][idx]).all(axis=-1)
                            parities.append(dict(camera=camera,frame=key[2],all_channel_max_source_px=float(distance.max()),
                                observed_channel_max_source_px=float(distance[valid].max()) if valid.any() else None,pixels_exact=True))
                    cam_predictions.append(pred)
                    print('INFERRED',seed,clip['cohort'],clip['session'],camera,flush=True)
                    write_state(inference_seed=seed,inference_clip=f'{clip["cohort"]}/{clip["session"]}',inference_camera=camera)
                preds=np.stack(cam_predictions);uv=(preds[:,:,:,:2]-A[:2,2])/scale
                filename=f'seed{seed}_{clip["cohort"]}_{clip["session"]}.npz'
                np.savez_compressed(dest/filename,uv=uv,confidence=preds[:,:,:,2],source_frames=np.arange(clip['source_start'],clip['source_end']+1),
                    cameras=clip['cameras'],ids=dm['ids_1based'],fps=52)
                records.append(dict(seed=seed,epoch=epoch,checkpoint=str(snapshot),sha256=sha,file=filename,clip=clip,
                    seconds=time.monotonic()-began,manual_frame_parity=parities))
                (dest/'progress.json').write_text(json.dumps(records,indent=2))
            del model;torch.cuda.empty_cache()
        (dest/'manifest.json').write_text(json.dumps(records,indent=2))
        write_state(stage='quadruped_frozen_temporal_complete',pid=None,active_run=None,mode='idle',job_deadline_utc=None,
            latest_result=str(dest/'manifest.json'),next_action='Validate temporal/sparse prediction parity, render all-view and A/F comparison clips, evaluate only existing manual frames. Smoothness is not accuracy; no correction/pseudoGT.')
        print('COMPLETE',len(records),'seed-clips',flush=True)
    except BaseException as exc:
        (dest/'failed.json').write_text(json.dumps(dict(error=repr(exc),completed_seed_clips=len(records)),indent=2))
        write_state(stage='quadruped_temporal_inference_failed',pid=None,active_run=None,mode='idle',error=repr(exc),job_deadline_utc=None)
        raise
    finally:
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':main()
