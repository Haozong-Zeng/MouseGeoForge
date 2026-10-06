"""Read-only CPU/GPU and batch sensitivity audit on two known one-cell differences."""
import datetime as dt
import argparse
import fcntl
import json
import os
os.environ['DLClight']='True'
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from bottom_anchor_common_20261002 import ROOT,OUT,PRIOR,budget,write_state


def main(run_name='quadruped_batch_peak_audit_v1'):
    plan,deadline,check=budget(300)
    lock=(ROOT/plan['gpu_lock']).open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert run_name in ('quadruped_batch_peak_audit_v1','quadruped_batch_peak_native_norm_v2')
    dest=OUT/run_name;dest.mkdir(exist_ok=False)
    write_state(stage='quadruped_batch_peak_audit',pid=os.getpid(),active_run=dest.name,mode='GPU_inference',job_deadline_utc=dt.datetime.fromtimestamp(deadline,dt.timezone.utc).isoformat())
    try:
        from deeplabcut.pose_estimation_pytorch.models import PoseModel
        torch.set_num_threads(6)
        dm=json.loads((OUT/'superanimal_quadruped_hrnet_v1/dataset_manifest.json').read_text())
        entries=[dm['entries'][i] for i in dm['validation_indices']]
        with np.load(OUT/'quadruped_seed29_epoch085_review_v1/predictions_source_pixels.npz') as z:sparse={k:z[k].copy() for k in z.files}
        clips=json.loads((PRIOR/'png_exact_clips_v1/manifest.json').read_text())
        cfg=json.loads((OUT/'superanimal_quadruped_preparation_v1/model_config.json').read_text())
        path=next((OUT/'superanimal_quadruped_hrnet_seed29_v1/project/dlc-models-pytorch').rglob('snapshot-best-085.pt'))
        model=PoseModel.build(cfg,pretrained_backbone=False)
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model'],strict=True);model.eval()
        bn_before={k:v.clone() for k,v in model.state_dict().items() if 'running_' in k or 'num_batches_tracked' in k}
        mean=torch.tensor([.485,.456,.406]).view(1,3,1,1);std=torch.tensor([.229,.224,.225]).view(1,3,1,1)
        A=np.array(dm['affine_source_to_input']);scale=np.diag(A)[:2]
        results=[]
        for session,camera,frame in [('AZR2L2','A',1234),('AZR2L2','E',1299)]:
            check();i=next(i for i,e in enumerate(entries) if (e['session'],e['camera'],e['frame'])==(session,camera,frame))
            image=np.asarray(Image.open(ROOT/entries[i]['derived']).convert('RGB'))
            clip=next(c for c in clips if c['session']==session and c['source_start']<=frame<=c['source_end'])
            fi=frame-clip['source_start'];offset=fi%8;start=fi-offset
            video=next(Path(clip['video_folder']).glob(f'{session}_cam{camera}_*.avi'))
            cap=cv2.VideoCapture(str(video));cap.set(cv2.CAP_PROP_POS_FRAMES,start);batch=[]
            for _ in range(8):ok,bgr=cap.read();assert ok;batch.append(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
            cap.release();assert np.array_equal(batch[offset],image)
            def tensor(images):return (torch.from_numpy(np.stack(images).copy()).permute(0,3,1,2).float()/255.-mean)/std
            single=tensor([image]);temporal=tensor(batch);repeat=single.repeat(8,1,1,1)
            assert torch.equal(single[0],temporal[offset])
            import albumentations as alb
            normalize=alb.Normalize(mean=[.485,.456,.406],std=[.229,.224,.225])
            native_single=torch.from_numpy(normalize(image=image)['image'].copy()).permute(2,0,1).unsqueeze(0)
            native_temporal=torch.stack([torch.from_numpy(normalize(image=im)['image'].copy()).permute(2,0,1) for im in batch])
            with np.load(OUT/f'quadruped_temporal_frozen_v1/seed29_{clip["cohort"]}_{session}.npz') as z:
                cached_temporal=z['uv'][clip['cameras'].index(camera),fi].copy()
            changed=np.flatnonzero(np.linalg.norm(cached_temporal-sparse['uv'][i],axis=-1)>.01)
            variants={};heatmaps={}
            for mode,x,pos,device in [('cpu_single',single,0,'cpu'),('gpu_single',single,0,'cuda'),
                ('gpu_repeat8',repeat,0,'cuda'),('gpu_temporal8',temporal,offset,'cuda'),('gpu_temporal8_repeat',temporal,offset,'cuda'),
                ('cpu_native_norm_single',native_single,0,'cpu'),('gpu_native_norm_single',native_single,0,'cuda'),('gpu_native_norm_temporal8',native_temporal,offset,'cuda')]:
                check();model.to(device)
                with torch.inference_mode():
                    output=model(x.to(device));hm=output['bodypart']['heatmap'][pos].cpu().numpy()
                    pose=model.get_predictions(output)['bodypart']['poses'][pos].cpu().numpy().reshape(7,3)
                uv=(pose[:,:2]-A[:2,2])/scale;heatmaps[mode]=hm
                details=[]
                for j in changed:
                    flat=hm[j].ravel();top=np.argsort(-flat,kind='stable')[:2]
                    details.append(dict(id=dm['ids_1based'][j],heatmap_shape=list(hm[j].shape),
                        top2_xy=[[int(t%hm.shape[-1]),int(t//hm.shape[-1])] for t in top],
                        top2_values=flat[top].tolist(),gap=float(flat[top[0]]-flat[top[1]]),
                        prediction_source=uv[j].tolist(),sparse_source=sparse['uv'][i,j].tolist(),temporal_source=cached_temporal[j].tolist()))
                variants[mode]=dict(changed_keypoints=details,max_distance_to_sparse=float(np.max(np.linalg.norm(uv-sparse['uv'][i],axis=-1))),
                    max_distance_to_temporal=float(np.max(np.linalg.norm(uv-cached_temporal,axis=-1))))
            comparisons={}
            for left,right in [('cpu_single','gpu_single'),('gpu_single','gpu_repeat8'),('gpu_repeat8','gpu_temporal8'),('gpu_temporal8','gpu_temporal8_repeat'),('cpu_single','cpu_native_norm_single'),('gpu_single','gpu_native_norm_single'),('gpu_temporal8','gpu_native_norm_temporal8')]:
                comparisons[left+'_vs_'+right]=dict(heatmap_max_abs_difference=float(np.max(np.abs(heatmaps[left]-heatmaps[right]))),
                    changed_channel_max_abs_difference={str(dm['ids_1based'][j]):float(np.max(np.abs(heatmaps[left][j]-heatmaps[right][j]))) for j in changed})
            results.append(dict(session=session,camera=camera,frame=frame,input_pixel_exact=True,input_normalized_tensor_exact=True,
                native_vs_torch_normalization_max_abs_difference=float(torch.max(torch.abs(native_single-single))),
                changed_ids=[dm['ids_1based'][j] for j in changed],variants=variants,comparisons=comparisons))
        after=model.state_dict();assert all(torch.equal(v,after[k].cpu()) for k,v in bn_before.items())
        report=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),results=results,
            predictor='HeatmapPredictor: integer argmax, stride4, +half-stride center offset, no locref/subpixel refinement. clip_scores clips only scores, not heatmap used for argmax.',
            source_file='DeepLabCut/deeplabcut/pose_estimation_pytorch/models/predictors/single_predictor.py',
            environment=dict(torch=torch.__version__,cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32),
            bn_buffers_unchanged=True,model_weights_not_updated=True,locked_holdouts_accessed=False,pseudo_gt_released=False)
        (dest/'summary.json').write_text(json.dumps(report,indent=2))
        write_state(stage='quadruped_batch_peak_audit_complete',pid=None,active_run=None,mode='idle',job_deadline_utc=None,
            latest_decision=str(dest/'summary.json'),next_action='Interpret numerical audit and synthesize completed teacher cycle. Do not replace weights/labels or infer dense accuracy.')
        print(json.dumps(report,indent=2))
    except BaseException as exc:
        (dest/'failed.json').write_text(json.dumps(dict(error=repr(exc))))
        write_state(stage='quadruped_batch_peak_audit_failed',pid=None,active_run=None,mode='idle',error=repr(exc),job_deadline_utc=None)
        raise
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run-name',default='quadruped_batch_peak_audit_v1');args=ap.parse_args();main(args.run_name)
