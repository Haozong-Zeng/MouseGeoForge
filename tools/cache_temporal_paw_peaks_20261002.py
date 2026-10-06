"""Actual LP top3 peaks on existing lossless clips; serial bounded GPU job."""
import argparse,datetime as dt,fcntl,json,os,time
from pathlib import Path
import cv2,numpy as np,torch
from torch.nn import functional as F
from torchvision import transforms
from lightning_pose.models.heatmap_tracker import HeatmapTracker
from run_teacher_cycle_dlc_20261001 import ROOT,OUT,write_state

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--exact-manifest');ap.add_argument('--output-run',default='temporal_paw_peaks_v1');args=ap.parse_args()
    assert '/' not in args.output_run
    lock=(OUT/'gpu_job.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads((OUT/'plan.json').read_text());deadline=min(time.time()+1200,dt.datetime.fromisoformat(plan['deadline_utc']).timestamp())
    def check():assert time.time()<deadline and not (OUT/'STOP').exists()
    check();dest=OUT/args.output_run;dest.mkdir(exist_ok=False)
    m=json.loads((OUT/'dlc_dataset_manifest.json').read_text());A=np.array(m['affine_source_to_input']);js=[m['ids_1based'].index(k) for k in [15,16,17,18]]
    excluded={(r['session'],r['frame']) for r in json.loads((ROOT/plan['exclusion_manifest']).read_text())['cases']};excluded.add(('AZR2L2',584))
    settings=dict(seeds=[17,31,73],nms_kernel=7,topk=3,ids=[15,16,17,18],fps=52,
        source_units='1080x1200pixels',normalization='CPU torchvision ToTensor+Normalize same as dataset',parity_tolerance_px=.1,exact_manifest=args.exact_manifest,train_allowed=False)
    (dest/'settings.json').write_text(json.dumps(settings,indent=2));records=[]
    write_state(stage='temporal_paw_peaks_caching',pid=os.getpid(),active_run=dest.name,mode='GPU_inference',job_deadline_utc=dt.datetime.fromtimestamp(deadline,dt.timezone.utc).isoformat())
    torch.set_num_threads(8);assert torch.cuda.is_available()
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    for seed in settings['seeds']:
        check();run=OUT/f'lp_seed{seed}_kl';checkpoint=json.loads((run/'summary.json').read_text())['checkpoint']
        model=HeatmapTracker.load_from_checkpoint(checkpoint,map_location='cpu',pretrained=False,loss_factory=None,weights_only=False).cuda().eval()
        with np.load(run/'predictions_source_pixels.npz') as z:saved=z['uv']
        first=m['validation_indices'][0];aligned=(first//8)*8
        x=torch.stack([transform(cv2.cvtColor(cv2.imread(str(ROOT/e['derived'])),cv2.COLOR_BGR2RGB)) for e in m['entries'][aligned:aligned+8]]).cuda()
        with torch.inference_mode():p,_=model.head.run_subpixelmaxima(model(x))
        parity_aligned=float(np.max(np.abs(p.cpu().numpy().reshape(-1,7,2)-(saved[aligned:aligned+8]*np.diag(A)[:2]+A[:2,2]))));assert parity_aligned<.1
        for cohort,suffix in [('pilot','_exactnorm'),('stress1','_stress1_exactnorm')]:
            (dest/cohort).mkdir(exist_ok=True)
            manifest=([r for r in json.loads((OUT/args.exact_manifest).read_text()) if r['cohort']==cohort] if args.exact_manifest else json.loads((OUT/f'continuous_lp_seed{seed}_kl{suffix}/manifest.json').read_text()))
            for clip in manifest:
                check();session=clip['session'];assert session in plan['development_validation_sessions']
                if args.exact_manifest:
                    frames=np.arange(clip['source_start'],clip['source_end']+1);cams=clip['cameras'];reference=None
                else:
                    with np.load(Path(clip['folder'])/'predictions_source_pixels.npz') as z:reference=dict(z)
                    frames=reference['source_frames'];cams=reference['cameras'].tolist()
                assert np.all(np.diff(frames)==1) and not any((session,int(f)) in excluded for f in frames)
                cam_peaks=[];cam_ratios=[];parities=[parity_aligned];peak_parities=[]
                for vi,c in enumerate(cams):
                    paths=list(Path(clip['video_folder']).glob(f'{session}_cam{c}_*.avi'));assert len(paths)==1
                    cap=cv2.VideoCapture(str(paths[0]));assert cap.isOpened();images=[];peaks=[];ratios=[]
                    for fi in range(len(frames)):
                        check();ok,im=cap.read();assert ok;images.append(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
                        if len(images)==8 or fi==len(frames)-1:
                            x=torch.stack([transform(im) for im in images]).cuda()
                            with torch.inference_mode():
                                hm=model(x)
                                if not peaks and reference is not None:
                                    p,_=model.head.run_subpixelmaxima(hm);p=(p.cpu().numpy().reshape(-1,7,2)-A[:2,2])/np.diag(A)[:2]
                                    parity=float(np.max(np.abs(p-reference['uv'][vi,:len(images)])));assert parity<.1,parity;parities.append(parity)
                                paw=hm[:,js];assert tuple(paw.shape[-2:])==(192,176)
                                local=paw.eq(F.max_pool2d(paw,7,stride=1,padding=3))
                                score,idx=paw.masked_fill(~local,-torch.inf).flatten(-2).topk(3,dim=-1)
                                xy=torch.stack([idx%176,idx//176],dim=-1)*4
                                peaks.append((xy.cpu().numpy()-A[:2,2])/np.diag(A)[:2]);ratios.append((score/score[...,:1]).cpu().numpy())
                            images=[]
                    cap.release();cam_peaks.append(np.concatenate(peaks));cam_ratios.append(np.concatenate(ratios))
                    if args.exact_manifest:
                        with np.load(OUT/f'paw_heatmap_peaks_v1/seed{seed}.npz') as z:sparse=dict(z)
                        for pos,idx0 in enumerate(sparse['indices']):
                            e=m['entries'][int(idx0)]
                            if e['session']==session and e['camera']==c and e['frame'] in frames:
                                fi=frames.tolist().index(e['frame']);err=np.linalg.norm(cam_peaks[-1][fi]-sparse['peaks_source_px'][pos,:,:3],axis=-1)
                                peak_parities.append(dict(camera=c,frame=e['frame'],peak_max_distance_px=float(err.max()),ratio_max_difference=float(np.max(np.abs(cam_ratios[-1][fi]-sparse['relative_scores'][pos,:,:3])))))
                    print('CACHED',seed,cohort,session,c,flush=True)
                path=dest/cohort/f'seed{seed}_{session}.npz'
                np.savez_compressed(path,peaks_source_px=cam_peaks,relative_scores=cam_ratios,cameras=cams,source_frames=frames,ids=[15,16,17,18],fps=52,train_allowed=False)
                record=dict(seed=seed,cohort=cohort,session=session,file=str(path),cameras=cams,frames=len(frames),source_start=int(frames[0]),source_end=int(frames[-1]),checkpoint=checkpoint,
                    parity_max_px=max(parities),sparse_peak_parities=peak_parities,video_folder=clip['video_folder'],train_allowed=False)
                records.append(record);(dest/'progress.json').write_text(json.dumps(records,indent=2))
        del model;torch.cuda.empty_cache()
    (dest/'manifest.json').write_text(json.dumps(records,indent=2))
    write_state(stage='temporal_paw_peaks_cached',pid=None,active_run=None,mode='idle_between_heartbeats',job_deadline_utc=None,latest_result=str(dest/'manifest.json'),
        next_action='Build fixedcamera3D joint hypotheses without heldview input, then compare same-state independent and temporal identity selection with missing states and no fill.')
    print('COMPLETE',len(records),'seed-clips',flush=True)

if __name__=='__main__':main()
