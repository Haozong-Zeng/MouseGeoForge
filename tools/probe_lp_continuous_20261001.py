"""Fixed LP single-view teacher on the identical pending-free continuous pilot."""
import argparse,datetime as dt,fcntl,json,os,time
from pathlib import Path
import cv2,numpy as np,torch
from lightning_pose.models.heatmap_tracker import HeatmapTracker
from torchvision import transforms
from run_teacher_cycle_dlc_20261001 import write_state
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/teacher_temporal_cycle_20261001'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=17);ap.add_argument('--output-suffix',default='');ap.add_argument('--clip-manifest',default='continuous_seed17_epoch30/manifest.json');args=ap.parse_args()
    assert '/' not in args.output_suffix
    run=f'lp_seed{args.seed}_kl'
    lock=(OUT/'gpu_job.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads((OUT/'plan.json').read_text());deadline=min(time.time()+600,dt.datetime.fromisoformat(plan['deadline_utc']).timestamp())
    m=json.loads((OUT/'dlc_dataset_manifest.json').read_text());A=np.array(m['affine_source_to_input'])
    excluded={(r['session'],r['frame']) for r in json.loads((ROOT/'output/pymousestudio_numbered_review_20261001/review_manifest.json').read_text())['cases']}
    base=json.loads((OUT/args.clip_manifest).read_text())
    ck=json.loads((OUT/run/'summary.json').read_text())['checkpoint']
    torch.set_num_threads(8);assert torch.cuda.is_available()
    model=HeatmapTracker.load_from_checkpoint(ck,map_location='cpu',pretrained=False,loss_factory=None,weights_only=False).to('cuda').eval()
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[.485,.456,.406],std=[.229,.224,.225])])
    def predict(images):
        assert time.time()<deadline and not (OUT/'STOP').exists()
        x=torch.stack([transform(im) for im in images]).to('cuda')
        with torch.inference_mode():p,c=model.head.run_subpixelmaxima(model(x))
        return p.cpu().numpy().reshape(-1,7,2),c.cpu().numpy()
    # Verify direct heatmap decoding against saved labeled-image inference.
    with np.load(OUT/run/'predictions_source_pixels.npz') as z:saved=z['uv']
    i=m['validation_indices'][0];start=(i//8)*8
    ims=[cv2.cvtColor(cv2.imread(str(ROOT/e['derived'])),cv2.COLOR_BGR2RGB) for e in m['entries'][start:start+8]]
    test,_=predict(ims);expected=saved[start:start+8]*np.diag(A)[:2]+A[:2,2]
    assert np.allclose(test,expected,atol=.1),np.max(np.abs(test-expected))
    dest=OUT/f'continuous_{run}{args.output_suffix}';dest.mkdir(exist_ok=False);manifest=[]
    write_state(stage='lp_continuous_inference',pid=os.getpid(),active_run=dest.name,checkpoint=ck)
    for clip in base:
        session=clip['session'];assert session in plan['development_validation_sessions']
        frames=list(range(clip['source_start'],clip['source_end_inclusive']+1))
        assert not any((session,f) in excluded for f in frames)
        folder=dest/session;folder.mkdir();uvs=[];confs=[]
        for cam in clip['cameras']:
            paths=list(Path(clip['folder']).glob(f'{session}_cam{cam}_*.avi'));assert len(paths)==1
            cap=cv2.VideoCapture(str(paths[0]));assert cap.isOpened();ims=[];uv=[];conf=[]
            for n in range(clip['frames']):
                ok,im=cap.read();assert ok;ims.append(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
                if len(ims)==8 or n==clip['frames']-1:
                    p,c=predict(ims);uv.extend(p);conf.extend(c);ims=[]
            cap.release();uvs.append((np.array(uv)-A[:2,2])/np.diag(A)[:2]);confs.append(conf)
            print('INFERRED',session,cam,flush=True)
        np.savez_compressed(folder/'predictions_source_pixels.npz',uv=uvs,confidence=confs,cameras=clip['cameras'],source_frames=frames,ids=m['ids_1based'],fps=52,train_allowed=False)
        manifest.append(dict(clip,folder=str(folder),video_folder=clip['folder'],scorer=f'LP-KL-seed{args.seed}',checkpoint=ck,
            normalization='torchvision CPU ToTensor+Normalize, identical to LP dataset',labeled_batch_parity_max_px=float(np.max(np.abs(test-expected)))))
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2))
    write_state(stage='lp_continuous_ready',next_action='Compare identical LP teacher predictions using framewise and temporal Anipose, true leave-camera-out.')

if __name__=='__main__':main()
