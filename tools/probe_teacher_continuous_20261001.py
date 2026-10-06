"""True DLC inference on fixed contiguous development clips, preserving transforms."""
import os
os.environ['DLClight']='True'
import datetime as dt
import fcntl
import json
import subprocess
import sys
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/teacher_temporal_cycle_20261001'
DEST=OUT/'continuous_seed17_epoch30'

def main():
    if '--launch' in sys.argv:
        assert not (OUT/'continuous_seed17_epoch30.log').exists()
        with (OUT/'continuous_seed17_epoch30.log').open('x') as f:
            p=subprocess.Popen([sys.executable,'-u',__file__],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        print('LAUNCHED',p.pid);return
    lock=(OUT/'gpu_job.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    from run_teacher_cycle_dlc_20261001 import write_state
    import torch
    import deeplabcut
    from deeplabcut.pose_estimation_pytorch.apis.videos import analyze_videos
    import pandas as pd
    torch.set_num_threads(8);assert torch.cuda.is_available()
    deadline=dt.datetime.fromisoformat(json.loads((OUT/'plan.json').read_text())['deadline_utc'])
    def check():
        if dt.datetime.now(dt.timezone.utc)>=deadline or (OUT/'STOP').exists():raise RuntimeError('deadline/STOP')
    DEST.mkdir(exist_ok=False)
    m=json.loads((OUT/'dlc_dataset_manifest.json').read_text())
    excluded={(r['session'],r['frame']) for r in json.loads((ROOT/'output/pymousestudio_numbered_review_20261001/review_manifest.json').read_text())['cases']}
    A=np.array(m['affine_source_to_input']);clips=[]
    write_state(stage='continuous_probe_preparation',pid=os.getpid(),job_log=str(OUT/'continuous_seed17_epoch30.log'))
    for session,start in [('AZR2L2',100),('FR2',700)]:
        indices=list(range(start,start+208));assert not any((session,f) in excluded for f in indices)
        folder=DEST/session;folder.mkdir();videos=ROOT/'dataset/PyMouseStudio/sessions'/session/'videos'
        sources=sorted(videos.glob('cam*.mp4'));cams=[];derived=[]
        for src in sources:
            check();cams.append(src.stem[3:]);target=folder/f'{session}_{src.stem}_{start:06d}_704x768.avi'
            cap=cv2.VideoCapture(str(src));fps=cap.get(cv2.CAP_PROP_FPS)
            assert fps==52 and cap.get(cv2.CAP_PROP_FRAME_COUNT)>=indices[-1]+1
            cap.set(cv2.CAP_PROP_POS_FRAMES,start)
            writer=cv2.VideoWriter(str(target),cv2.VideoWriter_fourcc(*'FFV1'),52,(704,768));assert writer.isOpened()
            for frame in indices:
                check();ok,bgr=cap.read();assert ok
                from PIL import Image
                rgb=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).resize((691,768),Image.Resampling.LANCZOS)
                canvas=np.zeros((768,704,3),np.uint8);canvas[:,6:697]=np.array(rgb)
                writer.write(cv2.cvtColor(canvas,cv2.COLOR_RGB2BGR))
            cap.release();writer.release();derived.append(target)
        write_state(stage='continuous_dlc_inference',session=session,pid=os.getpid())
        pred_dir=folder/'predictions';pred_dir.mkdir()
        # Lossless derived video uses exactly the training letterbox transform.
        # Independent single-view teacher: no other-view input or temporal context.
        scorer=analyze_videos(m['project_config'],[str(p) for p in derived],video_extensions=['avi'],
            snapshot_index='best',device='cuda',destfolder=pred_dir,batch_size=8,save_as_df=True,overwrite=False)
        alluv=[];allconf=[]
        for cam,target in zip(cams,derived):
            matches=list(pred_dir.glob(target.stem+'*.h5'));assert len(matches)==1,matches
            df=pd.read_hdf(matches[0]);assert len(df)==208
            def one(n,coord):
                x=df.xs(n,level='bodyparts',axis=1).xs(coord,level='coords',axis=1)
                return x.values[:,0]
            uv=np.stack([np.stack([one(n,'x'),one(n,'y')],axis=-1) for n in m['names']],axis=1)
            conf=np.stack([one(n,'likelihood') for n in m['names']],axis=1)
            alluv.append((uv-A[:2,2])/np.diag(A)[:2]);allconf.append(conf)
        np.savez_compressed(folder/'predictions_source_pixels.npz',uv=np.array(alluv),confidence=np.array(allconf),
            cameras=np.array(cams),source_frames=np.array(indices),fps=52,ids=m['ids_1based'],train_allowed=False)
        clips.append(dict(session=session,source_start=start,source_end_inclusive=indices[-1],fps=52,cameras=cams,
                          frames=208,scorer=scorer,folder=str(folder),pending_frames_present=False))
        (DEST/'manifest.json').write_text(json.dumps(clips,indent=2))
        print('DONE_CLIP',session,flush=True)
    write_state(stage='continuous_dlc_inference_complete',pid=os.getpid(),next_action='Compare framewise/temporal Anipose using fixed independent 2D teacher predictions; inspect source manual frames and motion fidelity')

if __name__=='__main__':main()
