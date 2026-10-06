"""Predeclared disjoint temporal segments; NOT a new independent test set."""
import datetime as dt,json,time
from pathlib import Path
import cv2,numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/teacher_temporal_cycle_20261001'
def main():
    plan=json.loads((OUT/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['deadline_utc']).timestamp()
    excluded={(r['session'],r['frame']) for r in json.loads((ROOT/'output/pymousestudio_numbered_review_20261001/review_manifest.json').read_text())['cases']}
    dest=OUT/'stress_clips_v1';dest.mkdir(exist_ok=False)
    settings=dict(utc_declared=dt.datetime.now(dt.timezone.utc).isoformat(),windows=[dict(session=s,start=1100,end_inclusive=1307) for s in ['AZR2L2','FR2']],
        selection='Fixed later time window, no content-based selection. No overlap with pilot windows. SAME development sequences, not independent animals/test set.',
        methods_frozen='3 LP-KL seeds; median; framewise equal/uncertainty weighting, 4px floor, soft_l1 fscale3; confidence>=.6 spread<=10px gates; max15px reprojection; ray angle>=3deg; EKS s=1 with4px variance floor.',train_allowed=False)
    (dest/'predeclaration.json').write_text(json.dumps(settings,indent=2));clips=[]
    for w in settings['windows']:
        s=w['session'];assert s in plan['development_validation_sessions'];frames=list(range(w['start'],w['end_inclusive']+1))
        assert not any((s,f) in excluded for f in frames)
        folder=dest/s;folder.mkdir();cams=[]
        for src in sorted((ROOT/'dataset/PyMouseStudio/sessions'/s/'videos').glob('cam*.mp4')):
            assert time.time()<deadline and not (OUT/'STOP').exists()
            c=src.stem[3:];cams.append(c);cap=cv2.VideoCapture(str(src));assert cap.isOpened()
            assert cap.get(cv2.CAP_PROP_FPS)==52 and cap.get(cv2.CAP_PROP_FRAME_COUNT)>frames[-1]
            cap.set(cv2.CAP_PROP_POS_FRAMES,frames[0]);path=folder/f'{s}_cam{c}_{frames[0]:06d}_704x768.avi'
            writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'FFV1'),52,(704,768));assert writer.isOpened()
            for frame in frames:
                ok,bgr=cap.read();assert ok
                rgb=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).resize((691,768),Image.Resampling.LANCZOS)
                canvas=np.zeros((768,704,3),np.uint8);canvas[:,6:697]=np.array(rgb);writer.write(cv2.cvtColor(canvas,cv2.COLOR_RGB2BGR))
            cap.release();writer.release();print('EXTRACTED',s,c,flush=True)
        clips.append(dict(session=s,source_start=frames[0],source_end_inclusive=frames[-1],fps=52,cameras=cams,frames=len(frames),folder=str(folder),pending_frames_present=False))
    (dest/'manifest.json').write_text(json.dumps(clips,indent=2))
if __name__=='__main__':main()
