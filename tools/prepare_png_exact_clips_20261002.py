"""Once-only remap of ORIGINAL unrectified raw clips; never remap rectified video."""
import datetime as dt,hashlib,json,os,time
from pathlib import Path
import cv2,numpy as np
from PIL import Image
from run_teacher_cycle_dlc_20261001 import ROOT,OUT,write_state

def main():
    plan=json.loads((OUT/'plan.json').read_text());deadline=min(time.time()+900,dt.datetime.fromisoformat(plan['deadline_utc']).timestamp())
    def check():assert time.time()<deadline and not (OUT/'STOP').exists()
    check();dest=OUT/'png_exact_clips_v1';dest.mkdir(exist_ok=False);cv2.setNumThreads(1)
    records=[json.loads(s) for s in (ROOT/'dataset/PyMouseStudioRaw/_validation_v2/sequences.jsonl').read_text().splitlines()]
    m=json.loads((OUT/'dlc_dataset_manifest.json').read_text());entries={(e['session'],e['frame'],e['camera']):e for e in m['entries'] if e['role']=='validation'}
    windows=[dict(cohort='pilot',session='AZR2L2',start=100,end=307),dict(cohort='pilot',session='FR2',start=700,end=907),dict(cohort='stress1',session='AZR2L2',start=1100,end=1307),dict(cohort='stress1',session='FR2',start=1100,end=1307)]
    excluded={(r['session'],r['frame']) for r in json.loads((ROOT/plan['exclusion_manifest']).read_text())['cases']};excluded.add(('AZR2L2',584))
    manifest=[];qa=[]
    (dest/'settings.json').write_text(json.dumps(dict(windows=windows,source='ORIGINAL PyMouseStudioRaw videos only',rectification='Once-only existing sourceK/D, originalK output, INTER_LINEAR/BORDER_CONSTANT',
        input_wh=[704,768],rgb_resize='PIL LANCZOS691x768,left6/right7',codec='FFV1 lossless BGR',train_allowed=False),indent=2))
    write_state(stage='png_exact_clips_preparing',pid=os.getpid(),active_run=dest.name,mode='CPU',job_deadline_utc=dt.datetime.fromtimestamp(deadline,dt.timezone.utc).isoformat())
    for w in windows:
        check();session=w['session'];assert session in plan['development_validation_sessions']
        assert not any((session,f) in excluded for f in range(w['start'],w['end']+1))
        r=next(r for r in records if r['sequence_id'].split('/')[-1]==session);assert r['split']=='real_development'
        cal=json.loads((ROOT/r['prepared_calibration']).read_text());existing=json.loads((ROOT/f'dataset/PyMouseStudio/sessions/{session}/source_calibration.json').read_text());assert cal==existing
        folder=dest/w['cohort']/session;folder.mkdir(parents=True);cams=sorted(r['cameras'])
        for c in cams:
            check();src=(ROOT/r['cameras'][c]['video']).resolve();assert src.is_relative_to((ROOT/'dataset/PyMouseStudioRaw').resolve())
            K=np.array(cal[c]['K']);D=np.array(cal[c]['distortion']);assert tuple(cal[c]['size_wh'])==(1080,1200)
            mx,my=cv2.initUndistortRectifyMap(K,D,None,K,(1080,1200),cv2.CV_32FC1)
            cap=cv2.VideoCapture(str(src));assert cap.isOpened();assert cap.get(cv2.CAP_PROP_FPS)==52
            cap.set(cv2.CAP_PROP_POS_FRAMES,w['start'])
            path=folder/f'{session}_cam{c}_{w["start"]:06d}_704x768.avi'
            writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'FFV1'),52,(704,768));assert writer.isOpened();expected={}
            for f in range(w['start'],w['end']+1):
                check();ok,bgr=cap.read();assert ok and bgr.shape==(1200,1080,3)
                rect=cv2.remap(bgr,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
                rgb=np.array(Image.fromarray(cv2.cvtColor(rect,cv2.COLOR_BGR2RGB)).resize((691,768),Image.Resampling.LANCZOS))
                canvas=np.zeros((768,704,3),np.uint8);canvas[:,6:697]=rgb;derived=cv2.cvtColor(canvas,cv2.COLOR_RGB2BGR)
                if (session,f,c) in entries:
                    e=entries[(session,f,c)];original=cv2.imread(str(ROOT/e['source']));teacher=cv2.imread(str(ROOT/e['derived']))
                    assert np.array_equal(rect,original),('rawremap_vs_manualPNG',session,f,c,float(np.abs(rect.astype(float)-original).max()))
                    assert np.array_equal(derived,teacher),('letterbox_vs_teacherPNG',session,f,c)
                    expected[f-w['start']]=hashlib.sha256(derived.tobytes()).hexdigest()
                writer.write(derived)
            cap.release();writer.release();verify=cv2.VideoCapture(str(path));assert verify.isOpened()
            for fi in range(w['end']-w['start']+1):
                ok,decoded=verify.read();assert ok
                if fi in expected:
                    assert hashlib.sha256(decoded.tobytes()).hexdigest()==expected[fi],('FFV1_roundtrip',session,c,fi)
                    qa.append(dict(cohort=w['cohort'],session=session,camera=c,frame=fi+w['start'],rawremap_equals_manualPNG=True,letterbox_equals_teacherPNG=True,FFV1_roundtrip_exact=True))
            assert not verify.read()[0];verify.release();print('EXACT_CLIP',w['cohort'],session,c,'manual_checks',len(expected),flush=True)
        manifest.append(dict(cohort=w['cohort'],session=session,cameras=cams,source_start=w['start'],source_end=w['end'],frames=w['end']-w['start']+1,video_folder=str(folder),fps=52,train_allowed=False))
        (dest/'progress.json').write_text(json.dumps(manifest,indent=2))
    assert len(qa)==84,len(qa)
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2));(dest/'verification.json').write_text(json.dumps(dict(passed=True,manual_frames_checked=len(qa),checks=qa,originals_modified=False,existing_rectified_videos_remapped=False),indent=2))
    write_state(stage='png_exact_clips_verified',pid=None,active_run=None,mode='idle_between_heartbeats',job_deadline_utc=None,latest_result=str(dest/'verification.json'),next_action='Run cached LP3seed peaks on newPNG-exactclips then identical frozen temporalidentity comparison.')

if __name__=='__main__':main()
