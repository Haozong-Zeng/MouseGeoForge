"""PNG-exact native LP +/-2 frame contexts; original inputs never changed."""
import datetime as dt,json,os,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import cv2,numpy as np
from PIL import Image
from run_teacher_cycle_dlc_20261001 import ROOT,OUT,write_state

def main():
    plan=json.loads((OUT/'plan.json').read_text());deadline=min(time.time()+1500,dt.datetime.fromisoformat(plan['deadline_utc']).timestamp())
    def check():assert time.time()<deadline and not (OUT/'STOP').exists()
    check();dest=OUT/'lp_context_dataset_v1';dest.mkdir(exist_ok=False);cv2.setNumThreads(1)
    m=json.loads((OUT/'dlc_dataset_manifest.json').read_text());excluded={(r['session'],r['frame']) for r in json.loads((ROOT/plan['exclusion_manifest']).read_text())['cases']};excluded.add(('AZR2L2',584))
    rawrecords=[json.loads(s) for s in (ROOT/'dataset/PyMouseStudioRaw/_validation_v2/sequences.jsonl').read_text().splitlines()]
    source={r['sequence_id'].split('/')[-1]:r for r in rawrecords if r['sequence_id'].split('/')[-1] in plan['training_sessions']+plan['development_validation_sessions']}
    entries=[];removed=[]
    for old_index,e in enumerate(m['entries']):
        if (e['session'],e['frame']) in excluded:removed.append(dict(old_index=old_index,reason='pending_or_SEM01'));continue
        s=e['session'];assert source[s]['split']=='real_development'
        context=[min(max(e['frame']+offset,0),source[s]['frame_count']-1) for offset in [-2,-1,0,1,2]]
        assert not any((s,f) in excluded for f in context),(s,e['frame'],context)
        entry=dict(e,old_index=old_index,context_frames=context,boundary_clamped=any(e['frame']+offset!=f for offset,f in zip([-2,-1,0,1,2],context)))
        entry['context_center']=(dest/'images'/s/f'cam{e["camera"]}'/f'frame_{e["frame"]:06d}.png').relative_to(ROOT).as_posix();entries.append(entry)
    jobs={}
    for e in entries:jobs.setdefault((e['session'],e['camera']),[]).append(e)
    write_state(stage='lp_context_dataset_preparing',pid=os.getpid(),active_run=dest.name,mode='CPU',job_deadline_utc=dt.datetime.fromtimestamp(deadline,dt.timezone.utc).isoformat())
    def job(key,group):
        check();s,c=key;r=source[s];cal=json.loads((ROOT/r['prepared_calibration']).read_text())[c]
        original=json.loads((ROOT/f'dataset/PyMouseStudio/sessions/{s}/source_calibration.json').read_text())[c];assert cal==original
        needed={f for e in group for f in e['context_frames']};centers={e['frame']:e for e in group}
        src=(ROOT/r['cameras'][c]['video']).resolve();assert src.is_relative_to((ROOT/'dataset/PyMouseStudioRaw').resolve())
        K=np.array(cal['K']);D=np.array(cal['distortion']);mx,my=cv2.initUndistortRectifyMap(K,D,None,K,(1080,1200),cv2.CV_32FC1)
        folder=dest/'images'/s/f'cam{c}';folder.mkdir(parents=True);cap=cv2.VideoCapture(str(src));assert cap.isOpened();assert cap.get(cv2.CAP_PROP_FPS)==52
        count=0;verified=0
        for f in range(max(needed)+1):
            check();ok,bgr=cap.read();assert ok
            if f not in needed:continue
            rect=cv2.remap(bgr,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
            if f in centers:assert np.array_equal(rect,cv2.imread(str(ROOT/centers[f]['source']))),(s,c,f,'sourcePNGmismatch')
            rgb=Image.fromarray(cv2.cvtColor(rect,cv2.COLOR_BGR2RGB)).resize((691,768),Image.Resampling.LANCZOS)
            canvas=Image.new('RGB',(704,768));canvas.paste(rgb,(6,0));canvas.save(folder/f'frame_{f:06d}.png',compress_level=3);count+=1
            if f in centers:
                assert np.array_equal(np.array(canvas),np.array(Image.open(ROOT/centers[f]['derived']).convert('RGB'))),(s,c,f,'teacherPNGmismatch');verified+=1
        cap.release();assert count==len(needed)
        return dict(session=s,camera=c,unique_context_images=count,center_exact_checks=verified)
    results=[]
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures=[executor.submit(job,key,group) for key,group in sorted(jobs.items())]
        for f in as_completed(futures):
            r=f.result();results.append(r);(dest/'progress.json').write_text(json.dumps(results,indent=2));print('CONTEXT_CAMERA_DONE',r,flush=True)
    for e in entries:
        center=ROOT/e['context_center']
        for f in e['context_frames']:assert center.with_name(f'frame_{f:06d}.png').exists()
    manifest=dict(entries=entries,ids_1based=m['ids_1based'],names=m['names'],affine_source_to_input=m['affine_source_to_input'],
        train_indices=[i for i,e in enumerate(entries) if e['role']=='train'],validation_indices=[i for i,e in enumerate(entries) if e['role']=='validation'],
        removed=removed,center_exact_checks=sum(r['center_exact_checks'] for r in results),context_images=sum(r['unique_context_images'] for r in results),
        boundary_centers=sum(e['boundary_clamped'] for e in entries),frames_per_context=5,raw_rectification_applied_once=True,locked_holdout_opened=False,pseudo_labels_used=False)
    assert manifest['center_exact_checks']==len(entries) and len(manifest['train_indices'])==1043 and len(manifest['validation_indices'])==288
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2));print('CONTEXT_DATASET_READY',len(entries),manifest['context_images'],flush=True)
    write_state(stage='lp_context_dataset_ready',pid=None,active_run=None,mode='idle_between_heartbeats',job_deadline_utc=None,latest_result=str(dest/'manifest.json'),next_action='Native LP MHCRNN context smoke/memory/missinggradient test, then one bounded seed17teacher if checks pass.')

if __name__=='__main__':main()
