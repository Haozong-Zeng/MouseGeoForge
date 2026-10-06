"""Rectify all raw videos under the explicit user assumption of distorted input.

Raw sources are never modified. Export same-K, same-size RGB movies, lossless
binary-mask movies, transformed source31 predictions, annotation PNGs and pilot
PNGs. Completed per-camera markers support safe restart of this one version.
"""
from pathlib import Path
import sys
import json
import hashlib
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mousegeoforge.raw_validation import rectify_points
DEST=ROOT/'dataset/PyMouseStudio'
cv2.setNumThreads(1)


def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(path)


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def encoder(path,gray=False):
    args=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt',
          'gray' if gray else 'bgr24','-s','1080x1200','-r','52','-i','pipe:0','-an']
    if gray:args+=['-c:v','ffv1','-level','3','-threads','2','-pix_fmt','gray']
    else:args+=['-c:v','libx264','-preset','veryfast','-crf','16','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart']
    return subprocess.Popen(args+[str(path)],stdin=subprocess.PIPE,stderr=subprocess.PIPE)


def export_camera(job):
    r,name,cal=job;s=r['sequence_id'].split('/')[-1];c=r['cameras'][name]
    dst=DEST/'sessions'/s;marker=dst/'completed'/f'cam{name}.json'
    if marker.exists():return json.loads(marker.read_text())
    K=np.asarray(cal['K']);D=np.asarray(cal['distortion']);w,h=cal['size_wh']
    assert (w,h)==(1080,1200)
    mx,my=cv2.initUndistortRectifyMap(K,D,None,K,(w,h),cv2.CV_32FC1)
    indices=set(np.linspace(0,r['frame_count']-1,25,dtype=int).tolist())
    pilot=set(range(780,812)) if s=='AZL3' else set()
    video=dst/'videos'/f'cam{name}.mp4';video.parent.mkdir(parents=True,exist_ok=True)
    temp=video.with_name(f'cam{name}.partial.mp4')
    enc=encoder(temp);cap=cv2.VideoCapture(str(ROOT/c['video']))
    maskcap=cv2.VideoCapture(str(ROOT/c['mask'])) if c['mask'] else None
    maskfile=dst/'masks_video'/f'cam{name}.mkv';masktmp=maskfile.with_name(f'cam{name}.partial.mkv')
    if maskcap is not None:
        maskfile.parent.mkdir(exist_ok=True);maskenc=encoder(masktmp,True)
    else:maskenc=None
    samples=[]
    try:
        for frame in range(r['frame_count']):
            ok,bgr=cap.read();assert ok and bgr.shape==(h,w,3),(s,name,frame)
            rgb=cv2.remap(bgr,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
            enc.stdin.write(rgb.tobytes())
            if frame in indices:
                rel=Path('annotation_pack_v2/real_images')/r['split']/s/f'frame_{frame:06d}'/f'cam{name}.png'
                p=DEST/rel;p.parent.mkdir(parents=True,exist_ok=True)
                assert cv2.imwrite(str(p),rgb,[cv2.IMWRITE_PNG_COMPRESSION,3])
                assert np.array_equal(cv2.imread(str(p)),rgb)
                samples.append(dict(image=rel.as_posix(),session=s,frame=frame,camera=name,
                                    split=r['split'],source_group=r['source_group'],width=w,height=h,
                                    timestamp_seconds=frame/52,sha256=digest(p),train_allowed=False,
                                    source_video=c['video'],lens_correction_applied=True,model_predictions_shown=False))
            if frame in pilot:
                p=dst/'pilot_rgb'/f'{frame:06d}'/f'cam{name}.png';p.parent.mkdir(parents=True,exist_ok=True)
                assert cv2.imwrite(str(p),rgb,[cv2.IMWRITE_PNG_COMPRESSION,3])
            if maskcap is not None:
                ok,b=maskcap.read();assert ok
                mask=np.uint8(cv2.cvtColor(b,cv2.COLOR_BGR2GRAY)>127)*255
                mask=cv2.remap(mask,mx,my,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
                maskenc.stdin.write(mask.tobytes())
                if frame in pilot:
                    p=dst/'pilot_input_masks'/f'{frame:06d}'/f'cam{name}.png';p.parent.mkdir(parents=True,exist_ok=True)
                    assert cv2.imwrite(str(p),mask)
        assert not cap.read()[0]
        if maskcap is not None:assert not maskcap.read()[0]
    finally:
        cap.release()
        if maskcap is not None:maskcap.release()
        for e in [enc,maskenc]:
            if e is not None:
                e.stdin.close();error=e.stderr.read().decode();code=e.wait()
                if code:raise RuntimeError(error)
    temp.replace(video)
    if maskenc is not None:masktmp.replace(maskfile)
    for p in [video]+([maskfile] if maskenc else []):
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
             '-show_entries','stream=width,height,nb_read_frames,r_frame_rate','-of','json',str(p)]))['streams'][0]
        assert int(probe['nb_read_frames'])==1560 and probe['width']==w and probe['height']==h,probe
        assert probe['r_frame_rate']=='52/1',probe
    if c['keypoints']:
        with np.load(ROOT/c['keypoints'],allow_pickle=False) as a:arr={k:a[k] for k in a.files}
        arr['uv']=rectify_points(arr['uv'],K,D).astype(np.float32)
        p=dst/'keypoints_source31'/f'cam{name}.npz';p.parent.mkdir(exist_ok=True)
        np.savez_compressed(p,**arr)
    valid=((mx>=0)&(mx<w-1)&(my>=0)&(my<h-1)).astype(np.uint8)*255
    p=dst/'valid_pixels'/f'cam{name}.png';p.parent.mkdir(exist_ok=True);assert cv2.imwrite(str(p),valid)
    record=dict(session=s,camera=name,frames=1560,fps=52,source_sha256=digest(ROOT/c['video']),
                rgb=str(video.relative_to(DEST)),rgb_sha256=digest(video),
                mask=str(maskfile.relative_to(DEST)) if maskenc else None,
                keypoints=f'sessions/{s}/keypoints_source31/cam{name}.npz' if c['keypoints'] else None,
                samples=samples,completed=True)
    write(marker,record)
    print(f'COMPLETE {s} cam{name}',flush=True)
    return record


def main():
    source=ROOT/'dataset/PyMouseStudioRaw/_validation_v2/sequences.jsonl'
    records=[json.loads(x) for x in source.read_text().splitlines()]
    config=dict(version='pymousestudio_rectified_v1',source_manifest_sha256=digest(source),
                assumption='USER: input video not lens-corrected; apply exactly once',
                output_size_wh=[1080,1200],same_K=True,new_distortion=[0]*5,fps=52,
                crop=False,resize=False,RGB_interpolation='bilinear',mask_interpolation='nearest',
                RGB_video_codec='H264 CRF16 yuv420p (lossy preview/archive); PNGs from direct raw remap',
                mask_video_codec='FFV1 gray lossless binary',annotation_time_groups=300,
                source_keypoints='DLC predictions transformed with same camera; NOT manual GT',
                original_videos_modified=False,train_allowed=False)
    DEST.mkdir(exist_ok=True)
    if (DEST/'processing_config.json').exists():assert json.loads((DEST/'processing_config.json').read_text())==config
    else:write(DEST/'processing_config.json',config)
    jobs=[];seq=[]
    for r in records:
        s=r['sequence_id'].split('/')[-1];cal=json.loads((ROOT/r['prepared_calibration']).read_text())
        new={}
        for n,c in cal.items():
            new[n]={**c,'source_distortion':c['distortion'],'distortion':[0.]*5,
                    'lens_correction_applied':True,'input_assumed_distorted':True}
        write(DEST/'sessions'/s/'calibration.json',new)
        write(DEST/'sessions'/s/'source_calibration.json',cal)
        for n in r['cameras']:jobs.append((r,n,cal[n]))
        seq.append(dict(session=s,source_group=r['source_group'],split=r['split'],train_allowed=False,
                        cameras=list(r['cameras']),missing_camera_videos=sorted(set(cal)-set(r['cameras'])),
                        fps=52,frame_count=1560,annotation_frame_indices=np.linspace(0,1559,25,dtype=int).tolist()))
    write(DEST/'sequences.json',seq)
    results=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures=[pool.submit(export_camera,j) for j in jobs]
        for f in as_completed(futures):
            results.append(f.result())
            write(DEST/'export_status.json',dict(state='running',completed_cameras=len(results),total_cameras=len(jobs)))
    samples=sorted([s for r in results for s in r['samples']],key=lambda x:(x['session'],x['frame'],x['camera']))
    assert len(samples)==1950
    write(DEST/'annotation_pack_v2/manifest.json',samples)
    write(DEST/'export_status.json',dict(state='complete',completed_cameras=len(results),total_cameras=len(jobs),
                                         annotation_images=len(samples),annotation_time_groups=300))
    print('EXPORT COMPLETE',flush=True)


if __name__=='__main__':main()
