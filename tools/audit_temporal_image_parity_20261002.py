"""Read-only compare labeled derived PNG to same/neighbor lossless clip frames."""
import json,shutil
from pathlib import Path
import cv2,numpy as np
from run_teacher_cycle_dlc_20261001 import ROOT,OUT

def main():
    dest=OUT/'temporal_joint_states_v1/image_parity_audit';dest.mkdir(exist_ok=True)
    m=json.loads((OUT/'dlc_dataset_manifest.json').read_text());manifest=json.loads((OUT/'temporal_paw_peaks_v1/manifest.json').read_text());rows=[]
    for clip in [r for r in manifest if r['seed']==17]:
        for c in clip['cameras']:
            paths=list(Path(clip['video_folder']).glob(f'{clip["session"]}_cam{c}_*.avi'));assert len(paths)==1
            cap=cv2.VideoCapture(str(paths[0]));assert cap.isOpened()
            entries=[e for e in m['entries'] if e['role']=='validation' and e['session']==clip['session'] and e['camera']==c and clip['source_start']<=e['frame']<=clip['source_end']]
            for e in entries:
                still=cv2.imread(str(ROOT/e['derived']));assert still.shape==(768,704,3);errs={};images={}
                for off in [-2,-1,0,1,2]:
                    pos=e['frame']-clip['source_start']+off
                    if not 0<=pos<clip['frames']:continue
                    cap.set(cv2.CAP_PROP_POS_FRAMES,pos);ok,im=cap.read();assert ok
                    errs[str(off)]=float(np.abs(im.astype(float)-still).mean());images[off]=im
                diff=np.abs(images[0].astype(float)-still)
                r=dict(cohort=clip['cohort'],session=e['session'],camera=c,frame=e['frame'],same_frame_mae=errs['0'],same_frame_max_diff=float(diff.max()),
                    fraction_pixels_exact=float(np.all(images[0]==still,axis=-1).mean()),neighbor_mae=errs,best_offset=int(min(errs,key=errs.get)),source=e['source'],derived=e['derived'])
                rows.append(r)
                if (e['session'],e['frame'],c) in [('FR2',779,'A'),('AZR2L2',129,'F')]:
                    panels=np.concatenate([still,images[0],np.clip(diff*8,0,255).astype(np.uint8)],axis=1)
                    panels=cv2.resize(panels,(1056,384));canvas=np.zeros((420,1056,3),np.uint8);canvas[36:]=panels
                    for j,label in enumerate(['Labeled derived PNG','Same-index video frame','Absolute difference x8']):cv2.putText(canvas,label,(j*352+5,24),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
                    cv2.imwrite(str(dest/f'{e["session"]}_f{e["frame"]}_cam{c}.jpg'),canvas)
            cap.release()
    report=dict(images=len(rows),same_index_best=sum(r['best_offset']==0 for r in rows),
        median_mae=float(np.median([r['same_frame_mae'] for r in rows])),max_mae=float(max(r['same_frame_mae'] for r in rows)),
        exactly_identical=sum(r['same_frame_max_diff']==0 for r in rows),
        nonzero_best_offset=[r for r in rows if r['best_offset']!=0],
        warning='Nearest-index pixel similarity is a diagnostic, not proof of synchronization. Different encoding/resizing can alter peak identities near decisions. No image/calibration/labels modified.')
    (dest/'summary.json').write_text(json.dumps(report,indent=2));(dest/'observations.json').write_text(json.dumps(rows,indent=2))
    export=Path('/mnt/c/Users/55395/Downloads/MouseGeoForge-TeacherTemporal-20261001/temporal_image_parity_audit');shutil.copytree(dest,export,dirs_exist_ok=True);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
