"""LOO video panels: each displayed camera excluded from both inference routes."""
import argparse,json,shutil
from pathlib import Path
import cv2,numpy as np
from multipeak_geometry_cycle_20261002 import Geometry,ROOT,OUT,BASE

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='temporal_joint_states_v1');ap.add_argument('--peak-run',default='temporal_paw_peaks_v1');args=ap.parse_args()
    dest=OUT/args.run/'review';dest.mkdir(exist_ok=True)
    manifest=json.loads((OUT/args.peak_run/'manifest.json').read_text());manual=json.loads((BASE/'ingestion_manifest.json').read_text())
    for cohort,session in [('pilot','AZR2L2'),('pilot','FR2'),('stress1','AZR2L2'),('stress1','FR2')]:
        clip=next(r for r in manifest if r['cohort']==cohort and r['session']==session and r['seed']==17)
        selected=['A','F'];data=[];caps=[];projections=[]
        for c in selected:
            with np.load(OUT/args.run/f'{cohort}_{session}_held{c}.npz') as z:data.append(dict(z))
            paths=list(Path(clip['video_folder']).glob(f'{session}_cam{c}_*.avi'));assert len(paths)==1
            cap=cv2.VideoCapture(str(paths[0]));assert cap.isOpened();caps.append(cap)
            cal=json.loads((BASE/'calibration'/session/'rectified_calibration.json').read_text());geo=Geometry(cal,[c])
            projections.append({k:np.array([[geo.project(v,np.array([0]))[0][0] if np.isfinite(v).all() else [np.nan,np.nan] for v in frame] for frame in data[-1][k]]) for k in ['xyz_framewise_mm','xyz_temporal_mm']})
        frames=data[0]['source_frames'];groups={g['frame']:g for g in manual['frame_groups'] if g['session']==session and g['frame'] in frames};gt={}
        for f,g in groups.items():
            with np.load(BASE/g['normalized_file']) as z:gt[f]=(g['cameras'],z['uv'][:,0,14:18])
        writer=cv2.VideoWriter(str(dest/f'{cohort}_{session}_LOO_temporal.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),26,(704,864));assert writer.isOpened()
        for fi,frame in enumerate(frames):
            canvas=np.zeros((864,704,3),np.uint8)
            for ri,c in enumerate(selected):
                ok,im=caps[ri].read();assert ok
                def xy(v):return tuple(np.rint((v*np.array([691/1080,768/1200])+[6,0])*.5).astype(int))
                for ci,(key,label) in enumerate([('xyz_framewise_mm','Independent joint'),('xyz_temporal_mm','Temporal identity')]):
                    panel=cv2.resize(im,(352,384))
                    if frame in gt and c in gt[frame][0]:
                        for v in gt[frame][1][gt[frame][0].index(c)]:
                            if np.isfinite(v).all():cv2.circle(panel,xy(v),5,(255,255,255),1)
                    for j,k in enumerate([15,16,17,18]):
                        v=projections[ri][key][fi,j]
                        if not np.isfinite(v).all():continue
                        x,y=xy(v)
                        if not(0<=x<352 and 0<=y<384):continue
                        color=(90,240,100) if k%2 else (240,110,230)
                        cv2.circle(panel,(x,y),2,color,-1);cv2.putText(panel,str(k),(x+3,y-3),cv2.FONT_HERSHEY_SIMPLEX,.38,color,1)
                    ox=ci*352;oy=ri*432;canvas[oy+32:oy+416,ox:ox+352]=panel
                    cv2.putText(canvas,f'{c} HELD OUT: {label}',(ox+4,oy+20),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
                    cv2.putText(canvas,f'{cohort}/{session} f{frame} | white=manual',(ox+4,oy+431),cv2.FONT_HERSHEY_SIMPLEX,.35,(200,220,255),1)
            writer.write(canvas)
            if frame in gt:cv2.imwrite(str(dest/f'{cohort}_{session}_f{frame}_LOO.jpg'),canvas)
        writer.release()
        for cap in caps:cap.release()
    (dest/'README.txt').write_text('Half-speed52fps->26fps video. Each displayed camera A/F excluded from ALL candidate generation, geometry and temporal decisions for its own row. Both methods use identical perframehypotheses; temporal chooses candidates without position averaging/interpolation. White manual rings only at labeled frames. No dense3Dtruth; no accuracy/lag guarantee.\n')
    export=Path('/mnt/c/Users/55395/Downloads/MouseGeoForge-TeacherTemporal-20261001')/args.run
    shutil.copytree(dest,export,dirs_exist_ok=True);shutil.copy2(OUT/args.run/'summary.json',export/'summary.json');print(export)

if __name__=='__main__':main()
