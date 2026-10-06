"""Matched weighted/EKS/gated video with sparse manual white-ring overlays."""
import argparse,json,shutil
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/teacher_temporal_cycle_20261001';BASE=ROOT/'dataset/PyMouseStudio/annotation_returns/20260930_full_v2'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--weighted',default='weighted_framewise_pilot_floor4');ap.add_argument('--eks',default='official_eks_17_31_73_floor4');ap.add_argument('--manifest',default='continuous_lp_seed17_kl_exactnorm/manifest.json');ap.add_argument('--candidate-label',default='Gated / no fill');args=ap.parse_args()
    dest=OUT/args.weighted/'review';dest.mkdir(exist_ok=True)
    clips=json.loads((OUT/args.manifest).read_text());manual=json.loads((BASE/'ingestion_manifest.json').read_text())
    for clip in clips:
        session=clip['session']
        with np.load(OUT/args.weighted/f'{session}.npz') as z:w=dict(z)
        with np.load(OUT/args.eks/f'{session}.npz') as z:e=dict(z)
        assert np.array_equal(w['source_frames'],e['source_frames'])
        groups={g['frame']:g for g in manual['frame_groups'] if g['session']==session and g['frame'] in w['source_frames']}
        gt={}
        for f,g in groups.items():
            with np.load(BASE/g['normalized_file']) as z:gt[f]=(g['cameras'],z['uv'][:,0,w['ids']-1])
        cams=w['cameras'].tolist();chosen=['A','F'];caps=[]
        for c in chosen:
            paths=list(Path(clip.get('video_folder',clip['folder'])).glob(f'{session}_cam{c}_*.avi'));assert len(paths)==1
            cap=cv2.VideoCapture(str(paths[0]));assert cap.isOpened();caps.append(cap)
        writer=cv2.VideoWriter(str(dest/f'{session}_weighted_eks_gated.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),26,(1056,864));assert writer.isOpened()
        for fi,frame in enumerate(w['source_frames']):
            canvas=np.zeros((864,1056,3),np.uint8)
            for ri,(c,cap) in enumerate(zip(chosen,caps)):
                ok,im=cap.read();assert ok;vi=cams.index(c)
                for ci,(name,uv) in enumerate([('Weighted robust / no time',w['uv_weighted']),('EKS / floor4 / s=1',e['uv_eks']),(args.candidate_label,w['uv_gated'])]):
                    panel=cv2.resize(im,(352,384))
                    def coord(v):return tuple(np.rint((v*np.array([691/1080,768/1200])+[6,0])*.5).astype(int))
                    if frame in gt and c in gt[frame][0]:
                        for v in gt[frame][1][gt[frame][0].index(c)]:
                            if np.isfinite(v).all():cv2.circle(panel,coord(v),5,(255,255,255),1)
                    for j,k in enumerate(w['ids']):
                        v=uv[vi,fi,j]
                        if not np.isfinite(v).all():continue
                        x,y=coord(v)
                        if not (0<=x<352 and 0<=y<384):continue
                        color=(255,170,70) if k<15 else ((90,240,100) if k%2 else (240,110,230))
                        cv2.circle(panel,(x,y),2,color,-1);cv2.putText(panel,str(k),(x+3,y-3),cv2.FONT_HERSHEY_SIMPLEX,.36,color,1)
                    x0,y0=ci*352,ri*432;canvas[y0+32:y0+416,x0:x0+352]=panel
                    cv2.putText(canvas,f'{c}: {name}',(x0+5,y0+22),cv2.FONT_HERSHEY_SIMPLEX,.44,(255,255,255),1)
                    cv2.putText(canvas,f'{session} f{frame} | white=manual | NOT GT',(x0+5,y0+431),cv2.FONT_HERSHEY_SIMPLEX,.35,(200,200,255),1)
            writer.write(canvas)
            if frame==min(groups):cv2.imwrite(str(dest/f'{session}_manual_frame.jpg'),canvas)
        writer.release()
        for cap in caps:cap.release()
    (dest/'README.txt').write_text('Half-speed video: source52fps, playback26fps. White rings exist only at manually labeled frames.\nWeighted robust framewise and EKS share a4px minimum observation standard deviation, but robust loss differs. Candidate panel: '+args.candidate_label+'. Rejected points remain missing. All projected3D outputs use all views for visualization, NOT LOO evaluation and NOT GT.\n')
    export=Path('/mnt/c/Users/55395/Downloads/MouseGeoForge-TeacherTemporal-20261001')/args.weighted
    shutil.copytree(dest,export,dirs_exist_ok=True);shutil.copy2(OUT/args.weighted/'summary.json',export/'summary.json');print(export)
if __name__=='__main__':main()
