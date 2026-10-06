"""CPU-only review of fixed teacher versus geometric/temporal pseudo labels."""
import json
import argparse
import shutil
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/teacher_temporal_cycle_20261001'
DEST=OUT/'temporal_anipose_seed17_epoch30/review'

def main():
    global DEST
    parser=argparse.ArgumentParser();parser.add_argument('--input-run',default='continuous_seed17_epoch30');parser.add_argument('--output-run',default='temporal_anipose_seed17_epoch30');parser.add_argument('--teacher',default='DLC')
    args=parser.parse_args();assert '/' not in args.input_run and '/' not in args.output_run
    DEST=OUT/args.output_run/'review'
    DEST.mkdir(exist_ok=True)
    clips=json.loads((OUT/args.input_run/'manifest.json').read_text())
    methods=[('uv_raw','2D teacher'),('uv_framewise','Framewise 3D projected'),('uv_temporal1','Temporal 3D projected (s=1)')]
    is_eks=args.teacher.startswith('EKS')
    if is_eks:methods=[('uv_ensemble','3-teacher median'),('uv_framewise','Framewise 3D projected'),('uv_eks','EKS 3D projected (s=1)')]
    for clip in clips:
        session=clip['session']
        with np.load(DEST.parent/f'{session}.npz') as z:data=dict(z)
        cameras=data['cameras'].tolist()
        chosen=['A','F']
        caps=[]
        for c in chosen:
            matches=list(Path(clip.get('video_folder',clip['folder'])).glob(f'{session}_cam{c}_*.avi'))
            assert len(matches)==1,matches
            path=matches[0]
            assert path.is_file(),path
            cap=cv2.VideoCapture(str(path));assert cap.isOpened();caps.append(cap)
        writer=cv2.VideoWriter(str(DEST/f'{session}_comparison.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),26,(1056,864))
        assert writer.isOpened()
        for f in range(clip['frames']):
            canvas=np.zeros((864,1056,3),np.uint8)
            for row,(cam,cap) in enumerate(zip(chosen,caps)):
                ok,im=cap.read();assert ok
                vi=cameras.index(cam)
                for col,(key,label) in enumerate(methods):
                    panel=cv2.resize(im,(352,384))
                    for j,k in enumerate(data['ids']):
                        uv=data[key][vi,f,j]
                        if not np.isfinite(uv).all():continue
                        x,y=np.rint((uv*np.array([691/1080,768/1200])+[6,0])*.5).astype(int)
                        if not (0<=x<352 and 0<=y<384):continue
                        color=(255,160,70) if k<15 else ((80,240,100) if k%2 else (220,100,255))
                        cv2.circle(panel,(x,y),3,color,-1)
                        cv2.putText(panel,str(k),(x+4,y-3),cv2.FONT_HERSHEY_SIMPLEX,.36,(0,0,0),2)
                        cv2.putText(panel,str(k),(x+4,y-3),cv2.FONT_HERSHEY_SIMPLEX,.36,color,1)
                    x0,y0=col*352,row*432
                    canvas[y0+32:y0+416,x0:x0+352]=panel
                    cv2.putText(canvas,f'{cam}: {label}',(x0+6,y0+21),cv2.FONT_HERSHEY_SIMPLEX,.47,(255,255,255),1)
                    cv2.putText(canvas,f'{session} frame {data["source_frames"][f]} | NOT GT',(x0+6,y0+430),cv2.FONT_HERSHEY_SIMPLEX,.4,(180,210,255),1)
            writer.write(canvas)
            if f==100:cv2.imwrite(str(DEST/f'{session}_comparison.jpg'),canvas)
        writer.release()
        for cap in caps:cap.release()
    report=json.loads((DEST.parent/'summary.json').read_text())
    lines=[f'Fixed {args.teacher} teacher: temporal pilot review','Videos play at HALF source speed (26 vs 52 fps).',
           'IDs retain Core23 definitions. A/F shown; reconstruction uses all available views.',
           'EKS and framewise use the same unfiltered median observations; EKS additionally uses ensemble variance.' if is_eks else 'Raw teacher panel is unfiltered; 3D fitting uses confidence >=0.3.',
           'Projected 3D is not ground truth. Cameras/calibration/sync are not independently verified.',
           'Protected abdomen/tail and pending-review frames are excluded.',
           'Paired manual observations: '+str(report.get('paired_observations',report.get('paired_n')))]
    metrics=report.get('paired_metrics',report.get('paired'))
    for k,v in metrics.items():lines.append(f'{k}: median={v["median"]:.2f}px P90={v["p90"]:.2f}px')
    conclusion='Diagnostic only: review independent errors, coverage and action fidelity; not ground truth.' if is_eks else 'Temporal fitting does NOT improve this pilot. Do not publish pseudo-GT.'
    lines+=[conclusion,
            'Inspect teacher uncertainty and manual-only geometric compatibility before smoothing harder.']
    (DEST/'README.txt').write_text('\n'.join(lines))
    (OUT/f'{args.output_run}_review_decision.json').write_text(json.dumps(dict(
        baseline_review='dlc_baseline_review',temporal_review=str(DEST.relative_to(OUT)),
        conclusion=conclusion,
        next=['Inspect independent teacher seeds and manual-only geometric compatibility.',
              'Only reconsider ensemble temporal fitting after 2D semantic accuracy improves.'],
        pseudo_labels_released=False,locked_holdout_opened=False),indent=2))
    export=Path('/mnt/c/Users/55395/Downloads/MouseGeoForge-TeacherTemporal-20261001')
    export.mkdir(exist_ok=True)
    shutil.copytree(DEST,export/args.output_run,dirs_exist_ok=True)
    shutil.copytree(OUT/'dlc_baseline_review',export/'dlc_baseline_review',dirs_exist_ok=True)
    print(export,flush=True)

if __name__=='__main__':main()
