"""Frozen temporal diagnostics: raw outputs, sparse manual evaluation, visual clips."""
import json
from pathlib import Path
import shutil
import subprocess
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from bottom_anchor_common_20261002 import ROOT,OUT,budget,write_state
from audit_extended_teachers_20261002 import metrics

COLORS={1:'#ff5555',6:'#ffee40',7:'#bd7bff',15:'#ff65da',16:'#20dfff',17:'#ff9a30',18:'#70ff70'}


def main():
    plan,_,check=budget(900);check()
    run=OUT/'quadruped_temporal_frozen_v1';records=json.loads((run/'manifest.json').read_text());assert len(records)==8
    dest=OUT/'quadruped_temporal_review_v1';dest.mkdir(exist_ok=False)
    dm=json.loads((OUT/'superanimal_quadruped_hrnet_v1/dataset_manifest.json').read_text())
    entries=[dm['entries'][i] for i in dm['validation_indices']];lookup={(e['session'],e['camera'],int(e['frame'])):i for i,e in enumerate(entries)}
    A=np.array(dm['affine_source_to_input']);scale=np.diag(A)[:2];ids=dm['ids_1based']
    with np.load(OUT/'quadruped_epoch110_review_v1/predictions_source_pixels.npz') as z:gt=z['gt'].copy()
    arrays={}
    for r in records:
        assert r['clip']['session'] in plan['development_sessions']
        with np.load(run/r['file']) as z:arrays[(r['seed'],r['clip']['cohort'],r['clip']['session'])]={k:z[k].copy() for k in z.files}
    sparse={17:[],29:[]};sparse_indices=[];temporal=[];cases=[];videos=[];storyboards=[]
    font=ImageFont.truetype('/mnt/c/Windows/Fonts/arial.ttf',18)
    small=ImageFont.truetype('/mnt/c/Windows/Fonts/arial.ttf',13)
    for rec in [r for r in records if r['seed']==17]:
        check();clip=rec['clip'];cohort=clip['cohort'];session=clip['session'];cam_names=clip['cameras']
        frame_ids=np.arange(clip['source_start'],clip['source_end']+1)
        for vi,camera in enumerate(cam_names):
            for fi,frame in enumerate(frame_ids):
                key=(session,camera,int(frame))
                if key not in lookup:continue
                idx=lookup[key];sparse_indices.append(idx)
                for seed in [17,29]:
                    uv=arrays[(seed,cohort,session)]['uv'][vi];sparse[seed].append(uv[fi])
                    err=np.linalg.norm(uv[fi]-gt[idx],axis=-1)
                    for j in np.flatnonzero(err>50):
                        window=uv[max(0,fi-2):min(208,fi+3),j]
                        # Stability relative to center prediction, NOT neighboring-frame ground truth.
                        cases.append(dict(seed=seed,session=session,cohort=cohort,camera=camera,frame=int(frame),id=ids[j],
                            center_error_source=float(err[j]),neighbor_radius2_max_motion_from_prediction=float(np.max(np.linalg.norm(window-uv[fi,j],axis=-1)))))
            for seed in [17,29]:
                uv=arrays[(seed,cohort,session)]['uv'][vi]
                step=np.linalg.norm(np.diff(uv,axis=0),axis=-1)
                temporal.append(dict(seed=seed,session=session,cohort=cohort,camera=camera,
                    raw_step_p50_by_id={str(k):float(np.median(step[:,j])) for j,k in enumerate(ids)},
                    raw_steps_over100_by_id={str(k):int(np.sum(step[:,j]>100)) for j,k in enumerate(ids)},
                    caveat='Raw frame displacement is NOT error; true motion and unannotated outputs included.'))
        # Render bottom A and side F for both seeds; all cameras remain in numeric cache.
        selected=['A','F'];images={}
        for camera in selected:
            paths=list(Path(clip['video_folder']).glob(f'{session}_cam{camera}_*.avi'));assert len(paths)==1
            cap=cv2.VideoCapture(str(paths[0]));frames=[]
            for _ in range(208):ok,im=cap.read();assert ok;frames.append(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
            cap.release();images[camera]=frames
        panels={c:[] for c in selected}
        filename=f'{cohort}_{session}_A_F_both_seeds_4xslow.mp4'
        proc=subprocess.Popen(['ffmpeg','-v','error','-nostdin','-n','-f','rawvideo','-pix_fmt','rgb24','-s','704x820','-r','13','-i','pipe:0',
            '-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(dest/filename)],stdin=subprocess.PIPE)
        try:
            for fi,frame in enumerate(frame_ids):
                check();canvas=Image.new('RGB',(704,820),'#161616')
                for row,seed in enumerate([17,29]):
                    data=arrays[(seed,cohort,session)]
                    for col,camera in enumerate(selected):
                        vi=cam_names.index(camera);im=Image.fromarray(images[camera][fi]).resize((352,384),Image.Resampling.LANCZOS);d=ImageDraw.Draw(im)
                        xy=(data['uv'][vi,fi]*scale+A[:2,2])/2
                        for j,k in enumerate(ids):
                            x,y=xy[j];d.ellipse((x-3,y-3,x+3,y+3),outline=COLORS[k],width=2)
                            tx=min(max(x+5,0),330);ty=min(max(y-15,0),363)
                            d.text((tx,ty),str(k),font=small,fill=COLORS[k],stroke_width=1,stroke_fill='black')
                        key=(session,camera,int(frame))
                        if key in lookup:
                            target=(gt[lookup[key]]*scale+A[:2,2])/2
                            for j in range(7):
                                if np.isfinite(target[j]).all():
                                    x,y=target[j];d.rectangle((x-5,y-5,x+5,y+5),outline='white',width=1)
                        canvas.paste(im,(col*352,row*410+26))
                        ImageDraw.Draw(canvas).text((col*352+4,row*410+3),f'S{seed} {session} {camera} f{frame}',font=font,fill='white')
                proc.stdin.write(np.asarray(canvas).tobytes())
                if fi in [0,29,65,94,130,199]:
                    for col,camera in enumerate(selected):
                        panels[camera].append(canvas.crop((col*352,0,(col+1)*352,820)))
            proc.stdin.close();assert proc.wait(timeout=45)==0
        except BaseException:
            proc.kill();proc.wait();raise
        for camera in selected:
            board=Image.new('RGB',(352*6,820),'#161616')
            for j,panel in enumerate(panels[camera]):board.paste(panel,(352*j,0))
            file=f'{cohort}_{session}_cam{camera}_storyboard.jpg';board.save(dest/file,quality=95);storyboards.append(file)
        videos.append(filename);print('RENDERED',filename,flush=True)
    assert len(sparse_indices)==len(set(sparse_indices))
    cohort_entries=[entries[i] for i in sparse_indices];target=gt[sparse_indices]
    summary_metrics={str(seed):metrics(np.stack(pred),target,cohort_entries,ids,scale) for seed,pred in sparse.items()}
    parities=[p for r in records for p in r['manual_frame_parity']]
    max_parity=max(p['all_channel_max_source_px'] for p in parities)
    report=dict(manual_images=len(sparse_indices),metrics=summary_metrics,source_input_scale=scale.tolist(),
        temporal_sparse_allchannel_max_distance_source_px=max_parity,image_parity_all=all(p['pixels_exact'] for p in parities),
        continuous_frames_per_clip=208,original_fps=52,display_fps=13,trained_or_adapted=False,
        temporal_diagnostics=temporal,center_errors_over50=cases,videos=videos,storyboards=storyboards,
        limitations=['No dense manual GT: temporal smoothness/stability is not accuracy.',
            'White squares appear only on manually labeled frames; colored circles show ALL raw7 outputs, including unknown/missing labels.',
            'A/F video visualization covers bottom and one side; all available cameras included in caches and sparse metrics.',
            'Same reused development videos, not independent test. No geometry correction, pseudo-GT or label changes.'],
        next='Visually inspect all8 temporal storyboards; distinguish persistent shared errors from abrupt switches, then decide whether any further label-preserving diagnostic is justified.',
        locked_holdouts_accessed=False,pseudo_gt_released=False)
    (dest/'summary.json').write_text(json.dumps(report,indent=2))
    page='<meta charset="utf-8"><h1>Quadruped冻结模型连续短片</h1><p>四段各208帧，原始52fps，以13fps慢放。视频左底视A、右侧视F，上seed17、下seed29。彩圈为原始预测；白方框只在人工标注帧出现。未标注点仍会显示，不能视为正确或可见。平滑不代表正确。</p>'
    page+=''.join(f'<h2>{p}</h2><video controls loop width="704" src="{p}"></video>' for p in videos)
    page+=''.join(f'<h2>{p}</h2><img style="max-width:100%" src="{p}">' for p in storyboards)
    (dest/'index.html').write_text(page+'<pre>'+json.dumps(report,indent=2)+'</pre>')
    export=Path('/mnt/c/Users/55395/Downloads/MouseGeoForge-BottomAnchor-20261002')/dest.name;shutil.copytree(dest,export)
    write_state(stage='quadruped_temporal_review_ready',latest_decision=str(dest/'summary.json'),latest_training_visual_directory=str(export),next_action=report['next'])
    print(json.dumps(dict(export=str(export),manual_images=len(sparse_indices),max_parity=max_parity,
        metrics={k:v['all'] for k,v in summary_metrics.items()},large_center_errors=len(cases)),indent=2))


if __name__=='__main__':main()
