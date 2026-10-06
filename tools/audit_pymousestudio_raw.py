"""Inventory sources and prepare explicit, non-training real validation access."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'output/pymousestudio_audit_deps'))
import collections
import json
import re
import tomllib
import cv2
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mousegeoforge.raw_validation import primitive_column_metadata,group_for_session,RawVideoValidation,undistort_points_compat

RAW=ROOT/'dataset/PyMouseStudioRaw'
OUT=RAW/'_validation_v2'


def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def video_metadata(path):
    c=cv2.VideoCapture(str(path))
    if not c.isOpened():raise OSError(path)
    result=dict(frames=int(c.get(cv2.CAP_PROP_FRAME_COUNT)),fps=c.get(cv2.CAP_PROP_FPS),
        size_wh=[int(c.get(cv2.CAP_PROP_FRAME_WIDTH)),int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))])
    c.release();return result


def read_keypoints(path):
    with h5py.File(path) as f:
        table=f['df_with_missing/table'];labels=primitive_column_metadata(table.attrs['values_block_0_kind'])
        array=table[...]
    if len(labels)!=93 or not all(len(x)==3 for x in labels):raise ValueError('Unexpected DLC schema')
    names=[]
    for i in range(0,93,3):
        block=labels[i:i+3]
        if [x[2] for x in block]!=['x','y','likelihood'] or len({x[1] for x in block})!=1:raise ValueError('DLC column order')
        names.append(block[0][1])
    values=array['values_block_0'].reshape(-1,31,3)
    return dict(frame_indices=array['index'],uv=values[...,:2],likelihood=values[...,2],source_names=np.array(names),scorer=labels[0][0])


def multiview_lens_check(record,calibration,arrays):
    # No model inference; DLC-based leave-one-view-out reprojection diagnostic.
    errors={'distorted_input':[],'already_rectified_input':[]}
    selected=('nose','eye_l','eye_r','ear_end_l','ear_end_r','fore_paw_l','fore_paw_r','hind_paw_l','hind_paw_r')
    cameras=list(arrays)
    for frame in np.linspace(0,record['frame_count']-1,9,dtype=int):
        for part in selected:
            observations=[]
            for name in cameras:
                a=arrays[name];idx=int(np.where(a['source_names']==part)[0][0]);uv=a['uv'][frame,idx];c=calibration[name]
                if a['likelihood'][frame,idx]<.9 or not np.isfinite(uv).all():continue
                if not (0<=uv[0]<c['size_wh'][0] and 0<=uv[1]<c['size_wh'][1]):continue
                observations.append((name,uv))
            if len(observations)<4:continue
            for hypothesis in errors:
                normalized={}
                for name,uv in observations:
                    c=calibration[name];K=np.asarray(c['K']);D=np.asarray(c['distortion']) if hypothesis=='distorted_input' else np.zeros(5)
                    normalized[name]=undistort_points_compat(uv.astype(float).reshape(1,1,2),K,D)[0,0]
                for held,observed in observations:
                    rows=[]
                    for name,_ in observations:
                        if name==held:continue
                        P=np.asarray(calibration[name]['world_to_camera_mm'])[:3];u,v=normalized[name]
                        for row in (u*P[2]-P[0],v*P[2]-P[1]):rows.append(row/np.linalg.norm(row[:3]))
                    _,_,vh=np.linalg.svd(np.asarray(rows));X=vh[-1,:3]/vh[-1,3]
                    c=calibration[held];E=np.asarray(c['world_to_camera_mm']);D=np.asarray(c['distortion']) if hypothesis=='distorted_input' else np.zeros(5)
                    if (E[:3,:3]@X+E[:3,3])[2]<=0:continue
                    projected,_=cv2.projectPoints(X.reshape(1,1,3),np.asarray(c['rotation_vector']),E[:3,3],np.asarray(c['K']),D)
                    errors[hypothesis].append(float(np.linalg.norm(projected[0,0]-observed)))
    return {k:dict(count=len(v),median_px=float(np.median(v)) if v else None,p90_px=float(np.percentile(v,90)) if v else None)
        for k,v in errors.items()}


def run():
    if OUT.exists():raise FileExistsError('Preserve prior validation preparation')
    imported=json.loads((RAW/'_metadata/import.json').read_text());assert imported['state']=='completed'
    OUT.mkdir();records=[];reports=[];calib_hashes=[];all_names=None
    for imported_session in imported['archives']:
        session=imported_session['session'];path=RAW/session
        calibration=tomllib.loads((path/'calibration.toml').read_text());prepared={}
        for _,c in calibration.items():
            if 'matrix' not in c:continue
            R,_=cv2.Rodrigues(np.asarray(c['rotation']));E=np.eye(4);E[:3,:3]=R;E[:3,3]=c['translation']
            prepared[c['name']]=dict(K=c['matrix'],distortion=c['distortions'],size_wh=c['size'],
                rotation_vector=c['rotation'],world_to_camera_mm=E.tolist(),physical_role='unverified',
                units_evidence='calibration metadata 3Derror mm; source configs pokem unit mm when present')
        calpath=OUT/session/'calibration.json';write(calpath,prepared)
        group=group_for_session(session);split='real_locked_holdout' if group in ('G','U') else 'real_development'
        record=dict(sequence_id=f'pymousestudio_raw/{session}',source_group=f'pymousestudio_20241122/family_{group}',
            split=split,train_allowed=False,prepared_calibration=str(calpath.relative_to(ROOT)),cameras={},
            annotation_kind='DLC_prediction_and_unverified_foreground_mask',manual_gt_available=False,
            depth_available=False,keypoints3d_available=False,instance_count_verified=False,
            synchronization='equal frame indices assumed; no hardware timestamps/offset metadata',
            lens_input_state='unconfirmed; retain raw and make correction explicit',fps=None,frame_count=None)
        audit=dict(session=session,split=split,videos={},keypoints={},calibration_metadata=calibration.get('metadata',{}),
            missing=[],numpy_files=[]);arrays={};video_counts=[];fps=[]
        for p in sorted((path/'videos').glob('*.mp4')):
            name=re.search(r'-cam([A-Z])\.mp4$',p.name)[1];meta=video_metadata(p)
            audit['videos'][name]=meta;video_counts.append(meta['frames']);fps.append(meta['fps'])
            if meta['size_wh']!=prepared[name]['size_wh']:raise ValueError(f'{session}/{name} calibration size mismatch')
            masks=list((path/'masks').glob(f'*-cam{name}_Mask.mp4'));kp=list((path/'keypoints2d').glob(f'*-cam{name}DLC*.h5'))
            if len(masks)>1 or len(kp)>1:raise ValueError('Ambiguous camera assets')
            item=dict(video=str(p.relative_to(ROOT)),mask=str(masks[0].relative_to(ROOT)) if masks else None,keypoints=None)
            if masks:
                maskmeta=video_metadata(masks[0]);audit['videos'][name]['mask']=maskmeta
                if maskmeta!={k:meta[k] for k in ('frames','fps','size_wh')}:raise ValueError('Mask/video timing/size mismatch')
            else:audit['missing'].append(f'camera{name}:mask')
            if kp:
                a=read_keypoints(kp[0]);arrays[name]=a
                if len(a['uv'])!=meta['frames'] or not np.array_equal(a['frame_indices'],np.arange(meta['frames'])):raise ValueError('Keypoint frame indices do not match video')
                names=a['source_names'].tolist()
                if all_names is None:all_names=names
                elif all_names!=names:raise ValueError('Inconsistent landmark names')
                target=OUT/session/f'keypoints_cam{name}.npz';np.savez_compressed(target,**a)
                item['keypoints']=str(target.relative_to(ROOT))
                audit['keypoints'][name]=dict(frames=len(a['uv']),landmarks=len(names),scorer=a['scorer'],
                    finite_xy_fraction=float(np.isfinite(a['uv']).all(-1).mean()),confidence_ge_09=float((a['likelihood']>=.9).mean()))
            else:audit['missing'].append(f'camera{name}:keypoints')
            record['cameras'][name]=item
        if len(set(video_counts))!=1 or len(set(fps))!=1:raise ValueError('Cross-camera timing metadata mismatch')
        record['frame_count']=video_counts[0];record['fps']=fps[0]
        record['diagnostic_frame_indices']=np.linspace(0,video_counts[0]-1,8,dtype=int).tolist()
        record['temporal_segment']=dict(start=video_counts[0]//2,stop=min(video_counts[0],video_counts[0]//2+32),stride=1)
        audit['missing'].extend(f'camera{c}:video' for c in prepared if c not in record['cameras'])
        for p in sorted(path.rglob('*.npy')):
            try:
                a=np.load(p,allow_pickle=False,mmap_mode='r');audit['numpy_files'].append(dict(path=str(p.relative_to(RAW)),shape=list(a.shape),dtype=str(a.dtype)))
            except ValueError:audit['numpy_files'].append(dict(path=str(p.relative_to(RAW)),object_payload_not_loaded=True))
        # Validation content inspection is limited to development sources; holdout stays sealed.
        if split=='real_development':audit['lens_reprojection']=multiview_lens_check(record,prepared,arrays)
        reports.append(audit);records.append(record);print('AUDITED',session,len(record['cameras']),flush=True)
    manifest=OUT/'sequences.jsonl';manifest.write_text(''.join(json.dumps(r)+'\n' for r in records))
    write(OUT/'inventory.json',dict(sessions=reports,source_names31=all_names,archive_count=len(records),
        archive_uncompressed_bytes=imported['uncompressed_bytes'],originals_unchanged=True,
        external_pickle_files_loaded=False,h5_read='numeric dataset and whitelist primitive metadata interpreter',
        source_group_policy='Conservative family prefix; not verified animal identity. No real sources assigned to training.',
        holdout_groups=['G','U'],official_gt=False,model_inference_runs=0))
    reader=RawVideoValidation(manifest,ROOT)
    sid='pymousestudio_raw/AZL3';sample=next(r for r in records if r['sequence_id']==sid)
    frames=[0,sample['frame_count']//2,sample['frame_count']-1]
    fig,axes=plt.subplots(3,7,figsize=(17,9),layout='constrained')
    for row,frame in enumerate(frames):
        x=reader.read(sid,frame,undistort=False)
        for col,name in enumerate(x['camera_names']):
            ax=axes[row,col];ax.imshow(x['images'][col]);ax.set_title(f'AZL3 {name} f{frame}');ax.axis('off')
    fig.savefig(OUT/'AZL3_raw_multiview.png',dpi=110);plt.close(fig)
    raw=reader.read(sid,frames[1],cameras=['A','B','D'],undistort=False)
    und=reader.read(sid,frames[1],cameras=['A','B','D'],undistort=True)
    fig,axes=plt.subplots(3,2,figsize=(9,13),layout='constrained')
    for row,name in enumerate(raw['camera_names']):
        for col,x in enumerate((raw,und)):
            ax=axes[row,col];ax.imshow(x['images'][row]);uv=x['keypoints2d_source31'][row,0];valid=np.isfinite(uv).all(-1)&(x['keypoint_confidence'][row,0]>=.9)
            ax.scatter(uv[valid,0],uv[valid,1],s=9,c='lime');ax.set_title(f'{name}: '+('raw' if col==0 else 'lens-corrected, sameK'));ax.axis('off')
    fig.savefig(OUT/'AZL3_lens_comparison.png',dpi=110);plt.close(fig)
    print('COMPLETE',len(records),manifest,flush=True)


if __name__=='__main__':run()
