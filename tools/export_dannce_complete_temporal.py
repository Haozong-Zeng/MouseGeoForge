"""Export independent rectified images, sync, calibration and sparse GT for DANNCE."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import sys
import time
import cv2
import numpy as np
from scipy.io import loadmat

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / 'BlenderScripts'))
from convert_dannce_mouse_to_blenderpymouse import (load_session, find_videos,
    pinhole_intrinsic, undistort_maps, write_camera, write_pair_file)
from core23_temporal_format import write_json_atomic, write_jsonl_atomic, save_npz_atomic


def pilot_annotations(session):
    """Attach completed pilots without mistaking them for complete dense depth."""
    frames,pilots={},[]
    for mp in sorted((session/'pseudo_depth_pilot').glob('*/metadata.json')):
        pm=json.loads(mp.read_text())
        rows=[json.loads(line) for line in (mp.parent/'frames.jsonl').read_text().splitlines()]
        assert len(rows)==pm['frame_count']
        for row in rows:
            fid=row['source_frame_id']
            if fid in frames:
                raise ValueError(f'Overlapping pilot source frame {fid}')
            frames[fid]={k:[str((mp.parent/p).relative_to(session)) for p in row[k]]
                         for k in ('depth','mask','confidence')}
            frames[fid].update(depth_is_pseudo_label=True,depth_pfm_flip_y=pm['pfm_flip_y'])
        pilots.append({'path':str(mp.parent.relative_to(session)),
            'source_frame_range':pm['source_frame_range'],'frame_count':pm['frame_count'],
            'method':pm['method'],'depth_unit':'mm','pfm_flip_y':pm['pfm_flip_y']})
    from dannce_pseudo_manifest import full_annotations
    full,full_meta=full_annotations(session)
    frames.update(full)  # Full validated chunks take precedence; old pilot files remain intact.
    return frames,{'pseudo_depth_pilots':pilots,'pseudo_depth_full':full_meta,
        'depth_available_frame_count':len(frames),
        'depth_complete':full_meta['complete'],'pseudo_depth_training_default':False}


def prepare(mouse,source_step=4):
    raw = BASE / f'dataset/DANNCE/raw/markerless_mouse_{mouse}'
    root = BASE / f'dataset/DANNCE-MarkerlessMouse{mouse}-Core23-RealMultiview'
    dst = root / f'sessions/markerless_mouse_{mouse}'
    dst.mkdir(parents=True, exist_ok=True)
    session = load_session(raw / 'label3d_dannce.mat')
    mat = loadmat(raw / 'label3d_dannce.mat', squeeze_me=True, struct_as_record=False)
    sync = np.asarray(mat['sync']).reshape(-1)
    frames = np.asarray(sync[0].data_frame, dtype=np.int64).reshape(-1)
    samples = np.asarray(sync[0].data_sampleID, dtype=np.int64).reshape(-1)
    assert np.array_equal(frames, np.arange(18000))
    for s in sync:
        assert np.array_equal(s.data_frame, frames)
        assert np.array_equal(s.data_sampleID, samples)
        assert not np.any(s.data_3d) and not np.any(s.data_2d)
    com = np.asarray(mat['com'].com3d, dtype=np.float32)
    assert np.array_equal(np.asarray(mat['com'].sampleID).reshape(-1), samples)
    assert com.shape == (len(frames), 3) and np.isfinite(com).all()
    retained_frames=frames[::source_step]
    save_npz_atomic(dst / 'sync_com.npz', source_frame_id=retained_frames,
        source_sample_id=samples[::source_step], nominal_time_seconds=retained_frames / 100.,
        com_world_xyz_mm=com[::source_step], com_valid=np.isfinite(com[::source_step]).all(-1),
        manual_source_frame_id=session['source_frames'])
    shutil.copy2(raw / 'label3d_dannce.mat', dst / 'label3d_dannce.mat')
    videos = find_videos(raw, session['camera_names'])
    camera_records, jobs = [], []
    (dst / 'cams').mkdir(exist_ok=True)
    for view, camera in enumerate(session['camera_models']):
        K = pinhole_intrinsic(camera['intrinsic_distorted'], 1.)
        E = np.eye(4)
        E[:3,:3] = camera['rotation']
        E[:3,3] = camera['translation']
        maps = undistort_maps(1152,1024,camera['intrinsic_distorted'],K,
                             camera['radial'],camera['tangential'])
        # Rectification support is separate from animal/depth validity.
        support = cv2.remap(np.ones((1024,1152),np.uint8)*255,*maps,
                            cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
        cv2.imwrite(str(dst/'cams'/f'{view:08d}_rectification_valid.png'),support)
        write_camera(dst/'cams'/f'{view:08d}_cam.txt',E,K)
        folder = dst/'rgb'/f'{view:08d}'
        folder.mkdir(parents=True,exist_ok=True)
        camera_records.append({'view_id':view,'name':camera['name'],
            'intrinsic':K.tolist(),'extrinsic':E.tolist(),
            'intrinsic_distorted':camera['intrinsic_distorted'].tolist(),
            'radial':camera['radial'].tolist(),'tangential':camera['tangential'].tolist(),
            'image_pattern':f'rgb/{view:08d}/{{source_frame_id:06d}}.jpg',
            'camera_file':f'cams/{view:08d}_cam.txt'})
        for path in videos[camera['name']]:
            capture=cv2.VideoCapture(str(path))
            count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            dimensions=(int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            fps=capture.get(cv2.CAP_PROP_FPS)
            capture.release()
            assert count==3000 and dimensions==(1152,1024) and abs(fps-100)<1e-6
            jobs.append((mouse,view,int(path.stem),path,folder,maps,dst,source_step))
    write_pair_file(dst/'cams'/'pair.txt',6)
    labels={int(frame):i+1 for i,frame in enumerate(session['source_frames'])}
    split='train' if mouse==1 else 'test'
    rows=[]
    pilot_frames,pilot_meta=pilot_annotations(dst)
    for local_index,i in enumerate(retained_frames):
        i=int(i)
        rows.append({'local_index':local_index,'source_frame_id':i,'source_sample_id':int(samples[i]),
            'time_seconds':i/100.,'segment_id':f'dannce_mouse_{mouse}:continuous0',
            'source_frame_delta':source_step if i else None,'transition_valid':i>0,
            'motion_source_id':f'dannce_mouse_{mouse}','split':split,
            'rgb':[f'rgb/{v:08d}/{i:06d}.jpg' for v in range(6)],
            'manual_keypoints':f'../../frame_{labels[i]:06d}/keypoints.npz' if i in labels else None,
            'keypoint_label_valid':i in labels,'depth':None})
        rows[-1].update(pilot_frames.get(i,{}))
    write_jsonl_atomic(dst/'frames.jsonl',rows)
    meta={'format':'mousegeoforge_blenderpymouse_temporal_multiview_v1',
        'dataset_id':root.name+f'/sessions/markerless_mouse_{mouse}',
        'path':str(dst.relative_to(BASE/'dataset')),'temporal_mode':'continuous',
        'temporal_eligible':False,'export_complete':False,'index':'frames.jsonl',
        'frame_count':len(retained_frames),'views':6,'resolution':[1152,1024],
        'fps':100./source_step,'source_fps':100.,'source_frame_count':18000,'timing':'nominal_video_fps','clock_verified':True,
        'timestamp_note':'Frame time derived from video 100 fps; no independent hardware timestamp supplied.',
        'source_frame_step':source_step,'segment_count':1,'camera_motion':'fixed',
        'intrinsics_constant':True,'motion_source_id':f'dannce_mouse_{mouse}',
        'recommended_split':split,'camera_convention':'world_to_camera_OpenCV',
        'world_coordinate_system':'original_DANNCE_calibration_not_ground_aligned',
        'length_unit':'mm','depth_semantics':'camera_z_mm; pseudo depth available only in separately indexed pilot',
        'distortion_corrected':True,'rectification_interpolation':'Lanczos4',
        'image_storage':'independent JPEG quality95, no links',
        'cameras':camera_records,'sparse_manual_label_frames':len(set(labels)&set(retained_frames)),
        'original_sparse_manual_label_frames':len(labels),
        'complete_keypoint_ground_truth':False,
        'annotation_note':'sync.data_2d/data_3d are zero placeholders, not GT; COM is not joint annotation.'}
    old=dst/'metadata.json'
    meta.update(pilot_meta)
    if old.exists():
        previous=json.loads(old.read_text())
        if previous.get('export_complete'):
            meta.update(export_complete=True,temporal_eligible=True)
    write_json_atomic(dst/'metadata.json',meta)
    return jobs,dst


def export_chunk(job):
    mouse,view,start,path,folder,maps,dst,source_step=job
    marker=dst/'export_chunks'/f'view{view:02d}_{start:06d}.json'
    if all((folder/f'{i:06d}.jpg').is_file() for i in range(start,start+3000,source_step)):
        write_json_atomic(marker,{'mouse':mouse,'view':view,'start':start,'count':3000//source_step,'source_frame_step':source_step})
        return {'mouse':mouse,'view':view,'start':start,'status':'already_complete'}
    capture=cv2.VideoCapture(str(path))
    began=time.monotonic()
    try:
        for index in range(3000):
            ok,raw=capture.read()
            if not ok:
                raise RuntimeError(f'Decode failed {path} {index}')
            if (start+index)%source_step:
                continue
            dst_image=folder/f'{start+index:06d}.jpg'
            if dst_image.exists():
                continue
            corrected=cv2.remap(raw,*maps,cv2.INTER_LANCZOS4,borderMode=cv2.BORDER_CONSTANT)
            temporary=dst_image.with_name(dst_image.stem+'.tmp.jpg')
            if not cv2.imwrite(str(temporary),corrected,[cv2.IMWRITE_JPEG_QUALITY,95]):
                raise IOError(temporary)
            temporary.replace(dst_image)
            if index in (499,1499,2499):
                print(json.dumps({'mouse':mouse,'view':view,'through_frame':start+index}),flush=True)
        if capture.read()[0]:
            raise RuntimeError(f'Unexpected extra frames in {path}')
    finally:
        capture.release()
    record={'mouse':mouse,'view':view,'start':start,'count':3000//source_step,'source_frame_step':source_step,
            'elapsed_seconds':round(time.monotonic()-began,2)}
    write_json_atomic(marker,record)
    return record


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=12)
    parser.add_argument('--source-frame-step',type=int,choices=(4,),default=4,
                        help='DANNCE exports are retained at 25fps; do not silently regenerate discarded 100fps RGB.')
    args=parser.parse_args()
    cv2.setNumThreads(1)
    jobs,destinations=[],[]
    for mouse in (1,2):
        task,dst=prepare(mouse,args.source_frame_step)
        jobs.extend(task)
        destinations.append(dst)
    # Both mice and all views become available for an initial pilot first.
    jobs.sort(key=lambda j:(j[2],j[0],j[1]))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(export_chunk,j) for j in jobs]
        for future in as_completed(futures):
            print(json.dumps(future.result()),flush=True)
    for dst in destinations:
        counts=[len(list((dst/'rgb'/f'{v:08d}').glob('*.jpg'))) for v in range(6)]
        assert counts==[18000//args.source_frame_step]*6,counts
        meta=json.loads((dst/'metadata.json').read_text())
        meta.update(export_complete=True,temporal_eligible=True,image_count=sum(counts))
        write_json_atomic(dst/'metadata.json',meta)
        print(json.dumps({'completed':str(dst),'image_counts':counts}),flush=True)


if __name__=='__main__':
    main()
