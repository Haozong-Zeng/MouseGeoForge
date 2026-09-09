"""Keep source frames 0,4,...17996, delete reproducible converted intermediates.

Raw 100 fps videos, calibration/MAT and all sparse manual scenes are preserved.
Exact deletion targets are recorded before execution, and original metadata is
archived outside dataset. Use --apply only after reviewing the printed plan.
"""
import argparse
from pathlib import Path
import json
import shutil
import sys
import cv2
import numpy as np
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'BlenderScripts'))
from convert_dannce_mouse_to_blenderpymouse import find_videos,load_session
from core23_temporal_format import write_json_atomic,write_jsonl_atomic,save_npz_atomic
OUT=BASE/'output/dannce_25fps_migration'


def read(path):
    return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    jobs=[]
    targets=[]
    for mouse in (1,2):
        root=BASE/f'dataset/DANNCE-MarkerlessMouse{mouse}-Core23-RealMultiview'
        session=root/f'sessions/markerless_mouse_{mouse}'
        raw=BASE/f'dataset/DANNCE/raw/markerless_mouse_{mouse}'
        source=load_session(raw/'label3d_dannce.mat')
        videos=find_videos(raw,source['camera_names'])
        for camera,paths in videos.items():
            assert len(paths)==6,(camera,paths)
            for path in paths:
                capture=cv2.VideoCapture(str(path))
                assert capture.isOpened() and int(capture.get(cv2.CAP_PROP_FRAME_COUNT))==3000,path
                assert abs(capture.get(cv2.CAP_PROP_FPS)-100)<1e-6,path
                ok,_=capture.read()
                capture.release()
                assert ok,path
        assert len(videos)==6
        keep=set(range(0,18000,4))
        # All kept frames must be present and decodable before deleting any others.
        for view in range(6):
            folder=session/f'rgb/{view:08d}'
            for fid in sorted(keep):
                path=folder/f'{fid:06d}.jpg'
                with path.open('rb') as f:
                    assert f.read(2)==b'\xff\xd8',path
                    f.seek(-2,2)
                    assert f.read(2)==b'\xff\xd9',path
            for fid in (0,4,3000,9000,17996):
                image=cv2.imread(str(folder/f'{fid:06d}.jpg'))
                assert image is not None and image.shape[:2]==(1024,1152)
            for p in folder.glob('*.jpg'):
                assert p.stem.isdigit() and p.resolve().parent==folder.resolve() and not p.is_symlink(),p
                if int(p.stem) not in keep:
                    targets.append(p)
        pilots=[]
        for mp in (session/'pseudo_depth_pilot').glob('*/metadata.json'):
            rows=[json.loads(s) for s in (mp.parent/'frames.jsonl').read_text().splitlines()]
            for row in rows:
                if row['source_frame_id'] not in keep:
                    for key in ('mask','depth','confidence'):
                        for value in row[key]:
                            p=(mp.parent/value).resolve()
                            assert p.is_relative_to(mp.parent.resolve()) and not p.is_symlink()
                            if p.exists():
                                targets.append(p)
            pilots.append(mp)
        partial=session/'pseudo_depth_full_v1'
        if partial.exists():
            assert read(partial/'config.json')['method']=='sam2.1_hiera_small_video_calibrated_6of6_hull_2mm_full_v1'
            assert not list(partial.glob('chunks/*/chunk_complete.json')),'Do not discard published full-depth chunks'
            assert partial.resolve().parent==session.resolve() and not partial.is_symlink()
            for p in partial.rglob('*'):
                if p.is_file():
                    assert p.resolve().is_relative_to(partial.resolve()) and not p.is_symlink()
                    targets.append(p)
        jobs.append((mouse,session,pilots,partial))
    assert len(set(targets))==len(targets)
    report={'source_fps':100,'output_fps':25,'source_frame_step':4,
        'frames_per_mouse':4500,'rgb_images_total':54000,
        'delete_files':len(targets),'delete_bytes':sum(p.stat().st_size for p in targets),
        'raw_videos_and_sparse_manual_scenes_preserved':True}
    write_json_atomic(OUT/'plan.json',report)
    write_jsonl_atomic(OUT/'deletion_targets.jsonl',[{'path':str(p.relative_to(BASE)),'bytes':p.stat().st_size} for p in targets])
    print(json.dumps(report),flush=True)
    if not args.apply:
        return
    for mouse,session,pilots,partial in jobs:
        archive=OUT/f'mouse{mouse}'
        archive.mkdir(exist_ok=True)
        for name in ('metadata.json','frames.jsonl','sync_com.npz'):
            if not (archive/name).exists():
                shutil.copy2(session/name,archive/name)
        rows=[json.loads(s) for s in (session/'frames.jsonl').read_text().splitlines()]
        rows=[r for r in rows if r['source_frame_id']%4==0]
        assert [r['source_frame_id'] for r in rows]==list(range(0,18000,4))
        for i,row in enumerate(rows):
            row.update(local_index=i,source_frame_delta=4 if i else None,transition_valid=i>0,
                time_seconds=row['source_frame_id']/100.)
        write_jsonl_atomic(session/'frames.jsonl',rows)
        with np.load(archive/'sync_com.npz') as arrays:
            data={k:arrays[k] for k in arrays.files}
            n=len(data['source_frame_id'])
            selection=data['source_frame_id']%4==0
            for key,value in list(data.items()):
                if value.ndim and len(value)==n and key!='manual_source_frame_id':
                    data[key]=value[selection]
        save_npz_atomic(session/'sync_com.npz',**data)
        for mp in pilots:
            provenance=archive/mp.parent.name
            provenance.mkdir(exist_ok=True)
            for name in ('metadata.json','frames.jsonl'):
                if not (provenance/name).exists():
                    shutil.copy2(mp.parent/name,provenance/name)
            pm=read(mp)
            prows=[json.loads(s) for s in (mp.parent/'frames.jsonl').read_text().splitlines()]
            prows=[r for r in prows if r['source_frame_id']%4==0]
            pm.update(frame_count=len(prows),source_frame_range=[prows[0]['source_frame_id'],prows[-1]['source_frame_id']],
                source_frame_step=4,stored_fps=25.,source_fps=100.,
                validation_metrics_note='Original 100fps validation retained in output; stored pilot filtered to 25fps.')
            pm.pop('metrics',None)
            pm.pop('held_out_keypoint_containment',None)
            write_json_atomic(mp,pm)
            write_jsonl_atomic(mp.parent/'frames.jsonl',prows)
        meta=read(session/'metadata.json')
        meta.update(frame_count=4500,image_count=27000,fps=25.,source_fps=100.,source_frame_count=18000,
            source_frame_step=4,sparse_manual_label_frames=sum(r['manual_keypoints'] is not None for r in rows),
            original_sparse_manual_label_frames=81 if mouse==1 else 91,
            depth_available_frame_count=sum(bool(r.get('depth')) for r in rows),depth_complete=False,
            storage_sampling='Keep source frame IDs 0,4,...17996; local_index=0..4499; no interpolation',
            timestamp_note='time_seconds=source_frame_id/100=local_index/25; nominal video clock',
            downsampling_provenance=str(archive.relative_to(BASE)))
        meta.pop('pseudo_depth_full',None)
        for p in meta.get('pseudo_depth_pilots',[]):
            pm=read(session/p['path']/'metadata.json')
            p.update(frame_count=pm['frame_count'],source_frame_range=pm['source_frame_range'],source_frame_step=4)
        write_json_atomic(session/'metadata.json',meta)
        # Keep only small provenance for the interrupted, unvalidated 100fps run.
        if partial.exists():
            shutil.copy2(partial/'config.json',archive/'discarded_full100fps_config.json')
    removed_bytes=0
    for i,p in enumerate(targets):
        removed_bytes+=p.stat().st_size
        p.unlink()  # Every exact target was resolved and recorded above; raw sources are untouched.
        if i and i%20000==0:
            print(f'Deleted reproducible converted files {i}/{len(targets)}',flush=True)
    for _,session,_,partial in jobs:
        if partial.exists():
            for directory in sorted((p for p in partial.rglob('*') if p.is_dir()),key=lambda p:len(p.parts),reverse=True):
                directory.rmdir()
            partial.rmdir()
        assert all(len(list((session/f'rgb/{v:08d}').glob('*.jpg')))==4500 for v in range(6))
    report.update(applied=True,removed_bytes=removed_bytes)
    write_json_atomic(OUT/'validation.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
