"""Split CranialWindow and publish temporal indices and motion-source groups."""
from __future__ import annotations
import json
import shutil
from pathlib import Path
import numpy as np
from core23_temporal_format import write_json_atomic, write_jsonl_atomic

BASE=Path(__file__).resolve().parents[1]
DATA=BASE/'dataset'
BLENDER=DATA/'BlenderPyMouse'
OLD='Openfield-BlackFur-CranialWindow'
ACTIONS=(('ClimbWall',1,91),('WashFace',92,182),('WalkAround',183,273))


def split_cranial():
    source=BLENDER/OLD
    archive=BASE/'output/dataset_reorganization'/OLD
    if not source.exists():
        if not all((BLENDER/f'{OLD}-{action}').is_dir() for action,_,_ in ACTIONS):
            raise FileNotFoundError('Incomplete CranialWindow split')
        return
    if source.resolve().parent!=BLENDER.resolve():
        raise ValueError('Unexpected split source')
    meta=json.loads((source/'metadata.json').read_text())
    expected={f'frame_{i:06d}' for i in range(1,274)}
    present={p.name for p in source.glob('frame_*') if p.is_dir()}
    for action,lo,hi in ACTIONS:
        target=BLENDER/f'{OLD}-{action}'
        if target.resolve().parent!=BLENDER.resolve():
            raise ValueError('Unexpected split destination')
        target.mkdir(exist_ok=True)
        for fid in range(lo,hi+1):
            src=source/f'frame_{fid:06d}'
            dst=target/src.name
            if src.exists() and dst.exists():
                raise ValueError(f'Duplicate frame target: {dst}')
            if src.exists():
                shutil.move(str(src),str(dst))
            elif not dst.exists():
                raise FileNotFoundError(src)
        for name in ('core23_vertex_support_map.audit.json','core23_vertex_support_map.pkl'):
            if (source/name).exists():
                shutil.copy2(source/name,target/name)
        update=dict(meta)
        update.update(dataset_name=target.name,dataset_root=str(target),frames=[lo,hi],
            frame_numbers=list(range(lo,hi+1)),scene_count=91,
            scene_folders=[f'frame_{f:06d}' for f in range(lo,hi+1)],view_counts={'1':91},
            source_dataset_before_split=OLD,action=action,
            frame_numbering='original Blender frames preserved; local_index in temporal index starts at zero')
        # Dataset-level render/calibration totals from the concatenation cannot describe a subset.
        update['pre_split_metadata_provenance']=str(archive/'metadata.json')
        write_json_atomic(target/'metadata.json',update)
        print(f'Split {target.name}: {lo}..{hi}',flush=True)
    leftovers=[p for p in source.iterdir() if p.is_dir()]
    if leftovers:
        raise ValueError(f'Unexpected remaining directories: {leftovers}')
    if archive.exists():
        raise FileExistsError(archive)
    archive.parent.mkdir(parents=True,exist_ok=True)
    shutil.move(str(source),str(archive))
    write_json_atomic(BASE/'output/dataset_reorganization/path_migrations.json',{
        str(source):[{'path':str(BLENDER/f'{OLD}-{a}'),'source_frame_range':[lo,hi]} for a,lo,hi in ACTIONS]})


def camera_values(path):
    lines=[s.strip() for s in path.read_text().splitlines() if s.strip()]
    e=lines.index('extrinsic')
    k=lines.index('intrinsic')
    return np.array([[float(v) for v in r.split()] for r in lines[e+1:e+5]]),np.array([[float(v) for v in r.split()] for r in lines[k+1:k+4]])


def main():
    split_cranial()
    catalog=[]
    groups={}
    for root in sorted(BLENDER.iterdir()):
        mp=root/'metadata.json'
        if not mp.exists():
            continue
        meta=json.loads(mp.read_text())
        action=next((a for a,_,_ in ACTIONS if root.name==f'{OLD}-{a}' or root.name==f'PyMouseStudio-BrownFur-{a}-Core23'),None)
        water='WaterReaching' in root.name and 'FrameToFrame' in root.name
        completed_animation=meta.get('temporal_completion',{}).get('export_complete',False)
        pseudo='PseudoSynthetic' in root.name
        eligible=bool(action or water) and not pseudo
        mode='continuous' if eligible else 'pose_set' if pseudo or 'DiversePose400' in root.name else 'unknown'
        group=None
        split=None
        if action:
            group={'ClimbWall':'mouse_climb_wall_91','WashFace':'mouse_wash_face_91','WalkAround':'mouse_walk_around_91'}[action]
            split={'ClimbWall':'train','WashFace':'val','WalkAround':'test'}[action]
        elif water:
            motion='wash_face' if 'WashFace' in root.name else 'fs_cam_c' if 'FScamC' in root.name else 'fg_cam_b'
            group=f'water_reaching_{motion}'
            split={'fg_cam_b':'train','fs_cam_c':'val','wash_face':'test'}[motion]
        elif pseudo:
            # A styled copy and its base are the same motion, never independent split units.
            trial=root.name.split('_PseudoSynthetic')[0]
            group='anipose_trial/'+trial
            split='val' if trial.startswith('FJ_') else 'test' if trial.startswith('FS_') else 'train'
        frames=sorted(int(p.name[6:]) for p in root.glob('frame_*') if p.is_dir() and p.name[6:].isdigit())
        if root.name.startswith(OLD+'-'):
            kp=meta.get('keypoint_annotations',{})
            kp['vertex_support_map']='core23_vertex_support_map.pkl'
            kp['visible_keypoint_instances']=sum(int(np.load(root/f'frame_{fid:06d}'/'keypoints.npz')['surface_visible'].sum()) for fid in frames)
            calibration=meta.get('camera_calibration',{})
            for key in ('legacy_projection_mean_error_px','legacy_projection_max_error_px'):
                calibration.pop(key,None)  # Old aggregate is retained only in the archived provenance.
        rec={'dataset_id':root.name,'path':str(root.relative_to(DATA)),'temporal_mode':mode,
             'temporal_eligible':eligible,'frame_count':len(frames),
             'motion_source_id':group,'recommended_split':split,'timing':'frame_order_only',
             'clock_verified':False,'evidence':'user_confirmed_not_reliable_temporal' if pseudo else 'user_confirmed_and_frame_index_audited' if water else 'source_animation_shape_match' if action else 'not_temporally_verified'}
        if 'temporal_completion' in meta:
            rec['export_complete']=completed_animation
            rec['source_animation_fps']=meta['temporal_completion']['source_animation_fps']
        if eligible:
            diffs=np.diff(frames)
            step=int(np.gcd.reduce(diffs)) if len(diffs) else 1
            records=[]
            seg=0
            previous=None
            reference_cams={}
            same_e=same_k=True
            for pos,fid in enumerate(frames):
                valid=previous is not None and fid-previous==step
                if previous is not None and not valid:
                    seg+=1
                scene=root/f'frame_{fid:06d}'
                cameras=sorted((scene/'cams').glob('*_cam.txt'))
                for cam in cameras:
                    e,k=camera_values(cam)
                    if cam.name not in reference_cams:
                        reference_cams[cam.name]=(e,k)
                    else:
                        same_e &= bool(np.allclose(e,reference_cams[cam.name][0],atol=1e-5,rtol=1e-6))
                        same_k &= bool(np.allclose(k,reference_cams[cam.name][1],atol=1e-5,rtol=1e-6))
                records.append({'local_index':pos,'source_frame_id':fid,'scene':scene.name,
                    'segment_id':f'{root.name}:segment{seg}','source_frame_delta':None if previous is None else fid-previous,
                    'transition_valid':valid,'motion_source_id':group,'split':split})
                previous=fid
            rec.update(index='temporal_index.jsonl',source_frame_step=step,segment_count=seg+1,
                camera_motion='fixed' if same_e else 'varying',intrinsics_constant=same_k,
                views=len(reference_cams))
            write_jsonl_atomic(root/'temporal_index.jsonl',records)
            groups.setdefault(group,{'split':split,'members':[]})['members'].append(str(root.relative_to(DATA)))
        meta['temporal']=rec
        write_json_atomic(mp,meta)
        catalog.append(rec)
    # Orbbec's own split/sequence manifests remain authoritative.
    orbbec=DATA/'Orbbec-Openfield'
    for line in (orbbec/'sequences.jsonl').read_text().splitlines():
        s=json.loads(line)
        catalog.append({'dataset_id':'Orbbec-Openfield/'+s['session_id'],
            'path':str((orbbec/s['session_path']).relative_to(DATA)),'temporal_mode':'continuous_with_gaps',
            'temporal_eligible':True,'index':'frames.jsonl','frame_count':s['frame_count'],
            'motion_source_id':'orbbec/'+s['subject_group'],'recommended_split':s['split'],
            'timing':'frame_order_only','clock_verified':False})
    for mouse in (1,2):
        root=DATA/f'DANNCE-MarkerlessMouse{mouse}-Core23-RealMultiview'
        meta=json.loads((root/'metadata.json').read_text())
        rec={'dataset_id':root.name+'/manual_sparse','path':str(root.relative_to(DATA)),
            'temporal_mode':'sparse','temporal_eligible':False,'frame_count':meta['scene_count'],
            'motion_source_id':f'dannce_mouse_{mouse}','recommended_split':'train' if mouse==1 else 'test',
            'note':'Existing frame folders are sparse annotation times; complete sessions have separate indices.'}
        meta['temporal']=rec
        meta['temporal_sessions']=[str(p.parent.relative_to(root)) for p in sorted((root/'sessions').glob('*/metadata.json'))]
        meta['depth_availability_scope']='Root depth_available refers only to original sparse frame scenes, not all frames of temporal_sessions.'
        meta['legacy_pseudo_depth_caution']='Original sparse depth is the earlier 4-view-tolerant hull. Continuous pilot uses stricter 6/6 support after reprojection checks; old sparse PFM files have not been replaced.'
        write_json_atomic(root/'metadata.json',meta)
        catalog.append(rec)
        for session_meta in sorted((root/'sessions').glob('*/metadata.json')):
            full=json.loads(session_meta.read_text())
            if full.get('format')!='mousegeoforge_blenderpymouse_temporal_multiview_v1':
                continue
            catalog.append({k:full[k] for k in ('dataset_id','path','temporal_mode',
                'temporal_eligible','frame_count','index','source_frame_step','segment_count',
                'camera_motion','intrinsics_constant','views','fps','timing','clock_verified',
                'motion_source_id','recommended_split','export_complete')})
    # Preserve existing real-data sequence manifests/splits, including pose-only data.
    for name in ('SBeA-Mouse16-Core23-RealMultiviewTemporal','sDANNCE-Mouse-Core23-PoseSequences'):
        manifest=DATA/name/'sequences.jsonl'
        for line in manifest.read_text().splitlines():
            seq=json.loads(line)
            seqpath=BASE/seq['sequence_npz']
            catalog.append({'dataset_id':name+'/'+seq['sequence_id'],
                'path':str(seqpath.parent.relative_to(DATA)),'index':'sequence.npz',
                'index_format':'core23_temporal_npz','manifest':str(manifest.relative_to(DATA)),
                'sequence_id':seq['sequence_id'],'temporal_mode':'continuous',
                'temporal_eligible':True,'frame_count':seq['frame_count'],
                'rgb_available':seq.get('rgb_available',False),'rgb_frame_count':seq.get('rgb_frame_count',0),
                'motion_source_id':seq['source']+'/'+seq['sequence_id'],
                'recommended_split':seq['split'],'fps':seq['fps'],
                'timing':'source_sequence_fps','clock_verified':True,
                'source_frame_step':1,'camera_motion':'fixed',
                'pose_to_camera_alignment_valid':seq.get('pose_to_camera_alignment_valid',False)})
    for rec in catalog:
        group=rec.get('motion_source_id')
        if not group:
            continue
        entry=groups.setdefault(group,{'split':rec['recommended_split'],'members':[]})
        if entry['split']!=rec['recommended_split']:
            raise ValueError(f'Motion source leaks between splits: {group}')
        if rec['path'] not in entry['members']:
            entry['members'].append(rec['path'])
    write_json_atomic(DATA/'temporal_catalog.json',{'version':1,'datasets':catalog})
    write_json_atomic(DATA/'motion_source_groups.json',{'version':1,
        'rule':'All members of a motion source must remain in the same split; do not random-split per frame or color.',
        'water_group_evidence':'conservative grouping by shared source-task name; no assertion that every frame is identical',
        'groups':groups})
    print(json.dumps({'catalog_entries':len(catalog),'motion_groups':len(groups),
        'eligible_blender_datasets':sum(r['temporal_eligible'] for r in catalog if r['path'].startswith('BlenderPyMouse/'))}),flush=True)


if __name__=='__main__':
    main()
