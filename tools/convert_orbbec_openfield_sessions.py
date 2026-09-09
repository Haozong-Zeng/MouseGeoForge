"""Materialize complete legacy RGB-D sessions and sparse BlenderPyMouse scenes.

Source RGB/encoded depth are independent copies suitable for archiving.
Only nonempty labeled frames receive float32 millimeter PFM caches.
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from core23_temporal_format import CORE23, save_npz_atomic, write_json_atomic, write_jsonl_atomic

SOURCE_NAMES = ('鼻子','头部','左耳','右耳','颈椎','胸椎','腰椎','尾根','尾巴中端','尾巴末端')
TARGETS = (0,1,2,3,4,5,6,7,10,13)
SCALE = 2.3529


def copy_file(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not os.path.samefile(source, target) and source.stat().st_size == target.stat().st_size:
            # Refuse silent reuse of arbitrary equal-sized files.
            import hashlib
            if hashlib.sha256(source.read_bytes()).digest() == hashlib.sha256(target.read_bytes()).digest():
                return 'copy'
        if not os.path.samefile(source, target):
            raise ValueError(f'Existing target differs: {target}')
    tmp = target.with_suffix(target.suffix+'.tmp')
    shutil.copy2(source, tmp)
    tmp.replace(target)
    return 'copy'


def decode_depth(path: Path):
    q = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if q is None or q.dtype != np.uint8:
        raise ValueError(f'Unexpected depth encoding: {path}')
    if q.ndim == 3:
        if q.shape[2] != 3 or not (np.array_equal(q[:,:,0], q[:,:,1]) and np.array_equal(q[:,:,0],q[:,:,2])):
            raise ValueError(f'Depth channels differ: {path}')
        q = q[:,:,0]
    valid = (q > 0) & (q < 255)
    depth = q.astype(np.float32) * SCALE
    depth[~valid] = 0
    return depth, valid


def write_pfm(path: Path, depth: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix('.pfm.tmp')
    with tmp.open('wb') as stream:
        stream.write(f'Pf\n{depth.shape[1]} {depth.shape[0]}\n-1.0\n'.encode())
        # Match this project's pfm_flip_y=False convention.
        stream.write(np.asarray(depth,dtype='<f4').tobytes())
    tmp.replace(path)


def parse_label(path: Path, size: tuple[int,int]):
    lines=[line.split() for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        return None
    if len(lines)!=1 or len(lines[0])!=25:
        raise ValueError(f'Expected single instance and 10 xy points: {path}')
    values=np.asarray(lines[0],dtype=np.float32)
    if not np.isfinite(values).all() or values[0]!=0:
        raise ValueError(f'Invalid label: {path}')
    box=values[1:5]
    box_ok=bool((box>=0).all() and (box<=1).all() and (box[2:]>0).all())
    if not box_ok:
        raise ValueError(f'Invalid bbox: {path}')
    xy=values[5:].reshape(10,2)
    valid=(xy>=0).all(-1)&(xy<1).all(-1)&(xy!=0).any(-1)
    visible=np.zeros(10,bool)
    vis_valid=np.zeros(10,bool)
    issues=[]
    jp=path.with_suffix('.json')
    if jp.exists():
        data=json.loads(jp.read_text())
        if (data['size']['width'],data['size']['height'])!=size:
            raise ValueError(f'JSON image size mismatch: {jp}')
        for i,name in enumerate(SOURCE_NAMES):
            matches=[p for p in data.get('labels',[]) if p.get('drawType')=='POINT'
                     and p.get('label')==name and len(p.get('points',[]))==2
                     and np.linalg.norm(np.array(p['points'])-xy[i]*size)<.01]
            if len(matches)!=1:
                valid[i]=False
                issues.append({'point':name,'reason':'json_name_coordinate_mismatch_or_duplicate'})
                continue
            attrs=matches[0].get('attr',{}).get('pointsAttr',[])
            flags={str(v)[0] for a in attrs for v in a.values() if str(v).startswith(('0','1'))}
            if len(flags)==1:
                vis_valid[i]=True
                visible[i]=next(iter(flags))=='1'
            elif len(flags)>1:
                issues.append({'point':name,'reason':'conflicting_visibility'})
    uv=np.zeros((1,1,23,2),np.float32)
    in_frame=np.zeros((1,1,23),bool)
    surface_visible=np.zeros_like(in_frame)
    visibility_valid=np.zeros_like(in_frame)
    uv[0,0,list(TARGETS)]=xy*np.asarray(size)
    in_frame[0,0,list(TARGETS)]=valid
    surface_visible[0,0,list(TARGETS)]=visible&valid
    visibility_valid[0,0,list(TARGETS)]=vis_valid&valid
    return dict(names=np.asarray(CORE23),view_ids=np.array([0]),instance_ids=np.array([0]),
                joint_world_xyz=np.zeros((1,23,3),np.float32),surface_world_xyz=np.zeros((1,23,3),np.float32),
                world_valid=np.zeros((1,23),bool),surface_valid=np.zeros((1,23),bool),
                uv=uv,in_frame=in_frame,surface_visible=surface_visible,visibility_valid=visibility_valid,
                boxes_cxcywh=box.reshape(1,1,4),box_valid=np.ones((1,1),bool)), issues


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('/home/haozong/projects/Orbbec OpenField/Orbbec OpenField'))
    parser.add_argument('--output',type=Path,default=Path('dataset/Orbbec-Openfield'))
    args=parser.parse_args()
    root=args.output.resolve()
    root.mkdir(parents=True,exist_ok=True)
    reference=json.loads((root/'unpacked_calibration/Cage531450_R2/metadata.json').read_text())
    color=reference['stream_profiles'][0]['intrinsic']
    K=np.array([[color['fx'],0,color['cx']-280],[0,color['fy'],color['cy']],[0,0,1]])
    sessions=[]
    split_scenes={k:[] for k in ('train','val','test')}
    for labeled in sorted(args.source.glob('*-labeled')):
        source=labeled.with_name(labeled.name.removesuffix('-labeled'))
        if not source.is_dir():
            raise FileNotFoundError(source)
        sid=source.name
        session=root/'sessions'/sid
        session.mkdir(parents=True,exist_ok=True)
        marker=session/'conversion_complete.json'
        if marker.exists():
            rec=json.loads(marker.read_text())
            sessions.append(rec)
            split_scenes[rec['split']].extend(rec['labeled_scenes'])
            print(f'Reusing completed {sid}',flush=True)
            continue
        # Deterministic subject-disjoint initial split for the nine audited animals.
        split='test' if 'Pva-jGcamp8s-R3-' in sid else 'val' if 'Q175xThyGcamp-M5-' in sid else 'train'
        calibration={'source':'reference_device_prior','verified':False,'camera_supervision_valid':False,
                     'reference_serial_number':reference['device']['serial_number'],'K_crop_prior':K.tolist(),
                     'extrinsic_camera_world':np.eye(4).tolist(),'world_coordinate_unit':'millimeter',
                     'crop_origin_xy_assumed':[280,0],'original_size_assumed':[1280,720],'image_size':[720,720],
                     'depth_registered_to_rgb':True,'registration_confirmation':'user',
                     'additional_depth_to_color_transform_applied':False,
                     'rgb_rectification_status':'unknown','color_distortion_prior':reference['camera']['color_distortion'],
                     'depth_axis_semantics':'registered_metric_depth; source_Z_vs_color_Z_unverified',
                     'depth_decode_scale_mm':SCALE,'invalid_depth_codes':[0,255]}
        write_json_atomic(session/'calibration.json',calibration)
        write_json_atomic(session/'reference_device_metadata.json',reference)
        rgb_ids={int(p.stem[4:]):p for p in source.glob('rgb_*.jpg')}
        depth_ids={int(p.stem[6:]):p for p in source.glob('depth_*.png')}
        if rgb_ids.keys()!=depth_ids.keys():
            raise ValueError(f'RGB/depth frame IDs differ: {sid}')
        label_files={int(p.stem[4:]):p for p in labeled.glob('rgb_*.txt')}
        if not label_files.keys()<=rgb_ids.keys():
            raise ValueError(f'Labels missing frames: {sid}')
        records=[]
        annotations=[]
        qa=[]
        storage=Counter()
        labeled_scenes=[]
        previous=None
        for pos,fid in enumerate(sorted(rgb_ids)):
            rgb_rel=f'rgb/{rgb_ids[fid].name}'
            depth_rel=f'depth_encoded/{depth_ids[fid].name}'
            storage[copy_file(rgb_ids[fid],session/rgb_rel)]+=1
            storage[copy_file(depth_ids[fid],session/depth_rel)]+=1
            row={'frame_id':fid,'rgb':rgb_rel,'depth_encoded':depth_rel,'timestamp_us':None,
                 'consecutive_with_previous':previous is not None and fid==previous+1,
                 'pose_annotated':False,'annotation_status':'unlabeled','scene':None,'annotation_index':-1}
            previous=fid
            if fid in label_files:
                lp=label_files[fid]
                copy_file(lp,session/'source_annotations'/lp.name)
                if lp.with_suffix('.json').exists():
                    copy_file(lp.with_suffix('.json'),session/'source_annotations'/lp.with_suffix('.json').name)
                im=cv2.imread(str(session/rgb_rel))
                if im is None or im.shape[:2]!=(720,720):
                    raise ValueError(f'Unexpected RGB shape: {rgb_ids[fid]}')
                parsed=parse_label(lp,(720,720))
                if parsed is None:
                    row['annotation_status']='empty_file_not_a_negative'
                else:
                    arrays,issues=parsed
                    scene=session/f'frame_{fid:012d}'
                    copy_file(session/rgb_rel,scene/'blended_images/00000000.jpg')
                    depth,valid=decode_depth(session/depth_rel)
                    if depth.shape!=(720,720):
                        raise ValueError(f'Unexpected depth shape: {depth_ids[fid]}')
                    write_pfm(scene/'rendered_depth_maps/00000000.pfm',depth)
                    (scene/'depth_valid').mkdir(exist_ok=True)
                    if not cv2.imwrite(str(scene/'depth_valid/00000000.png'),valid.astype(np.uint8)*255):
                        raise OSError('Could not write depth validity mask')
                    save_npz_atomic(scene/'keypoints.npz',**arrays)
                    # Prior is intentionally not named *_cam.txt: the stock loader treats those as GT.
                    (scene/'cams').mkdir(exist_ok=True)
                    prior='extrinsic\n'+'\n'.join(' '.join(map(str,r)) for r in np.eye(4))+'\n\nintrinsic\n'+'\n'.join(' '.join(map(str,r)) for r in K)+'\n\n0 1\n'
                    (scene/'cams/00000000_cam_prior.txt').write_text(prior)
                    write_json_atomic(scene/'annotations/00000000.json',{'instance_id':0,'class_id':0,
                        'bbox_cxcywh_normalized':arrays['boxes_cxcywh'][0,0].tolist(),
                        'source_label':f'../source_annotations/{lp.name}','issues':issues,'mask_available':False})
                    write_json_atomic(scene/'scene_meta.json',{'format':'blenderpymouse_core23_partial_rgbd_v1',
                        'session_id':sid,'frame':fid,'views':[{'view_id':0,'depth_available':True,'camera_available':False}],
                        'depth_units':'millimeter','depth_decode_scale_mm':SCALE,'pfm_flip_y':False,
                        'camera_prior':'cams/00000000_cam_prior.txt','camera_supervision_valid':False,
                        'calibration':'../calibration.json','keypoint_definition':'manual_2d_surface_landmarks',
                        'world_valid':False,'depth_registered_to_rgb':True,'split':split})
                    row.update(pose_annotated=True,annotation_status='labeled_with_qa' if issues else 'labeled',
                               scene=scene.name,annotation_index=len(annotations))
                    annotations.append((fid,arrays))
                    qa.extend({'frame_id':fid,**issue} for issue in issues)
                    labeled_scenes.append(str(scene.relative_to(root)))
            records.append(row)
            if (pos+1)%5000==0:
                print(f'{sid}: {pos+1}/{len(rgb_ids)} frames, {len(annotations)} labeled',flush=True)
        compact={key:np.stack([a[key] for _,a in annotations]) for key in ('uv','in_frame','surface_visible','visibility_valid','boxes_cxcywh','box_valid')}
        # Preserve original JSONs even when their corresponding image/TXT was not exported.
        for jp in labeled.glob('*.json'):
            if not (session/'source_annotations'/jp.name).exists():
                copy_file(jp,session/'source_annotations'/jp.name)
        save_npz_atomic(session/'keypoint_annotations.npz',names=np.asarray(CORE23),frame_ids=np.array([f for f,_ in annotations]),**compact)
        write_jsonl_atomic(session/'frames.jsonl',records)
        save_npz_atomic(session/'frame_index.npz',frame_ids=np.array([r['frame_id'] for r in records],np.int64),
                        annotation_index=np.array([r['annotation_index'] for r in records],np.int32),
                        consecutive_with_previous=np.array([r['consecutive_with_previous'] for r in records]))
        write_json_atomic(session/'label_qa.json',qa)
        rec={'session_id':sid,'session_path':str(session.relative_to(root)),'source_type':'legacy_registered_quantized_rgbd',
             'source_path':str(source.resolve()),'frame_count':len(records),'labeled_frame_count':len(annotations),
             'split':split,'subject_group':sid,'frame_index':'frame_index.npz','frames_manifest':'frames.jsonl',
             'image_size':[720,720],'views':1,'fps':None,'timestamp_valid':False,
             'frame_id_gap_count':max(rgb_ids)-min(rgb_ids)+1-len(rgb_ids),'storage':dict(storage),
             'complete_rgbd_local':True,'depth_units':'millimeter','depth_decode_scale_mm':SCALE,
             'labeled_scenes':labeled_scenes,'qa_issue_count':len(qa)}
        write_json_atomic(marker,rec)
        sessions.append(rec)
        split_scenes[split].extend(labeled_scenes)
        print(f'Completed {sid}: {len(records)} frames / {len(annotations)} labeled',flush=True)
    write_jsonl_atomic(root/'sequences.jsonl',[{k:v for k,v in s.items() if k!='labeled_scenes'} for s in sessions])
    write_json_atomic(root/'splits.json',{'strategy':'subject-disjoint initial legacy split; extend grouping before adding sessions of same animal',
        'sessions':{k:[s['session_id'] for s in sessions if s['split']==k] for k in split_scenes},'labeled_scenes':split_scenes})
    write_json_atomic(root/'metadata.json',{'format':'orbbec_openfield_sessions_v1','session_count':len(sessions),
        'frame_count':sum(s['frame_count'] for s in sessions),'labeled_frame_count':sum(s['labeled_frame_count'] for s in sessions),
        'keypoint_names':list(CORE23),'source_keypoint_names':list(SOURCE_NAMES),'source_to_core23_indices':list(TARGETS),
        'complete_sequence_storage':'independent local file copies',
        'depth_units':'millimeter','image_size':[720,720],'depth_decode_scale_mm':SCALE,'pfm_flip_y':False,
        'camera_supervision_valid':False,'world_supervision_valid':False,'rgb_rectification_status':'unknown',
        'scope':'nine labeled legacy sessions and their complete RGB-D timelines',
        'existing_bag_exports':'retained under unpacked; not yet registered/included in this session manifest',
        'other_legacy_sessions':'remaining 29 source exports not included in this first conversion'})
    print('All manifests published.',flush=True)


if __name__=='__main__':
    main()
