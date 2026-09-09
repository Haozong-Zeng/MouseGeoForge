"""Read-only audit of synthetic segment boundaries and DANNCE video coverage."""
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from scipy.io import loadmat

BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'output/temporal_dataset_audit'
OUT.mkdir(parents=True,exist_ok=True)


def probe(p):
    data=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=width,height,avg_frame_rate,nb_frames,duration','-of','json',str(p)]))['streams'][0]
    return {'file':str(p.relative_to(BASE)),**data}


def main():
    result={'synthetic':[],'dannce':[]}
    all_poses={}
    names=['Openfield-BlackFur-CranialWindow']+['PyMouseStudio-BrownFur-'+a+'-Core23' for a in ('WalkAround','WashFace','ClimbWall')]
    for name in names:
        root=BASE/'dataset/BlenderPyMouse'/name
        if not root.exists() and name=='Openfield-BlackFur-CranialWindow':
            parts=[root.parent/(name+'-'+a) for a in ('ClimbWall','WashFace','WalkAround')]
            meta=json.loads((parts[0]/'metadata.json').read_text())
            meta['frame_numbers']=list(range(1,274))
            meta['view_counts']={'1':273}
            by_frame={int(p.name[6:]):p for part in parts for p in part.glob('frame_*') if p.is_dir()}
        else:
            meta=json.loads((root/'metadata.json').read_text())
            by_frame={f:root/f'frame_{f:06d}' for f in meta['frame_numbers']}
        frames=meta['frame_numbers']
        poses=[]
        for f in frames:
            with np.load(by_frame[f]/'keypoints.npz') as a:
                poses.append(a['joint_world_xyz'].reshape(-1,23,3)[0])
        poses=np.stack(poses)
        all_poses[name]=poses
        delta=np.linalg.norm(np.diff(poses,axis=0),axis=-1)
        centered=poses-poses[:,[5]]
        shape_delta=np.linalg.norm(np.diff(centered,axis=0),axis=-1).mean(-1)
        row={'name':name,'frame_count':len(frames),'frame_range':[min(frames),max(frames)],
            'source_blend':meta.get('source_blend'),'view_counts':meta.get('view_counts'),
            'mean_joint_displacement_quantiles':np.percentile(delta.mean(-1),[50,90,99,100]).tolist(),
            'largest_shape_transitions':[{'from':frames[i],'to':frames[i+1],'mean_delta':float(shape_delta[i])}
                                        for i in np.argsort(shape_delta)[-6:][::-1]],
            'boundary_candidates':[{'from':frames[i],'to':frames[i+1],'mean_joint_delta':float(delta[i].mean()),
                                    'mean_centered_delta':float(shape_delta[i])} for i in (90,181) if i<len(delta)]}
        result['synthetic'].append(row)
    segments=all_poses[names[0]].reshape(3,91,23,3)
    comparisons=[]
    for si,seg in enumerate(segments):
        target=(seg-seg[:,[5]]).reshape(-1,3)
        candidates=[]
        for name in names[1:]:
            pose=all_poses[name]
            source=(pose-pose[:,[5]]).reshape(-1,3)
            u,s,vt=np.linalg.svd(source.T@target)
            d=np.eye(3)
            d[-1,-1]=np.linalg.det(u@vt)
            R=u@d@vt
            scale=np.sum((source@R)*target)/np.sum(source*source)
            error=np.sqrt(np.mean((source@R*scale-target)**2))
            candidates.append({'source':name,'rms_after_constant_rotation_scale':float(error),'scale':float(scale)})
        comparisons.append({'segment':si,'frames':[si*91+1,(si+1)*91],'matches':sorted(candidates,key=lambda x:x['rms_after_constant_rotation_scale'])})
    result['synthetic_segment_matches']=comparisons
    for session in ('markerless_mouse_1','markerless_mouse_2'):
        root=BASE/'dataset/DANNCE/raw'/session
        files=sorted((root/'videos').rglob('*.mp4'))
        with ThreadPoolExecutor(max_workers=8) as pool:
            videos=list(pool.map(probe,files))
        cams={}
        for cam in sorted((root/'videos').iterdir()):
            segments=sorted([v for v in videos if Path(v['file']).parent.name==cam.name],key=lambda v:int(Path(v['file']).stem))
            ends=[int(Path(v['file']).stem)+int(v['nb_frames']) for v in segments]
            continuous=all(ends[i]==int(Path(segments[i+1]['file']).stem) for i in range(len(segments)-1))
            cams[cam.name]={'segments':len(segments),'frames':sum(int(v['nb_frames']) for v in segments),
                'end_exclusive':ends[-1],'continuous_segment_numbering':continuous,
                'fps':sorted({v['avg_frame_rate'] for v in segments}),
                'image_sizes':sorted({(v['width'],v['height']) for v in segments})}
        mat=loadmat(root/'label3d_dannce.mat',squeeze_me=True,struct_as_record=False)
        labels=list(np.asarray(mat['labelData'],dtype=object).reshape(-1))
        frameids=np.asarray(labels[0].data_frame,dtype=int).reshape(-1)
        row={'session':session,'cameras':cams,'video_count':len(files),'total_video_bytes':sum(p.stat().st_size for p in files),
            'mat_keys':[k for k in mat if not k.startswith('__')],
            'manual_label_count':len(frameids),'manual_frame_range':[int(frameids.min()),int(frameids.max())],
            'manual_frame_ids':frameids.tolist(),'manual_frame_step_quantiles':np.percentile(np.diff(frameids),[0,50,100]).tolist(),
            'manual_camera_frame_ids_equal':all(np.array_equal(frameids,np.asarray(l.data_frame).reshape(-1)) for l in labels)}
        if 'sync' in mat:
            sync=list(np.asarray(mat['sync'],dtype=object).reshape(-1))
            row['sync']=[{'fields':s._fieldnames,'shapes':{k:list(np.asarray(getattr(s,k)).shape) for k in s._fieldnames},
                         'finite_3d_values':int(np.isfinite(s.data_3d).sum()),'finite_2d_values':int(np.isfinite(s.data_2d).sum()),
                         'nonzero_3d_values':int(np.count_nonzero(s.data_3d)),
                         'nonzero_2d_values':int(np.count_nonzero(s.data_2d)),
                         'frame_ids_equal_arange':bool(np.array_equal(np.asarray(s.data_frame).reshape(-1),np.arange(18000))),
                         'sample_ids_range':[float(np.min(s.data_sampleID)),float(np.max(s.data_sampleID))],
                         'sample_id_steps':np.unique(np.diff(np.asarray(s.data_sampleID).reshape(-1))).tolist()} for s in sync]
        com=mat.get('com')
        row['com_fields']=getattr(com,'_fieldnames',None)
        if row['com_fields']:
            row['com']={k:{'shape':list(np.asarray(getattr(com,k)).shape),'finite_values':int(np.isfinite(np.asarray(getattr(com,k))).sum())} for k in row['com_fields']}
            row['com']['com3d']['min_xyz']=np.min(com.com3d,axis=0).tolist()
            row['com']['com3d']['max_xyz']=np.max(com.com3d,axis=0).tolist()
        result['dannce'].append(row)
    (OUT/'coverage.json').write_text(json.dumps(result,indent=2))
    for r in result['dannce']:
        print(r['session'], 'sync_nonzero_3d', [s['nonzero_3d_values'] for s in r['sync']],
              'sync_nonzero_2d', [s['nonzero_2d_values'] for s in r['sync']], 'com', r['com'])
    print(json.dumps(result['synthetic_segment_matches']))


if __name__=='__main__':
    main()
