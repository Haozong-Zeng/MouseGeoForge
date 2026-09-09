"""Full-length DANNCE foreground pseudo labels, bounded and resumable.

SAM2.1 is conditioned on real manual frames and the previous segment mask.
No interpolated skeleton is used as ground truth. Calibrated 6/6 visual hulls
use the pilot's 2 mm world lattice/rasterizer and float32 camera-Z mm PFM.
Only validated chunks enter the session index; original RGB/labels/pilots stay.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import subprocess
import time
import traceback
import cv2
import numpy as np
import torch
from scipy.io import loadmat

BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'output/dannce_full_pseudo_depth'
sys.path.insert(0,str(BASE/'BlenderScripts'))
from convert_dannce_mouse_to_blenderpymouse import load_session
from add_dannce_real_multiview_pseudo_depth import project,prompts_for_view,rasterize_depth,write_pfm,read_pfm
from segment_dannce_4dgs_sam2 import clean_mask,overlay
from validate_dannce_temporal_pseudo_depth import silhouette_confidence,fixed_depth_color
from core23_temporal_format import write_json_atomic,write_jsonl_atomic

METHOD='sam2.1_hiera_small_video_calibrated_6of6_hull_2mm_full_25fps_v1'
CHECKPOINT=BASE/'third_party/checkpoints/sam2.1_hiera_small.pt'


def read(path):
    return json.loads(path.read_text())


def status(**values):
    previous=read(OUT/'status.json') if (OUT/'status.json').exists() else {}
    previous.update(values,pid=os.getpid(),updated_at_utc=datetime.now(timezone.utc).isoformat())
    write_json_atomic(OUT/'status.json',previous)
    print(json.dumps(values),flush=True)


def session_path(mouse):
    return BASE/f'dataset/DANNCE-MarkerlessMouse{mouse}-Core23-RealMultiview/sessions/markerless_mouse_{mouse}'


def prepare(mouse,chunk_size):
    session=session_path(mouse)
    meta=read(session/'metadata.json')
    assert meta['export_complete'] and meta['frame_count']==4500 and meta['resolution']==[1152,1024]
    assert meta['source_frame_step']==4 and meta['fps']==25.
    raw=load_session(session/'label3d_dannce.mat')
    labels={int(f):xyz for f,xyz in zip(raw['source_frames'],raw['source_xyz'])}
    # Original compact MAT keeps COM for prompt times outside the retained 25fps grid.
    com=np.asarray(loadmat(session/'label3d_dannce.mat',squeeze_me=True,struct_as_record=False)['com'].com3d)
    assert com.shape==(18000,3) and np.isfinite(com).all()
    dst=session/'pseudo_depth_full_v1'
    dst.mkdir(exist_ok=True)
    config={'method':METHOD,'chunk_size_source_frames':chunk_size,'frame_count':4500,'views':6,
        'source_frame_count':18000,'source_frame_step':4,'fps':25.,'source_fps':100.,
        'resolution':[1152,1024],'voxel_mm':2.,'minimum_views':6,
        'box_padding_px':18. if mouse==1 else 96.,'bounds_padding_mm':30.,
        'pfm_flip_y':True,'depth_unit':'mm','depth_semantics':'camera_z',
        'mask_is_pseudo_label':True,'depth_is_pseudo_label':True,
        'background_depth':0,'invalid_depth':0,'checkpoint_sha256':hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        'calibration_sha256':hashlib.sha256(json.dumps(meta['cameras'],sort_keys=True).encode()).hexdigest(),
        'segmentation':'manual anchor + previous-chunk mask continuity; no fabricated pose labels',
        'confidence_semantics':'uint8 floor(255 * reprojected silhouette votes / 6), not probability'}
    if (dst/'config.json').exists():
        assert read(dst/'config.json')==config,'Configuration changed; do not mix incompatible pseudo labels'
    else:
        archive=OUT/f'original_mouse{mouse}'
        archive.mkdir(parents=True,exist_ok=True)
        for name in ('metadata.json','frames.jsonl'):
            if not (archive/name).exists():
                shutil.copy2(session/name,archive/name)
        write_json_atomic(dst/'config.json',config)
    return session,dst,meta,labels,com,config


def track_chunk(session,chunk,start,end,labels,cameras,predictor,padding):
    # Include one previous published frame as an explicit mask conditioning frame.
    begin=max(0,start-4)
    retained=list(range(start,end+1,4))
    manual=sorted(fid for fid in labels if start<=fid<=end)
    if start==0 and not manual:
        raise ValueError('First chunk must include a real manual seed (mouse2 starts at 307)')
    seed=manual[0] if manual else None
    # Real off-grid manual RGB is used only as an extra SAM conditioning image,
    # recovered from the preserved sparse annotation scene, never stored as a new output frame.
    input_frames=sorted(set(range(begin,end+1,4))|({seed} if seed is not None else set()))
    for view,camera in enumerate(cameras):
        marker=chunk/f'masks_view{view}_complete.json'
        if marker.exists():
            record=read(marker)
            assert record['frame_range']==[start,end]
            assert all((chunk/f'masks/{view:08d}/{fid:06d}.png').is_file() for fid in retained)
            continue
        status(phase='sam2_tracking',mouse=int(session.name[-1]),chunk=[start,end],view=view)
        numeric=chunk/'tracking_rgb'/f'{view:08d}'
        masks=chunk/'masks'/f'{view:08d}'
        numeric.mkdir(parents=True,exist_ok=True)
        masks.mkdir(parents=True,exist_ok=True)
        for index,fid in enumerate(input_frames):
            target=numeric/f'{index:06d}.jpg'
            src=session/f'rgb/{view:08d}/{fid:06d}.jpg'
            if fid%4:
                sparse_id=list(labels).index(fid)+1
                src=session.parent.parent/f'frame_{sparse_id:06d}/blended_images/{view:08d}.jpg'
            if not target.exists() or target.stat().st_size!=src.stat().st_size:
                shutil.copy2(src,target)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            state=predictor.init_state(video_path=str(numeric),offload_video_to_cpu=True,
                offload_state_to_cpu=False,async_loading_frames=False)
            if start>0:
                preceding=[p for p in (session/'pseudo_depth_full_v1/chunks').glob('*/chunk_complete.json')
                    if read(p)['frame_range'][1]==start-4]
                assert len(preceding)==1,'A preceding completed mask chunk is required'
                anchor=cv2.imread(str(preceding[0].parent/f'masks/{view:08d}/{start-4:06d}.png'),0)
                assert anchor is not None and np.count_nonzero(anchor)>100
                predictor.add_new_mask(state,frame_idx=0,obj_id=1,mask=anchor>0)
            if seed is not None:
                xyz=labels[seed]
                K,E=np.asarray(camera['intrinsic']),np.asarray(camera['extrinsic'])
                uv,z=project(xyz,K,E)
                valid=np.isfinite(xyz).all(-1)&(np.linalg.norm(xyz,axis=-1)>1e-5)&(z>0)
                valid&=(uv[:,0]>=0)&(uv[:,0]<1152)&(uv[:,1]>=0)&(uv[:,1]<1024)
                box,points=prompts_for_view(uv,valid,1152,1024,padding)
                predictor.add_new_points_or_box(state,frame_idx=input_frames.index(seed),obj_id=1,
                    points=points,labels=np.ones(len(points),np.int32),box=box)
            anchor_index=input_frames.index(seed) if seed is not None else 0
            written=set()
            for reverse in (False,True) if anchor_index>0 else (False,):
                for index,ids,logits in predictor.propagate_in_video(state,
                    start_frame_idx=anchor_index,reverse=reverse):
                    fid=input_frames[index]
                    if fid not in retained:
                        continue
                    mask=clean_mask((logits[ids.index(1),0]>0).cpu().numpy())
                    if not cv2.imwrite(str(masks/f'{fid:06d}.png'),mask):
                        raise IOError(masks)
                    written.add(fid)
            predictor.reset_state(state)
        del state
        gc.collect()
        torch.cuda.empty_cache()
        assert written==set(retained)
        write_json_atomic(marker,{'frame_range':[start,end],'manual_seed':seed,
            'previous_mask_frame':start-4 if start else None,'box_padding_px':padding,
            'source_frame_step':4,'off_grid_manual_seed_rgb':seed is not None and seed%4!=0,
            'unused_manual_frames':[fid for fid in manual if fid!=seed]})
        # This numeric-JPEG copy is disposable; verify canonical images first.
        assert numeric.resolve().is_relative_to((session/'pseudo_depth_full_v1').resolve())
        assert all((session/f'rgb/{view:08d}/{fid:06d}.jpg').is_file() for fid in range(begin,end+1,4))
        shutil.rmtree(numeric)
    return seed


class CachedHull:
    """Same NumPy projection/rounding as the validated hull, CUDA silhouette voting."""
    def __init__(self,bounds,Ks,Es):
        self.bounds=bounds
        axes=[np.arange(bounds[0,i],bounds[1,i]+1,2.) for i in range(3)]
        self.points=np.stack(np.meshgrid(*axes,indexing='ij'),-1).reshape(-1,3).astype(np.float32)
        self.indices=[]
        self.inside=[]
        for K,E in zip(Ks,Es):
            uv,z=project(self.points,K,E)
            x,y=np.rint(uv).astype(np.int32).T
            good=(z>0)&(x>=0)&(x<1152)&(y>=0)&(y<1024)
            indices=np.clip(y,0,1023).astype(np.int64)*1152+np.clip(x,0,1151)
            self.indices.append(torch.from_numpy(indices).cuda())
            self.inside.append(torch.from_numpy(good).cuda())

    def __call__(self,masks):
        with torch.inference_mode():
            votes=torch.zeros(len(self.points),dtype=torch.uint8,device='cuda')
            for mask,idx,inside in zip(masks,self.indices,self.inside):
                image=torch.from_numpy(mask).cuda().flatten()
                votes+=((image[idx]>127)&inside).to(torch.uint8)
            keep=(votes==6).cpu().numpy()
        return self.points[keep]


def frame_depth(chunk,fid,masks,hull,Ks,Es,executor):
    def one(view):
        depth=rasterize_depth(hull,masks[view],Ks[view],Es[view],2.) if len(hull)>=100 else np.zeros((1024,1152),np.float32)
        confidence=silhouette_confidence(depth,masks,Ks,Es,view)
        for name in ('rendered_depth_maps','confidence'):
            (chunk/name/f'{view:08d}').mkdir(parents=True,exist_ok=True)
        path=chunk/f'rendered_depth_maps/{view:08d}/{fid:06d}.pfm'
        temporary=path.with_suffix('.pfm.tmp')
        write_pfm(temporary,depth)
        temporary.replace(path)
        if not cv2.imwrite(str(chunk/f'confidence/{view:08d}/{fid:06d}.png'),confidence):
            raise IOError('Cannot write confidence')
        return depth,confidence,{'view_id':view,'mask_area_px':int(np.count_nonzero(masks[view])),
            'depth_coverage':float(np.count_nonzero(depth)/max(1,np.count_nonzero(masks[view]))),
            'confidence5_fraction':float(np.count_nonzero(confidence>=212)/max(1,np.count_nonzero(depth)))}
    return list(executor.map(one,range(6)))


def verify_frame(chunk,fid):
    for v in range(6):
        d=read_pfm(chunk/f'rendered_depth_maps/{v:08d}/{fid:06d}.pfm')
        m=cv2.imread(str(chunk/f'masks/{v:08d}/{fid:06d}.png'),0)
        c=cv2.imread(str(chunk/f'confidence/{v:08d}/{fid:06d}.png'),0)
        assert m is not None and c is not None and d.shape==m.shape==c.shape==(1024,1152)
        assert d.dtype==np.float32 and np.isfinite(d).all() and (d>=0).all()
        assert not np.any(d[m==0]>0) and not np.any(c[d==0]>0)
        assert np.all((m==0)|(m==255))
        assert np.isin(c,[0,42,85,127,170,212,255]).all()


def review_frame(session,chunk,fid,results,masks,limits):
    panels=[]
    for v,(depth,confidence,_) in enumerate(results):
        image=cv2.imread(str(session/f'rgb/{v:08d}/{fid:06d}.jpg'))
        rgb=cv2.resize(overlay(image,masks[v]),(288,256))
        colored=cv2.resize(fixed_depth_color(depth,*limits[v]),(288,256))
        cv2.putText(rgb,f'view {v} frame {fid}',(5,18),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
        panels.append(np.concatenate([rgb,colored],1))
    grid=np.concatenate([np.concatenate(panels[i:i+2],1) for i in (0,2,4)],0)
    out=OUT/session.name/'review'
    out.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(out/f'{fid:06d}.jpg'),grid)


def publish(session,dst):
    from dannce_pseudo_manifest import full_annotations
    annotations,info=full_annotations(session)
    rows=[json.loads(line) for line in (session/'frames.jsonl').read_text().splitlines()]
    for row in rows:
        if row['source_frame_id'] in annotations:
            row.update(annotations[row['source_frame_id']])
    write_jsonl_atomic(session/'frames.jsonl',rows)
    metadata=read(session/'metadata.json')
    available=sum(bool(row.get('depth')) for row in rows)
    metadata.update(pseudo_depth_full=info,depth_available_frame_count=available,
        depth_complete=info['complete'],mask_available_frame_count=sum(bool(r.get('mask')) for r in rows),
        mask_complete=info['complete'],depth_semantics='camera_z_mm; foreground pseudo labels, background/invalid=0',
        pseudo_depth_training_default=False)
    write_json_atomic(session/'metadata.json',metadata)
    write_json_atomic(dst/'metadata.json',{**read(dst/'config.json'),**info,'export_complete':info['complete']})


def process_chunk(mouse,start,end,predictor,executor,args):
    session,dst,metadata,labels,com,config=prepare(mouse,args.chunk_size)
    chunk=dst/f'chunks/f{start:06d}-{end:06d}'
    if (chunk/'chunk_complete.json').exists():
        publish(session,dst)
        return
    chunk.mkdir(parents=True,exist_ok=True)
    seed=track_chunk(session,chunk,start,end,labels,metadata['cameras'],predictor,config['box_padding_px'])
    Ks=[np.asarray(c['intrinsic']) for c in metadata['cameras']]
    Es=[np.asarray(c['extrinsic']) for c in metadata['cameras']]
    xyz=np.stack(list(labels.values()))
    good=np.isfinite(xyz).all(-1)&(np.linalg.norm(xyz,axis=-1)>1e-5)
    relative=xyz-com[np.array(list(labels)),None,:]
    low=np.floor((com[start:end+1].min(0)+relative[good].min(0)-30)/2)*2
    high=np.ceil((com[start:end+1].max(0)+relative[good].max(0)+30)/2)*2
    cached=CachedHull(np.stack([low,high]),Ks,Es)
    corners=np.array([[x,y,z] for x in (low[0],high[0]) for y in (low[1],high[1]) for z in (low[2],high[2])])
    limits=[(float(project(corners,K,E)[1].min()),float(project(corners,K,E)[1].max())) for K,E in zip(Ks,Es)]
    rows=[]
    metrics=[]
    bad=0
    begin=time.monotonic()
    selected_frames=list(range(start,end+1,4))
    for position,fid in enumerate(selected_frames):
        metric_path=chunk/f'frame_metrics/{fid:06d}.json'
        if metric_path.exists():
            record=read(metric_path)
            verify_frame(chunk,fid)
        else:
            masks=[cv2.imread(str(chunk/f'masks/{v:08d}/{fid:06d}.png'),0) for v in range(6)]
            assert all(m is not None and m.shape==(1024,1152) for m in masks)
            hull=cached(masks)
            results=frame_depth(chunk,fid,masks,hull,Ks,Es,executor)
            touches=bool(len(hull) and np.any((hull<=low+2)|(hull>=high-2)))
            valid=len(hull)>=100 and not touches and all(r[2]['mask_area_px']>100 for r in results)
            record={'source_frame_id':fid,'occupied_voxels':len(hull),'hull_touches_bounds':touches,
                'views':[r[2] for r in results],'training_valid':valid,
                'manual_seed_frame':seed,'prompt_frame':fid==seed,'warnings':[]}
            if not valid:
                record['warnings'].append('empty_or_clipped_hull_or_empty_mask')
            if min(r[2]['depth_coverage'] for r in results)<.8:
                record['warnings'].append('low_depth_coverage')
            if fid in labels:
                for v,(K,E) in enumerate(zip(Ks,Es)):
                    p=labels[fid]
                    uv,z=project(p,K,E)
                    ij=np.rint(np.nan_to_num(uv,nan=-99999,posinf=-99999,neginf=-99999)).astype(int)
                    goodp=np.isfinite(p).all(-1)&(np.linalg.norm(p,axis=-1)>1e-5)&(z>0)
                    goodp&=(ij[:,0]>=0)&(ij[:,0]<1152)&(ij[:,1]>=0)&(ij[:,1]<1024)
                    record['views'][v]['manual_keypoints_in_mask']=int((masks[v][ij[goodp,1],ij[goodp,0]]>0).sum())
                    record['views'][v]['manual_keypoint_count']=int(goodp.sum())
            verify_frame(chunk,fid)
            write_json_atomic(metric_path,record)
            if fid in {start,selected_frames[len(selected_frames)//2],end,seed} or record['warnings']:
                review_frame(session,chunk,fid,results,masks,limits)
        metrics.append(record)
        bad+=not record['training_valid']
        relative_chunk=chunk.relative_to(session)
        row={'source_frame_id':fid,**{key:[str(relative_chunk/f'{folder}/{v:08d}/{fid:06d}.{ext}') for v in range(6)]
            for key,folder,ext in [('mask','masks','png'),('depth','rendered_depth_maps','pfm'),('confidence','confidence','png')]},
            'depth_pfm_flip_y':True,'depth_is_pseudo_label':True,'mask_is_pseudo_label':True,
            'depth_unit':'mm','depth_semantics':'camera_z','pseudo_depth_method':METHOD,
            'pseudo_depth_training_valid':record['training_valid'],'pseudo_mask_training_valid':record['training_valid'],
            'pseudo_prompt_frame':fid==seed,'pseudo_quality_warnings':record['warnings']}
        rows.append(row)
        if position%20==0 or fid==end:
            status(phase='visual_hull_depth',mouse=mouse,chunk=[start,end],source_frame=fid,
                chunk_depth_frames=position+1,chunk_seconds=round(time.monotonic()-begin,1),
                candidate_voxels=len(cached.points),invalid_frames=bad)
    if bad>max(2,int(len(rows)*.05)):
        raise RuntimeError(f'Mouse {mouse} chunk {start}: {bad}/{len(rows)} invalid hulls; not publishing. Inspect reviews.')
    write_jsonl_atomic(chunk/'frames.jsonl',rows)
    summary={'frame_range':[start,end],'frame_count':len(rows),'views':6,'invalid_depth_frames':bad,
        'source_frame_step':4,'fps':25.,
        'warning_frames':sum(bool(r['warnings']) for r in metrics),'manual_seed_frame':seed,
        'hull_bounds_mm':[low.tolist(),high.tolist()],'method':METHOD,'validated':True}
    write_json_atomic(chunk/'chunk_complete.json',summary)
    del cached
    torch.cuda.empty_cache()
    publish(session,dst)
    status(phase='chunk_published',mouse=mouse,chunk=[start,end],published_chunk_frames=len(rows))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chunk-size',type=int,default=400)
    parser.add_argument('--max-chunks',type=int,default=0)
    parser.add_argument('--mouse',type=int,choices=(1,2),nargs='+',default=[1,2])
    args=parser.parse_args()
    assert args.chunk_size>=312 and args.chunk_size%4==0,'Source-frame chunk span must include manual frame 307 and be divisible by 4'
    OUT.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (OUT/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            status(phase='starting',complete=False,error=None)
            needed=0
            for mouse in args.mouse:
                _,dst,_,_,_,_=prepare(mouse,args.chunk_size)
                done=sum(read(p)['frame_count'] for p in dst.glob('chunks/*/chunk_complete.json'))
                needed+=(4500-done)*6*1152*1024*4
            assert shutil.disk_usage(BASE).free>needed*1.1,'Not enough space for uncompressed depth plus safety margin'
            cv2.setNumThreads(1)
            torch.set_num_threads(4)
            torch.backends.cuda.matmul.allow_tf32=True
            from sam2.build_sam import build_sam2_video_predictor
            predictor=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_s.yaml',str(CHECKPOINT),device='cuda')
            count=0
            with ThreadPoolExecutor(max_workers=6) as executor:
                for start in range(0,18000,args.chunk_size):
                    end=min(17996,start+args.chunk_size-4)
                    for mouse in args.mouse:
                        done=session_path(mouse)/f'pseudo_depth_full_v1/chunks/f{start:06d}-{end:06d}/chunk_complete.json'
                        if done.exists():
                            continue
                        process_chunk(mouse,start,end,predictor,executor,args)
                        count+=1
                        if args.max_chunks and count>=args.max_chunks:
                            status(phase='bounded_run_finished',complete=False)
                            return
            for mouse in args.mouse:
                publish(session_path(mouse),session_path(mouse)/'pseudo_depth_full_v1')
            if set(args.mouse)=={1,2}:
                status(phase='final_file_and_loader_validation',complete=False)
                subprocess.run([sys.executable,str(BASE/'tools/validate_dannce_25fps.py'),
                    '--require-full-depth'],cwd=BASE,check=True)
            status(phase='complete',complete=True)
        except Exception:
            status(phase='failed',complete=False,error=traceback.format_exc())
            raise


if __name__=='__main__':
    main()
