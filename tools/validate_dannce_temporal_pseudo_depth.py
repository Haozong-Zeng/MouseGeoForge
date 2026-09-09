"""Continuous SAM2 tracking + calibrated visual-hull depth, with held-out labels."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import cv2
import numpy as np
import torch

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'BlenderScripts'))
from convert_dannce_mouse_to_blenderpymouse import load_session
from add_dannce_real_multiview_pseudo_depth import (project, prompts_for_view,
    visual_hull, rasterize_depth, write_pfm, read_pfm)
from segment_dannce_4dgs_sam2 import clean_mask, overlay
from core23_temporal_format import write_json_atomic, write_jsonl_atomic


def track(session,dst,start,count,seed,labels,cameras,predictor,box_padding):
    for view,camera in enumerate(cameras):
        marker=dst/f'masks_view{view}_complete.json'
        if marker.exists():
            old=json.loads(marker.read_text())
            if old.get('box_padding_px',18.)==box_padding:
                continue
        numeric=dst/'tracking_rgb'/f'{view:08d}'
        numeric.mkdir(parents=True,exist_ok=True)
        masks=dst/'masks'/f'{view:08d}'
        masks.mkdir(parents=True,exist_ok=True)
        for i in range(count):
            src=session/'rgb'/f'{view:08d}/{start+i:06d}.jpg'
            if not src.exists():
                raise FileNotFoundError(f'Full RGB export has not reached {src}')
            target=numeric/f'{i:06d}.jpg'
            if not target.exists():
                shutil.copy2(src,target)
        K,E=np.asarray(camera['intrinsic']),np.asarray(camera['extrinsic'])
        xyz=labels[seed]
        valid=np.isfinite(xyz).all(-1)&(np.linalg.norm(xyz,axis=-1)>1e-5)
        uv,z=project(xyz,K,E)
        valid &= (z>0)&(uv[:,0]>=0)&(uv[:,0]<1152)&(uv[:,1]>=0)&(uv[:,1]<1024)
        box,points=prompts_for_view(uv,valid,1152,1024,box_padding)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            state=predictor.init_state(video_path=str(numeric),offload_video_to_cpu=True,
                offload_state_to_cpu=False,async_loading_frames=False)
            predictor.add_new_points_or_box(inference_state=state,frame_idx=seed-start,
                obj_id=1,points=points,labels=np.ones(len(points),np.int32),box=box)
            written=set()
            for reverse in (False,True):
                for index,object_ids,logits in predictor.propagate_in_video(state,
                        start_frame_idx=seed-start,reverse=reverse):
                    mask=clean_mask((logits[object_ids.index(1),0]>0).cpu().numpy())
                    if not cv2.imwrite(str(masks/f'{start+index:06d}.png'),mask):
                        raise IOError(masks)
                    written.add(index)
            predictor.reset_state(state)
        assert len(written)==count
        write_json_atomic(marker,{'seed_source_frame':seed,'frame_count':count,'box_padding_px':box_padding,
            'held_out_labels_not_used':sorted(f for f in labels if start<=f<start+count and f!=seed)})
        print(f'SAM2 complete {session.name} view {view}',flush=True)


def fixed_depth_color(depth,near,far):
    gray=np.uint8(np.clip((depth-near)/max(far-near,1),0,1)*255)
    image=cv2.applyColorMap(gray,cv2.COLORMAP_TURBO)
    image[depth<=0]=0
    return image


def silhouette_confidence(depth,masks,Ks,Es,source):
    yy,xx=np.nonzero(depth>0)
    z=depth[yy,xx]
    ray=np.stack((xx,yy,np.ones_like(xx)),axis=-1)@np.linalg.inv(Ks[source]).T
    xyz=(ray*z[:,None]-Es[source][:3,3])@Es[source][:3,:3]
    votes=np.zeros(len(z),np.uint8)
    for mask,K,E in zip(masks,Ks,Es):
        uv,pz=project(xyz,K,E)
        xy=np.rint(uv).astype(int)
        inside=(pz>0)&(xy[:,0]>=0)&(xy[:,0]<1152)&(xy[:,1]>=0)&(xy[:,1]<1024)
        votes[inside]+=(mask[xy[inside,1],xy[inside,0]]>0).astype(np.uint8)
    confidence=np.zeros(depth.shape,np.uint8)
    confidence[yy,xx]=np.uint8(votes.astype(float)/6*255)
    return confidence


def process(mouse,start,count,predictor,box_padding,minimum_views,bounds_padding):
    root=BASE/f'dataset/DANNCE-MarkerlessMouse{mouse}-Core23-RealMultiview'
    session=root/f'sessions/markerless_mouse_{mouse}'
    dst=session/f'pseudo_depth_pilot/f{start:06d}-{start+count-1:06d}'
    dst.mkdir(parents=True,exist_ok=True)
    meta=json.loads((session/'metadata.json').read_text())
    cameras=meta['cameras']
    raw=load_session(session/'label3d_dannce.mat')
    labels={int(f):xyz for f,xyz in zip(raw['source_frames'],raw['source_xyz'])}
    overlap=sorted(f for f in labels if start<=f<start+count)
    seed=overlap[0]
    track(session,dst,start,count,seed,labels,cameras,predictor,box_padding)
    Ks=[np.asarray(c['intrinsic']) for c in cameras]
    Es=[np.asarray(c['extrinsic']) for c in cameras]
    com=np.load(session/'sync_com.npz')['com_world_xyz_mm']
    seed_xyz=labels[seed]
    seed_xyz=seed_xyz[np.isfinite(seed_xyz).all(-1)&(np.linalg.norm(seed_xyz,axis=-1)>1e-5)]
    delta=com[start:start+count]-com[seed]
    low=np.floor((seed_xyz.min(0)+delta.min(0)-bounds_padding)/2)*2
    high=np.ceil((seed_xyz.max(0)+delta.max(0)+bounds_padding)/2)*2
    bounds=np.stack([low,high])
    # Use a fixed world lattice across time; COM/seed only bound it, not deform the pose.
    review=BASE/'output/dannce_temporal_depth_pilot_v1'/f'mouse{mouse}'
    review.mkdir(parents=True,exist_ok=True)
    stats=[]
    previous_masks=previous_depths=previous_gray=None
    writer=cv2.VideoWriter(str(review/'review_10fps.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10.,(1152,768))
    if not writer.isOpened():
        raise IOError('Cannot create review video')
    index=[]
    color_limits=[]
    for K,E in zip(Ks,Es):
        _,zs=project(np.array([[x,y,z] for x in (low[0],high[0]) for y in (low[1],high[1]) for z in (low[2],high[2])]),K,E)
        color_limits.append((float(zs.min()),float(zs.max())))
    for fid in range(start,start+count):
        masks=[cv2.imread(str(dst/'masks'/f'{v:08d}/{fid:06d}.png'),0) for v in range(6)]
        hull,hstats=visual_hull(bounds,masks,Ks,Es,2.,0.,minimum_views)
        depths=[rasterize_depth(hull,m,K,E,2.) for m,K,E in zip(masks,Ks,Es)]
        record={'source_frame_id':fid,**hstats,'views':[],
            'hull_touches_bounds':bool(np.any((hull<=low+2)|(hull>=high-2)))}
        panels=[]
        gray=[]
        for v,(mask,depth,K,E) in enumerate(zip(masks,depths,Ks,Es)):
            folder=dst/'rendered_depth_maps'/f'{v:08d}'
            folder.mkdir(parents=True,exist_ok=True)
            write_pfm(folder/f'{fid:06d}.pfm',depth)
            confidence=silhouette_confidence(depth,masks,Ks,Es,v)
            confidence_dir=dst/'confidence'/f'{v:08d}'
            confidence_dir.mkdir(parents=True,exist_ok=True)
            cv2.imwrite(str(confidence_dir/f'{fid:06d}.png'),confidence)
            image=cv2.imread(str(session/'rgb'/f'{v:08d}/{fid:06d}.jpg'))
            g=cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),(384,342))
            gray.append(g)
            vr={'view_id':v,'mask_area_px':int(np.count_nonzero(mask)),
                'depth_coverage':float(np.count_nonzero(depth)/max(np.count_nonzero(mask),1)),
                'depth_median_mm':float(np.median(depth[depth>0])),
                'confidence5_fraction':float(np.count_nonzero(confidence>=212)/max(np.count_nonzero(depth),1))}
            if fid in labels:
                xyz=labels[fid]
                uv,z=project(xyz,K,E)
                xy=np.rint(np.nan_to_num(uv,nan=-10000,posinf=-10000,neginf=-10000)).astype(int)
                good=np.isfinite(xyz).all(-1)&(np.linalg.norm(xyz,axis=-1)>1e-5)&(z>0)&(xy[:,0]>=0)&(xy[:,0]<1152)&(xy[:,1]>=0)&(xy[:,1]<1024)
                hit=mask[xy[good,1],xy[good,0]]>0
                vr.update(manual_keypoint_count=int(good.sum()),manual_keypoints_in_mask=int(hit.sum()),
                          is_prompt_frame=fid==seed)
            if previous_masks is not None:
                a,b=mask>0,previous_masks[v]>0
                vr['adjacent_mask_iou']=float(np.count_nonzero(a&b)/max(np.count_nonzero(a|b),1))
                # Backward image flow maps current pixels to previous pixels.
                flow=cv2.calcOpticalFlowFarneback(g,previous_gray[v],None,.5,3,15,3,5,1.2,0)
                gy,gx=np.mgrid[:342,:384].astype(np.float32)
                pm=cv2.resize(previous_masks[v],(384,342),interpolation=cv2.INTER_NEAREST)
                warped=cv2.remap(pm,gx+flow[:,:,0],gy+flow[:,:,1],cv2.INTER_NEAREST)>0
                current=cv2.resize(mask,(384,342),interpolation=cv2.INTER_NEAREST)>0
                vr['flow_warp_mask_iou']=float(np.count_nonzero(warped&current)/max(np.count_nonzero(warped|current),1))
                pd=cv2.resize(previous_depths[v],(384,342),interpolation=cv2.INTER_NEAREST)
                warped_z=cv2.remap(pd,gx+flow[:,:,0],gy+flow[:,:,1],cv2.INTER_NEAREST)
                current_z=cv2.resize(depth,(384,342),interpolation=cv2.INTER_NEAREST)
                valid=(warped_z>0)&(current_z>0)&warped&current
                vr['flow_warp_depth_median_abs_delta_mm']=float(np.median(np.abs(warped_z[valid]-current_z[valid]))) if valid.any() else None
            record['views'].append(vr)
            rgb_tile=cv2.resize(overlay(image,mask),(288,256))
            depth_tile=cv2.resize(fixed_depth_color(depth,*color_limits[v]),(288,256))
            cv2.putText(rgb_tile,f'view{v} f{fid}',(4,15),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
            panels.append(np.concatenate([rgb_tile,depth_tile],axis=1))
        # Six view-pairs: two columns, three rows. Preserve RGB aspect ratio.
        grid=np.concatenate([np.concatenate(panels[i:i+2],axis=1) for i in (0,2,4)],axis=0)
        writer.write(grid)
        if fid in {start,start+count//2,start+count-1,*overlap}:
            cv2.imwrite(str(review/f'frame_{fid:06d}.jpg'),grid)
        stats.append(record)
        index.append({'source_frame_id':fid,'rgb':[f'../../rgb/{v:08d}/{fid:06d}.jpg' for v in range(6)],
            'mask':[f'masks/{v:08d}/{fid:06d}.png' for v in range(6)],
            'depth':[f'rendered_depth_maps/{v:08d}/{fid:06d}.pfm' for v in range(6)],
            'confidence':[f'confidence/{v:08d}/{fid:06d}.png' for v in range(6)]})
        previous_masks,previous_depths,previous_gray=masks,depths,gray
        if (fid-start)%20==0:
            print(f'Depth {session.name} {fid} voxels={len(hull)}',flush=True)
    writer.release()
    write_jsonl_atomic(dst/'frames.jsonl',index)
    write_json_atomic(review/'per_frame_metrics.json',stats)
    flat=[v for s in stats for v in s['views']]
    summary={'mouse':mouse,'source_frame_range':[start,start+count-1],'frame_count':count,
        'views':6,'seed_source_frame':seed,'seed_box_padding_px':box_padding,
        'bounds_padding_mm':bounds_padding,
        'held_out_manual_frames':[f for f in overlap if f!=seed],
        'resolution':[1152,1024],'depth_unit':'mm','depth_semantics':'camera_z',
        'pfm_flip_y':True,'depth_is_pseudo_label':True,'mask_is_pseudo_label':True,
        'method':f'SAM2.1 video tracking + calibrated {minimum_views}-of-6 silhouette visual hull 2mm fixed world lattice',
        'confidence_semantics':'0..255 = reprojected silhouette votes / 6; >=212 means >=5/6 views; not calibrated probability',
        'hull_boundary_touch_frames':sum(s['hull_touches_bounds'] for s in stats),
        'metrics':{},'review_playback_fps':10.,'source_fps':100.,
        'validation_limit':'Held-out keypoint containment, optical-flow temporal consistency and multiview silhouette support are not ground-truth surface-depth accuracy.',
        'training_status':'experimental_foreground_pseudo_labels_requires_confidence_gating_not_full_depth_GT'}
    for key in ('depth_coverage','confidence5_fraction','adjacent_mask_iou','flow_warp_mask_iou','flow_warp_depth_median_abs_delta_mm'):
        vals=[v[key] for v in flat if v.get(key) is not None]
        summary['metrics'][key]={'min':float(np.min(vals)),'median':float(np.median(vals)),
            'p05':float(np.percentile(vals,5)),'p95':float(np.percentile(vals,95)),'max':float(np.max(vals))}
    heldout=[v for v in flat if v.get('is_prompt_frame') is False]
    summary['held_out_keypoint_containment']={'inside':sum(v['manual_keypoints_in_mask'] for v in heldout),
        'total':sum(v['manual_keypoint_count'] for v in heldout)}
    write_json_atomic(dst/'metadata.json',summary)
    write_json_atomic(review/'summary.json',summary)
    print(json.dumps(summary),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mouse',type=int,choices=(1,2),required=True)
    parser.add_argument('--count',type=int,default=200)
    parser.add_argument('--box-padding',type=float,default=None)
    parser.add_argument('--minimum-views',type=int,choices=(4,5,6),default=6)
    parser.add_argument('--bounds-padding',type=float,default=None)
    args=parser.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True
    from sam2.build_sam import build_sam2_video_predictor
    predictor=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_s.yaml',
        str(BASE/'third_party/checkpoints/sam2.1_hiera_small.pt'),device='cuda')
    padding=args.box_padding if args.box_padding is not None else (18. if args.mouse==1 else 96.)
    bounds_padding=args.bounds_padding if args.bounds_padding is not None else (50. if args.mouse==1 else 30.)
    process(args.mouse,0 if args.mouse==1 else 800,args.count,predictor,padding,args.minimum_views,bounds_padding)


if __name__=='__main__':
    main()
