"""Conservative tail-only silhouette relaxation, anchored by reliable DLC root.

No guessed tail curve, no forced low-confidence point interpolation. Preserve
the existing body hull. Any added geometry is separately flagged pseudo data.
"""
from pathlib import Path
import sys,json,itertools,argparse
import numpy as np
import cv2
import torch
from scipy.ndimage import label
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from add_dannce_real_multiview_pseudo_depth import project,rasterize_depth,write_pfm,read_pfm
from reconstruct_pymousestudio_pilot import write_ply
from export_pymousestudio_rectified import write
cv2.setNumThreads(1)


def tail_roi(mask,root_uv,next_uv=None):
    """Separate thin posterior component from broad body; never draw a curve."""
    opened=cv2.morphologyEx(mask,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(61,61)))
    body=cv2.dilate(opened,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13)))
    residual=np.uint8((mask>0)&(body==0))
    yy,xx=np.nonzero(opened)
    if len(xx)<100:return np.zeros_like(mask),dict(valid=False,reason='no broad body')
    center=np.array([np.median(xx),np.median(yy)])
    direction=(next_uv-root_uv) if next_uv is not None else (root_uv-center)
    norm=np.linalg.norm(direction)
    if norm<3:return np.zeros_like(mask),dict(valid=False,reason='ambiguous posterior direction')
    direction=direction/norm;gy,gx=np.indices(mask.shape)
    residual[((gx-root_uv[0])*direction[0]+(gy-root_uv[1])*direction[1]) < -15]=0
    count,ids,stats,_=cv2.connectedComponentsWithStats(residual,8)
    choices=[]
    target=next_uv if next_uv is not None else root_uv
    for i in range(1,count):
        y,x=np.nonzero(ids==i);area=len(x)
        if area<120:continue
        extent=max(int(np.ptp(x)),int(np.ptp(y)))
        distance=float(np.sqrt(((np.stack([x,y],-1)-target)**2).sum(-1)).min())
        if extent<100 or distance>110:continue
        choices.append((area/(1+distance/80),i,area,extent,distance))
    if not choices:return np.zeros_like(mask),dict(valid=False,reason='no long thin posterior component')
    _,i,area,extent,distance=max(choices)
    return np.uint8(ids==i)*255,dict(valid=True,area=area,extent_px=extent,seed_distance_px=distance,
                                  direction_source='triangulated tail0' if next_uv is not None else 'root minus body centroid')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--video-tail',action='store_true');args=parser.parse_args()
    session=ROOT/'dataset/PyMouseStudio/sessions/AZL3';previous=session/'reconstruction_bottom_refined_v1'
    out=session/('reconstruction_tail_guarded_v2' if args.video_tail else 'reconstruction_tail_guarded_v1');out.mkdir(exist_ok=False)
    config=json.loads((previous/'config.json').read_text());cal=json.loads((session/'calibration.json').read_text())
    audit=json.loads((session/'tail_dlc_audit_v1/report.json').read_text())
    records={(r['frame'],r['id']):r for r in audit['records']}
    names=list('ABCDEFG');Ks={n:np.array(cal[n]['K']) for n in names};Es={n:np.array(cal[n]['world_to_camera_mm']) for n in names}
    centers={n:-Es[n][:3,:3].T@Es[n][:3,3] for n in names}
    low,high=np.array(config['bounds_mm']);axes=[np.arange(low[i],high[i]+.5,1.) for i in range(3)]
    shape=tuple(len(a) for a in axes);points=np.stack(np.meshgrid(*axes,indexing='ij'),-1).reshape(-1,3).astype(np.float32)
    cache={}
    for n in names:
        uv,z=project(points,Ks[n],Es[n]);x,y=np.rint(uv).astype(np.int32).T
        good=(z>0)&(x>=0)&(x<1080)&(y>=0)&(y<1200)
        ids=np.clip(y,0,1199).astype(np.int64)*1080+np.clip(x,0,1079)
        cache[n]=(torch.from_numpy(ids).cuda(),torch.from_numpy(good).cuda())
    reviews=out/'review';reviews.mkdir();report=[];writer=None;all_indices=[]
    for frame in config['frame_indices']:
        root=np.array(records[frame,24]['xyz_mm'],float)
        assert root.shape==(3,) and np.isfinite(root).all()
        next_point=records[frame,25]['xyz_mm']
        root_uv=project(root[None],Ks['A'],Es['A'])[0][0]
        next_uv=project(np.array(next_point)[None],Ks['A'],Es['A'])[0][0] if next_point is not None else None
        masks={n:cv2.imread(str(session/f'pilot_input_masks/{frame:06d}/cam{n}.png'),0) for n in 'BCDEFG'}
        masks['A']=cv2.imread(str(previous/f'frame_{frame:06d}/bottom_sam2_mask.png'),0)
        original_bottom=masks['A'].copy()
        roi,roi_meta=tail_roi(masks['A'],root_uv,next_uv)
        if args.video_tail:
            tracked=session/'tail_dlc_video_masks_v1'
            roi=cv2.imread(str(tracked/f'tail_roi/{frame:06d}.png'),0)
            masks['A']=cv2.imread(str(tracked/f'tail_augmented_masks/{frame:06d}.png'),0)
            assert roi is not None and masks['A'] is not None
            roi_meta=dict(valid=bool(roi.any()),area=int(np.count_nonzero(roi)),
                          source='SAM2 video with geometry-consistent DLC tip anchors at783/796; tail-only merge')
        scene=out/f'frame_{frame:06d}';scene.mkdir();cv2.imwrite(str(scene/'bottom_tail_roi.png'),roi)
        hit={}
        for n in names:
            ids,good=cache[n];m=torch.from_numpy(masks[n].reshape(-1)).cuda();hit[n]=(m[ids]>127)&good
        votes=sum(hit[n].to(torch.uint8) for n in 'BCDEFG')
        ids,good=cache['A'];original_hit=(torch.from_numpy(original_bottom.reshape(-1)).cuda()[ids]>127)&good
        old=((votes>=5)&original_hit).cpu().numpy()
        ids,good=cache['A'];m=torch.from_numpy(roi.reshape(-1)).cuda();tail=(m[ids]>127)&good
        wide=torch.zeros(len(points),dtype=torch.bool,device='cuda');pairs=[]
        for a,b in itertools.combinations('BCDEFG',2):
            u=centers[a]-root;v=centers[b]-root
            angle=float(np.degrees(np.arccos(np.clip(np.dot(u,v)/np.linalg.norm(u)/np.linalg.norm(v),-1,1))))
            if angle>=60:wide|=hit[a]&hit[b];pairs.append([a,b])
        proposed=((votes>=3)&tail&wide).cpu().numpy()
        union=old|proposed
        # Added components must touch existing occupied geometry. No floating tube.
        components,count=label(union.reshape(shape),structure=np.ones((3,3,3)))
        touched=np.unique(components.reshape(-1)[old]);keep=np.isin(components.reshape(-1),touched[touched>0])
        added=proposed&~old&keep
        selected=old|added;cloud=points[selected]
        scene_meta=[];fig,axs=plt.subplots(3,7,figsize=(21,10),layout='constrained')
        for v,n in enumerate(names):
            rgb=cv2.cvtColor(cv2.imread(str(session/f'pilot_rgb/{frame:06d}/cam{n}.png')),cv2.COLOR_BGR2RGB)
            raw_depth=rasterize_depth(cloud,np.full((1200,1080),255,np.uint8),Ks[n],Es[n],1.)
            # Continue clipping to each actual input silhouette. The new tail only
            # relaxes occupancy constraints, not the final foreground boundary.
            depth=raw_depth.copy();depth[masks[n]==0]=0
            before=read_pfm(previous/f'frame_{frame:06d}/rendered_depth_maps/{v:08d}.pfm')
            if args.video_tail:
                # Avoid whole-body raster changes caused by the rasterizer's
                # cloud-median-dependent splat radius. A is strictly tail-only;
                # side views can change only near projections of added voxels.
                permitted=roi>0
                if n!='A':
                    permitted=np.zeros((1200,1080),np.uint8)
                    if added.any():
                        xy,z=project(points[added],Ks[n],Es[n]);x,y=np.rint(xy).astype(int).T
                        valid=(z>0)&(x>=0)&(x<1080)&(y>=0)&(y<1200)
                        permitted[y[valid],x[valid]]=255
                        permitted=cv2.dilate(permitted,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(45,45)))
                    permitted=permitted>0
                depth[~permitted]=before[~permitted]
            p=scene/'rendered_depth_maps'/f'{v:08d}.pfm';write_pfm(p,depth)
            assert np.array_equal(read_pfm(p),depth) and np.isfinite(depth).all()
            mask=np.uint8(depth>0)*255;p=scene/'masks'/f'{v:08d}.png';p.parent.mkdir(exist_ok=True);cv2.imwrite(str(p),mask)
            changed=(abs(depth-before)>.01);p=scene/'tail_changed_pixels'/f'{v:08d}.png';p.parent.mkdir(exist_ok=True);cv2.imwrite(str(p),np.uint8(changed)*255)
            old_overlay=rgb.copy();old_overlay[before>0]=(.6*old_overlay[before>0]+.4*np.array([255,130,0])).astype(np.uint8)
            new_overlay=rgb.copy();new_overlay[depth>0]=(.6*new_overlay[depth>0]+.4*np.array([0,255,80])).astype(np.uint8)
            axs[0,v].imshow(old_overlay);axs[0,v].set_title(f'cam{n} before (orange)')
            axs[1,v].imshow(new_overlay);axs[1,v].set_title('tail-only repair (green)',fontsize=9)
            cm=axs[2,v].imshow(np.ma.masked_where(depth==0,depth),cmap='turbo',vmin=100,vmax=320)
            for ax in axs[:,v]:ax.axis('off')
            scene_meta.append(dict(view_id=v,camera=n,foreground_before=int(np.count_nonzero(before)),
                   foreground_after=int(np.count_nonzero(depth)),changed_pixels=int(changed.sum()),
                   source_mask_pixels=int(np.count_nonzero(masks[n])),
                   depth=str((scene/'rendered_depth_maps'/f'{v:08d}.pfm').relative_to(session))))
        fig.colorbar(cm,ax=list(axs[2,:]),orientation='horizontal',shrink=.6,label='camera-Z mm; white invalid/background; tail additions NOT depth GT')
        fig.suptitle(f'Frame{frame} | body unchanged in 3D | DLC root + image-tail ROI + >=3 side masks + wide baseline + connectivity')
        p=reviews/f'{frame:06d}.png';fig.savefig(p,dpi=90);plt.close(fig)
        im=cv2.imread(str(p))
        if writer is None:writer=cv2.VideoWriter(str(reviews/'tail_repair_comparison_8fps.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),8,(im.shape[1],im.shape[0]))
        writer.write(im)
        # Inspect precisely the tail region, rather than whole-body IoU dominance.
        before=read_pfm(previous/f'frame_{frame:06d}/rendered_depth_maps/00000000.pfm')
        after=read_pfm(scene/'rendered_depth_maps/00000000.pfm');target=roi>0
        outside=((before>0)^(after>0))&~(cv2.dilate(roi,np.ones((21,21),np.uint8))>0)
        stats=dict(frame=frame,tail_roi=roi_meta,old_voxels=int(old.sum()),added_tail_voxels=int(added.sum()),
                   rejected_disconnected_added_voxels=int(np.count_nonzero(proposed&~old&~keep)),
                   tail_input_mask_coverage_before=float(np.count_nonzero(target&(before>0))/max(1,target.sum())),
                   tail_input_mask_coverage_after=float(np.count_nonzero(target&(after>0))/max(1,target.sum())),
                   bottom_changed_foreground_outside_tail_margin=int(outside.sum()),
                   views=scene_meta,train_allowed=False)
        report.append(stats)
        if frame in [780,796,811]:
            fig,axes=plt.subplots(1,4,figsize=(16,5),layout='constrained')
            rgb=cv2.cvtColor(cv2.imread(str(session/f'pilot_rgb/{frame:06d}/camA.png')),cv2.COLOR_BGR2RGB)
            for ax,sel,title,color in zip(axes,[None,before>0,after>0,target],
                       ['Rectified bottom RGB','Before foreground','Tail-only repaired','Image tail ROI + DLC root'],
                       [[0,0,0],[255,130,0],[0,255,80],[0,180,255]]):
                im=rgb.copy()
                if sel is not None:im[sel]=(.6*im[sel]+.4*np.array(color)).astype(np.uint8)
                ax.imshow(im);ax.axis('off');ax.set_title(title,fontsize=10)
            axes[-1].scatter(*root_uv,c='red',s=30)
            fig.savefig(reviews/f'bottom_tail_{frame:06d}.png',dpi=120);plt.close(fig)
            subset=cloud[::max(1,len(cloud)//100000)];write_ply(scene/'hull_voxels_mm.ply',subset,np.full((len(subset),3),140,np.uint8))
        all_indices.append(dict(source_frame=frame,time_seconds=frame/52,views=scene_meta,train_allowed=False))
        write(out/'progress.json',dict(completed=len(report),total=32));print('TAIL',frame,roi_meta,'added',int(added.sum()),flush=True)
    writer.release()
    write(out/'report.json',dict(state='complete',frames=report,
         mean_tail_coverage_before=float(np.mean([r['tail_input_mask_coverage_before'] for r in report])),
         mean_tail_coverage_after=float(np.mean([r['tail_input_mask_coverage_after'] for r in report])),
         metric_warning='Coverage of the input SAM2 tail mask, not independent accuracy.',
         video_tail_anchors_used=args.video_tail,
         full_tail_3d_points_reliable=False,training_supervision_enabled=False))
    write(out/'config.json',dict(method='image-supported tail-only silhouette relaxation',
         old_reconstruction=str(previous.relative_to(ROOT)),body_rule='bottom AND >=5/6 side masks, unchanged',
         tail_rule='bottom thin posterior component anchored by DLC root AND >=3/6 side masks AND pair angle>=60deg AND touches old 3D hull',
         root_confidence=.8,root_min_inlier_views=3,root_max_reprojection_px=15,tail_middle_points_used=False,
         bottom_video_tracking=args.video_tail,tail_tip_anchor_frames=[783,796] if args.video_tail else [],
         preserve_old_depth_outside_tail_footprint=args.video_tail,
         voxel_mm=1.,depth_unit='mm',depth_semantics='camera_z',size_wh=[1080,1200],
         bounds_mm=[low.tolist(),high.tolist()],train_allowed=False,
         limitation='Occluded tail still cannot be reconstructed merely by fitting low-confidence DLC points.'))
    (out/'frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in all_indices))


if __name__=='__main__':main()
