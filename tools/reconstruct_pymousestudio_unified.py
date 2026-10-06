"""Unified body/limb/tail z-buffer rendering and bounded bottom uncertainty.

Never paste depth layers. Keep the old geometry, reconstruct its tail and limb
additions exactly, then add only main-body-connected, side-supported voxels in
a small neighborhood. Outputs remain unverified pseudo data.
"""
from pathlib import Path
import argparse,itertools,json,sys
import cv2
import numpy as np
from scipy.ndimage import label,distance_transform_edt

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from add_dannce_real_multiview_pseudo_depth import project,rasterize_depth,read_pfm,write_pfm
from export_pymousestudio_rectified import write
from repair_pymousestudio_limb_hull import connected_additions,component_metrics

cv2.setNumThreads(1)


def seam_stats(before,after,added):
    values=[[],[]]
    for axis in (0,1):
        p=[slice(None),slice(None)];q=p.copy();p[axis]=slice(None,-1);q[axis]=slice(1,None)
        p,q=tuple(p),tuple(q)
        good=(before[p]>0)&(before[q]>0)&(after[p]>0)&(after[q]>0)&(added[p]^added[q])
        values[0].extend(abs(before[p]-before[q])[good].tolist())
        values[1].extend(abs(after[p]-after[q])[good].tolist())
    return dict(pairs=len(values[0]),before_p95_mm=float(np.percentile(values[0],95)) if values[0] else None,
                after_p95_mm=float(np.percentile(values[1],95)) if values[1] else None)


def overlay(rgb,mask,color):
    im=rgb.copy();im[mask]=(.55*im[mask]+.45*np.array(color)).astype(np.uint8)
    return im


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--frames',nargs='+',type=int,default=list(range(780,812)))
    ap.add_argument('--output',default='reconstruction_unified_guarded_v1');args=ap.parse_args()
    s=ROOT/'dataset/PyMouseStudio/sessions/AZL3';prev=s/'reconstruction_limb_guarded_v1'
    body=s/'reconstruction_bottom_refined_v1';track=s/'tail_dlc_video_masks_v1'
    out=s/args.output;out.mkdir(exist_ok=False);(out/'review').mkdir()
    cal=json.loads((s/'calibration.json').read_text());cfg=json.loads((body/'config.json').read_text())
    K={n:np.array(cal[n]['K']) for n in 'ABCDEFG'};E={n:np.array(cal[n]['world_to_camera_mm']) for n in 'ABCDEFG'}
    centers={n:-E[n][:3,:3].T@E[n][:3,3] for n in 'ABCDEFG'}
    roots={r['frame']:np.array(r['xyz_mm']) for r in json.loads((s/'tail_dlc_audit_v1/report.json').read_text())['records'] if r['id']==24}
    lo,hi=np.array(cfg['bounds_mm']);axes=[np.arange(lo[i],hi[i]+.5) for i in range(3)];shape=tuple(map(len,axes))
    points=np.stack(np.meshgrid(*axes,indexing='ij'),-1).reshape(-1,3).astype(np.float32)
    cache={}
    for n in 'ABCDEFG':
        uv,z=project(points,K[n],E[n]);x,y=np.rint(uv).astype(int).T
        cache[n]=(np.clip(y,0,1199)*1080+np.clip(x,0,1079),(z>0)&(x>=0)&(x<1080)&(y>=0)&(y<1200))
    rows=[];indices=[];writer=None;fwriter=None
    for f in args.frames:
        scene=out/f'frame_{f:06d}';scene.mkdir()
        masks={n:cv2.imread(str(s/f'pilot_input_masks/{f:06d}/cam{n}.png'),0) for n in 'BCDEFG'}
        olda=cv2.imread(str(body/f'frame_{f:06d}/bottom_sam2_mask.png'),0)
        tail=cv2.imread(str(track/f'tail_roi/{f:06d}.png'),0)
        extension=cv2.imread(str(prev/f'frame_{f:06d}/bottom_limb_mask_extension.png'),0)
        masks['A']=olda|tail|extension
        hit={n:(masks[n].ravel()[cache[n][0]]>0)&cache[n][1] for n in 'ABCDEFG'}
        votes=sum(hit[n].astype(np.uint8) for n in 'BCDEFG')
        ai,ag=cache['A'];old=(votes>=5)&(olda.ravel()[ai]>0)&ag
        wide=np.zeros(len(points),bool)
        for a,b in itertools.combinations('BCDEFG',2):
            u=centers[a]-roots[f];v=centers[b]-roots[f]
            angle=np.degrees(np.arccos(np.clip(u@v/np.linalg.norm(u)/np.linalg.norm(v),-1,1)))
            if angle>=60:wide|=hit[a]&hit[b]
        tail_prop=(votes>=3)&(tail.ravel()[ai]>0)&ag&wide
        labels,_=label((old|tail_prop).reshape(shape),np.ones((3,3,3)))
        touched=np.unique(labels.ravel()[old]);tail_added=tail_prop&~old&np.isin(labels.ravel(),touched[touched>0])
        limb_prop=(votes>=5)&(extension.ravel()[ai]>0)&ag
        limb_added=connected_additions(old,limb_prop,shape)
        unified=old|tail_added|limb_added
        # Only camera-F body-silhouette disagreements are allowed to trigger the
        # extra 3D search. Opening defines an ROI; it does not fill the silhouette.
        before_f=read_pfm(prev/f'frame_{f:06d}/rendered_depth_maps/00000005.pfm')
        body_f=cv2.morphologyEx(masks['F'],cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(61,61)))
        search_f=(body_f>0)&(before_f==0)
        fi,fg=cache['F']
        dist3d=distance_transform_edt(~old.reshape(shape)).ravel()
        dista=cv2.distanceTransform(np.uint8(masks['A']==0),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        soft_prop=(votes>=5)&hit['F']&search_f.ravel()[fi]&fg&ag&(dist3d<=3.)&(dista.ravel()[ai]<=40.)&~hit['A']
        soft=connected_additions(old,soft_prop,shape)&~unified
        # Guard against gross expansion; do not silently retune the budget.
        assert soft.sum()<=.15*max(1,old.sum()),'Bottom relaxation exceeds 15% body-volume review cap'
        selected=unified|soft
        cloud=points[selected]
        np.savez_compressed(scene/'geometry.npz',points_world_mm=cloud,
             bottom_relaxed=soft[selected],tail_added=tail_added[selected],limb_added=limb_added[selected])
        panels=[[],[],[]];fpanels=[];views=[]
        for v,n in enumerate('ABCDEFG'):
            before=read_pfm(prev/f'frame_{f:06d}/rendered_depth_maps/{v:08d}.pfm')
            control=rasterize_depth(points[unified],masks[n],K[n],E[n],1.)
            after=rasterize_depth(cloud,masks[n],K[n],E[n],1.)
            # No old depth paste-back: one common cloud, one z-buffer per camera.
            write_pfm(scene/f'rendered_depth_maps/{v:08d}.pfm',after)
            uncertain=rasterize_depth(points[soft],masks[n],K[n],E[n],1.)>0 if soft.any() else np.zeros_like(after,bool)
            for folder,img in [('masks',after>0),('changed_pixels',abs(after-before)>.01),('bottom_relaxed_projection',uncertain)]:
                path=scene/folder/f'{v:08d}.png';path.parent.mkdir(exist_ok=True);cv2.imwrite(str(path),np.uint8(img)*255)
            src=masks[n]>0;oldseam=cv2.imread(str(prev/f'frame_{f:06d}/limb_added_pixels/{v:08d}.png'),0)>0
            view=dict(camera=n,view_id=v,coverage_before=float(np.count_nonzero((before>0)&src)/max(1,src.sum())),
                coverage_unified=float(np.count_nonzero(control>0)/max(1,src.sum())),coverage_after=float(np.count_nonzero(after>0)/max(1,src.sum())),
                foreground_lost=int(np.count_nonzero((before>0)&(after==0))),foreground_added=int(np.count_nonzero((before==0)&(after>0))),
                components_before=component_metrics(before>0),components_after=component_metrics(after>0),
                seam_unified=seam_stats(before,control,oldseam),seam_final=seam_stats(before,after,oldseam))
            if n=='A':view['tail_roi_coverage']=float(np.count_nonzero((tail>0)&(after>0))/max(1,np.count_nonzero(tail)))
            views.append(view)
            rgb=cv2.imread(str(s/f'pilot_rgb/{f:06d}/cam{n}.png'))
            oldim=overlay(rgb,before>0,(0,130,255));newim=overlay(rgb,after>0,(60,255,0))
            dep=cv2.applyColorMap(np.uint8(np.clip((after-100)/220*255,0,255)),cv2.COLORMAP_TURBO);dep[after==0]=255
            for row,im,title in zip(panels,[oldim,newim,dep],[f'{n} before {f}',f'{n} unified + local',f'{n} Z mm 100-320']):
                resized=cv2.resize(im,(270,300));cv2.putText(resized,title,(5,18),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1);row.append(resized)
            if n=='F':
                for im,title in [(rgb,'RGB'),(oldim,'Before'),(overlay(rgb,control>0,(60,255,0)),'Unified only'),(newim,'Unified + local'),(dep,'Depth 100-320mm')]:
                    im=cv2.resize(im[200:1100,220:960],(370,450));cv2.rectangle(im,(0,0),(369,32),(35,35,35),-1)
                    cv2.putText(im,title,(7,22),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1);fpanels.append(im)
                if f in [780,802,803,808,811]:
                    local=[]
                    for im in [rgb,oldim,newim]:local.append(im[850:1070,310:600])
                    cv2.imwrite(str(out/f'review/camF_legs_{f:06d}.png'),np.concatenate(local,1))
            if f in [780,802,811]:
                p=scene/'control_depth'/f'{v:08d}.pfm';write_pfm(p,control)
        montage=np.concatenate([np.concatenate(row,1) for row in panels],0)
        fmontage=np.concatenate(fpanels,1)
        cv2.imwrite(str(out/f'review/{f:06d}.png'),montage);cv2.imwrite(str(out/f'review/camF_{f:06d}.png'),fmontage)
        if writer is None:
            writer=cv2.VideoWriter(str(out/'review/multiview_8fps.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),8,(1890,900))
            fwriter=cv2.VideoWriter(str(out/'review/camF_8fps.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),8,(1850,450))
        writer.write(montage);fwriter.write(fmontage)
        row=dict(frame=f,body_voxels=int(old.sum()),tail_added_voxels=int(tail_added.sum()),limb_added_voxels=int(limb_added.sum()),
                 bottom_relaxed_voxels=int(soft.sum()),views=views);rows.append(row)
        indices.append(dict(source_frame=f,time_seconds=f/52,train_allowed=False,
            views=[dict(camera=n,view_id=v,depth=str((scene/f'rendered_depth_maps/{v:08d}.pfm').relative_to(s)),
                   mask=str((scene/f'masks/{v:08d}.png').relative_to(s))) for v,n in enumerate('ABCDEFG')]))
        write(out/'progress.json',dict(completed=len(rows),total=len(args.frames)))
        print(f,'soft',int(soft.sum()),'F coverage',views[5]['coverage_before'],views[5]['coverage_after'],'seam',views[5]['seam_final'],flush=True)
    writer.release();fwriter.release()
    write(out/'report.json',dict(state='complete',frames=rows,train_allowed=False))
    write(out/'config.json',dict(method='one merged geometry and one z-buffer; bounded local bottom uncertainty',
        frames=args.frames,depth_unit='mm',depth_semantics='camera_z',size_wh=[1080,1200],voxel_mm=1.,
        bottom_soft_radius_3d_mm=3.,bottom_soft_radius_image_px=40.,minimum_side_votes=5,
        trigger='camF missing broad body silhouette; no tail or limb ROI expansion',max_added_body_volume_fraction=.15,
        final_silhouette_clipping=True,depth_layer_pasting=False,train_allowed=False,
        limitation='Local hypothesis, not calibration correction or independent ground truth. Bottom-relaxed projection is uncertainty, not confidence.'))
    (out/'frames.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in indices))


if __name__=='__main__':main()
