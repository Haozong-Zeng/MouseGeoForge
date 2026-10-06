"""Calibrated foreground visual-hull pilot; not scene depth or pose ground truth.

Reuse DANNCE PFM/rasterization rules. Native camera-Z mm, 1 mm lattice for
these close-range cameras. Triangulated DLC points determine search bounds
only; silhouettes, never a fitted skeleton, determine occupied geometry.
"""
from pathlib import Path
import sys,json,itertools,argparse
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mousegeoforge.raw_validation import rectify_points
from add_dannce_real_multiview_pseudo_depth import project,rasterize_depth,write_pfm,read_pfm
from export_pymousestudio_rectified import write
cv2.setNumThreads(1)


def triangulate(observations,Ks,Es):
    names=list(observations)
    if len(names)<3:return None,None
    def solve(chosen):
        rows=[]
        for n in chosen:
            u,v=(np.linalg.inv(Ks[n])@np.r_[observations[n],1])[:2];P=Es[n][:3]
            for row in [u*P[2]-P[0],v*P[2]-P[1]]:rows.append(row/np.linalg.norm(row[:3]))
        _,_,vh=np.linalg.svd(rows)
        if abs(vh[-1,3])<1e-8:return np.full(3,np.nan)
        return vh[-1,:3]/vh[-1,3]
    best=None
    for pair in itertools.combinations(names,2):
        X=solve(pair)
        if not np.isfinite(X).all():continue
        errors=[]
        for n in names:
            uv,z=project(X[None],Ks[n],Es[n]);errors.append(float(np.linalg.norm(uv[0]-observations[n])) if z[0]>0 else 1e6)
        inliers=[n for n,e in zip(names,errors) if e<15]
        score=(len(inliers),-float(np.median(errors)))
        if best is None or score>best[0]:best=(score,inliers)
    if best is None or len(best[1])<3:return None,None
    X=solve(best[1]);errors=[]
    for n in best[1]:
        uv,z=project(X[None],Ks[n],Es[n]);errors.append(float(np.linalg.norm(uv[0]-observations[n])))
    if not np.isfinite(X).all() or np.median(errors)>15:return None,None
    return X,float(np.median(errors))


def write_ply(path,points,colors):
    vertices=np.empty(len(points),dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1')])
    for i,n in enumerate(['x','y','z']):vertices[n]=points[:,i]
    for i,n in enumerate(['red','green','blue']):vertices[n]=colors[:,i]
    with path.open('wb') as f:
        f.write(('ply\nformat binary_little_endian 1.0\nelement vertex '+str(len(points))+'\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n').encode())
        vertices.tofile(f)


def load_clip(r,cal,out,frames):
    images={};masks={};uv={};conf={}
    for n,c in r['cameras'].items():
        K=np.array(cal[n]['K']);D=np.array(cal[n]['distortion'])
        mx,my=cv2.initUndistortRectifyMap(K,D,None,K,(1080,1200),cv2.CV_32FC1)
        for kind,src,interpolation in [('rgb',c['video'],cv2.INTER_LINEAR),('mask',c['mask'],cv2.INTER_NEAREST)]:
            if not src:continue
            cap=cv2.VideoCapture(str(ROOT/src));cap.set(cv2.CAP_PROP_POS_FRAMES,frames[0]);clip=[]
            for frame in frames:
                ok,bgr=cap.read();assert ok
                if kind=='mask':bgr=np.uint8(cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)>127)*255
                a=cv2.remap(bgr,mx,my,interpolation,borderMode=cv2.BORDER_CONSTANT);clip.append(a)
            cap.release()
            if kind=='rgb':images[n]=clip
            else:masks[n]=clip
        if c['keypoints']:
            with np.load(ROOT/c['keypoints'],allow_pickle=False) as a:
                uv[n]=rectify_points(a['uv'][frames],K,D);conf[n]=a['likelihood'][frames]
    return images,masks,uv,conf


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--frames',type=int,default=32);args=parser.parse_args()
    r=next(json.loads(l) for l in (ROOT/'dataset/PyMouseStudioRaw/_validation_v2/sequences.jsonl').read_text().splitlines() if json.loads(l)['sequence_id'].endswith('/AZL3'))
    assert r['split']=='real_development'
    cal=json.loads((ROOT/r['prepared_calibration']).read_text())
    out=ROOT/'dataset/PyMouseStudio/sessions/AZL3/reconstruction_pilot_v2';out.mkdir(parents=True,exist_ok=False)
    frames=list(range(780,780+args.frames));names=list(r['cameras']);side=[n for n in names if r['cameras'][n]['mask']]
    Ks={n:np.array(cal[n]['K']) for n in names};Es={n:np.array(cal[n]['world_to_camera_mm']) for n in names}
    images,masks,uv,conf=load_clip(r,cal,out,frames)
    xyz=np.full((len(frames),31,3),np.nan);errors=np.full((len(frames),31),np.nan)
    for t in range(len(frames)):
        for j in range(31):
            obs={n:uv[n][t,j] for n in side if conf[n][t,j]>=.8 and np.isfinite(uv[n][t,j]).all()}
            X,e=triangulate(obs,Ks,Es)
            if X is not None:xyz[t,j]=X;errors[t,j]=e
    valid=xyz[np.isfinite(xyz).all(-1)]
    assert np.min(np.isfinite(xyz).all(-1).sum(-1))>=4,'Insufficient calibrated DLC landmarks for bounds'
    center=np.median(valid,axis=0);valid=valid[np.linalg.norm(valid-center,axis=1)<150]
    low=valid.min(0)-40;high=valid.max(0)+40
    assert np.max(high-low)<300, 'Implausible bounds; calibration needs inspection'
    np.savez_compressed(out/'triangulation_diagnostic_NOT_GT.npz',xyz_mm=xyz,reprojection_error_px=errors,source_frames=frames)
    voxel=1.;axes=[np.arange(low[i],high[i]+.5,voxel) for i in range(3)]
    points=np.stack(np.meshgrid(*axes,indexing='ij'),-1).reshape(-1,3).astype(np.float32)
    assert len(points)<15000000
    cache=[]
    for n in side:
        xy,z=project(points,Ks[n],Es[n]);x,y=np.rint(xy).astype(np.int32).T
        good=(z>0)&(x>=0)&(x<1080)&(y>=0)&(y<1200)
        ids=np.clip(y,0,1199).astype(np.int64)*1080+np.clip(x,0,1079)
        cache.append((torch.from_numpy(ids).cuda(),torch.from_numpy(good).cuda()))
    config=dict(method='DANNCE-style calibrated silhouette visual hull',side_mask_views=side,
                bottom_input_mask_available=False,bottom_depth_source='rendered side-view hull, no independent bottom constraint',
                voxel_mm=voxel,depth_units='mm',depth_semantics='camera_z',invalid_background_depth=0,
                pfm_flip_y=True,size_wh=[1080,1200],bounds_mm=[low.tolist(),high.tolist()],
                candidate_voxels=len(points),frame_indices=frames,fps=52,
                bounds_source='robust triangulated source31 DLC, NOT manual labels or pose ground truth',
                bounds_likelihood_threshold=.8,bounds_padding_mm=40,
                mask_source='unverified supplied side-view masks; rectified once',
                lens_input_assumption='raw distorted, per user',train_allowed=False,
                strict_min_votes=len(side),relaxed_min_votes=len(side)-1,
                synchronization='equal frame indices assumed, not hardware verified',
                triangulation_median_reprojection_px=float(np.nanmedian(errors)))
    write(out/'config.json',config)
    reviews=out/'review';reviews.mkdir();records=[];video=None
    for t,frame in enumerate(frames):
        votes=torch.zeros(len(points),device='cuda',dtype=torch.uint8)
        for n,(ids,good) in zip(side,cache):
            mask=torch.from_numpy(masks[n][t].reshape(-1)).cuda()
            votes+=((mask[ids]>127)&good).to(torch.uint8)
        vote_np=votes.cpu().numpy();strict=points[vote_np==len(side)];relaxed=points[vote_np>=len(side)-1]
        if len(strict)<100:raise RuntimeError(f'Empty strict hull at {frame}: {len(strict)}')
        boundary=float(np.mean(np.any((strict-low<1.01)|(high-strict<1.01),axis=1)))
        scene=out/f'frame_{frame:06d}';scene.mkdir();viewstats=[]
        fig,axs=plt.subplots(3,len(names),figsize=(21,10),layout='constrained')
        for v,n in enumerate(names):
            full=np.full((1200,1080),255,dtype=np.uint8)
            unclip=rasterize_depth(strict,full,Ks[n],Es[n],voxel)
            relaxed_depth=rasterize_depth(relaxed,full,Ks[n],Es[n],voxel) if t in [0,len(frames)//2,len(frames)-1] else None
            inputmask=masks[n][t] if n in masks else None
            depth=unclip.copy()
            if inputmask is not None:depth[inputmask==0]=0
            pfm=scene/'rendered_depth_maps'/f'{v:08d}.pfm';write_pfm(pfm,depth)
            maskpath=scene/'masks'/f'{v:08d}.png';maskpath.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(maskpath),np.uint8(depth>0)*255)
            assert np.array_equal(read_pfm(pfm),depth) and np.isfinite(depth).all() and (depth>=0).all()
            if relaxed_depth is not None:
                write_pfm(scene/'relaxed_diagnostic_depth'/f'{v:08d}.pfm',relaxed_depth)
            hullmask=unclip>0
            if inputmask is not None:
                im=inputmask>0;iou=float(np.count_nonzero(hullmask&im)/max(1,np.count_nonzero(hullmask|im)))
                coverage=float(np.count_nonzero(hullmask&im)/max(1,np.count_nonzero(im)))
            else:iou=coverage=None
            values=depth[depth>0]
            viewstats.append(dict(camera=n,view_id=v,depth=str(pfm.relative_to(out)),
                foreground_pixels=int(len(values)),depth_p2_p98_mm=np.percentile(values,[2,98]).tolist() if len(values) else [],
                unclipped_hull_vs_supplied_mask_iou=iou,supplied_mask_coverage=coverage,
                mask_provenance='hull projection' if inputmask is None else 'hull projection clipped by supplied mask'))
            rgb=cv2.cvtColor(images[n][t],cv2.COLOR_BGR2RGB);overlay=rgb.copy()
            overlay[hullmask]=(.6*overlay[hullmask]+.4*np.array([0,255,80])).astype(np.uint8)
            axs[0,v].imshow(rgb);axs[0,v].set_title(f'cam{n}: rectified RGB')
            axs[1,v].imshow(overlay);axs[1,v].set_title('Hull projection'+(f' IoU {iou:.2f}' if iou is not None else ' (no input mask)'),fontsize=9)
            # Shared fixed scale across cameras and time; exact mm retained in PFM.
            cm=axs[2,v].imshow(np.ma.masked_where(depth==0,depth),cmap='turbo',vmin=100,vmax=320)
            axs[2,v].set_title('camera-Z mm; black = invalid',fontsize=8);axs[2,v].set_facecolor('black')
            for ax in axs[:,v]:ax.axis('off')
        fig.colorbar(cm,ax=list(axs[2,:]),orientation='horizontal',shrink=.6,label='Foreground pseudo-depth, mm (NOT scene-depth GT)')
        fig.suptitle(f'AZL3 source frame {frame} | strict {len(side)}/{len(side)} side masks | bottom camA unconstrained | 1 mm voxels')
        png=reviews/f'{frame:06d}.png';fig.savefig(png,dpi=90);plt.close(fig)
        im=cv2.imread(str(png))
        if video is None:video=cv2.VideoWriter(str(reviews/'multiview_depth_review.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),8,(im.shape[1],im.shape[0]))
        assert video.isOpened();video.write(im)
        if t in [0,len(frames)//2,len(frames)-1]:
            # Voxel-center point cloud, not a watertight surface mesh.
            stride=max(1,len(strict)//100000);cloud=strict[::stride];xy,z=project(cloud,Ks['A'],Es['A'])
            x,y=np.rint(xy).astype(int).T;colors=np.full((len(cloud),3),120,dtype=np.uint8)
            good=(z>0)&(x>=0)&(x<1080)&(y>=0)&(y<1200)
            colors[good]=images['A'][t][y[good],x[good],::-1]
            write_ply(scene/'hull_voxels_mm.ply',cloud,colors)
        records.append(dict(frame=frame,strict_voxels=len(strict),relaxed_voxels=len(relaxed),
                            boundary_voxel_fraction=boundary,views=viewstats,train_allowed=False))
        write(out/'progress.json',dict(completed=len(records),total=len(frames),latest_frame=frame))
        print(f'PILOT {t+1}/{len(frames)} frame{frame}: strict={len(strict)}, relaxed={len(relaxed)}',flush=True)
    video.release()
    write(out/'report.json',dict(state='complete',frames=records,
          median_side_unclipped_iou=float(np.median([v['unclipped_hull_vs_supplied_mask_iou'] for r in records for v in r['views'] if v['unclipped_hull_vs_supplied_mask_iou'] is not None])),
          automated_file_checks_passed=True,manual_visual_review_required=True,
          bottom_geometry_independently_validated=False,train_allowed=False))


if __name__=='__main__':main()
