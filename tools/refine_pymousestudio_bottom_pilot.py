"""Add SAM2 bottom silhouettes to the development visual-hull pilot.

Bottom SAM2 is prompted by coarse projected geometry, not human keypoints.
Compare strict side-only, all seven masks, and bottom + 5/6 side votes.
"""
from pathlib import Path
import sys,json
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from add_dannce_real_multiview_pseudo_depth import project,rasterize_depth,write_pfm,read_pfm
from reconstruct_pymousestudio_pilot import write_ply
from export_pymousestudio_rectified import write
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
cv2.setNumThreads(1)


def main():
    session=ROOT/'dataset/PyMouseStudio/sessions/AZL3';base=session/'reconstruction_pilot_v2'
    config=json.loads((base/'config.json').read_text());cal=json.loads((session/'calibration.json').read_text())
    out=session/'reconstruction_bottom_refined_v1';out.mkdir(exist_ok=False)
    names=list('ABCDEFG');frames=config['frame_indices'];side=list('BCDEFG')
    Ks={n:np.array(cal[n]['K']) for n in names};Es={n:np.array(cal[n]['world_to_camera_mm']) for n in names}
    low,high=np.array(config['bounds_mm']);axes=[np.arange(low[i],high[i]+.5,1.) for i in range(3)]
    points=np.stack(np.meshgrid(*axes,indexing='ij'),-1).reshape(-1,3).astype(np.float32)
    cache={}
    for n in names:
        uv,z=project(points,Ks[n],Es[n]);x,y=np.rint(uv).astype(np.int32).T
        good=(z>0)&(x>=0)&(x<1080)&(y>=0)&(y<1200)
        ids=np.clip(y,0,1199).astype(np.int64)*1080+np.clip(x,0,1079)
        cache[n]=(torch.from_numpy(ids).cuda(),torch.from_numpy(good).cuda())
    predictor=SAM2ImagePredictor(build_sam2('configs/sam2.1/sam2.1_hiera_s.yaml',
                        str(ROOT/'third_party/checkpoints/sam2.1_hiera_small.pt'),device='cuda'))
    reviews=out/'review';reviews.mkdir();records=[];writer=None
    for index,frame in enumerate(frames):
        imgs={n:cv2.imread(str(session/'pilot_rgb'/f'{frame:06d}'/f'cam{n}.png')) for n in names}
        assert all(a is not None for a in imgs.values())
        masks={n:cv2.imread(str(session/'pilot_input_masks'/f'{frame:06d}'/f'cam{n}.png'),0) for n in side}
        coarse=read_pfm(base/f'frame_{frame:06d}/rendered_depth_maps/00000000.pfm')>0
        y,x=np.nonzero(coarse);assert len(x)>100
        box=np.array([max(0,x.min()-20),max(0,y.min()-20),min(1079,x.max()+20),min(1199,y.max()+20)],np.float32)
        distance=cv2.distanceTransform(np.uint8(coarse),cv2.DIST_L2,5)
        yy,xx=np.unravel_index(np.argmax(distance),distance.shape)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            predictor.set_image(cv2.cvtColor(imgs['A'],cv2.COLOR_BGR2RGB))
            candidates,scores,_=predictor.predict(point_coords=np.array([[xx,yy]],np.float32),
                                point_labels=np.array([1],np.int32),box=box,multimask_output=True)
        best=int(np.argmax(scores));masks['A']=np.uint8(candidates[best]>0)*255
        scene=out/f'frame_{frame:06d}';scene.mkdir();p=scene/'bottom_sam2_mask.png';cv2.imwrite(str(p),masks['A'])
        votes=torch.zeros(len(points),device='cuda',dtype=torch.uint8)
        bottom=None
        for n in names:
            ids,good=cache[n];mask=torch.from_numpy(masks[n].reshape(-1)).cuda()
            hit=(mask[ids]>127)&good
            if n=='A':bottom=hit
            else:votes+=hit.to(torch.uint8)
        strict=points[((votes==6)&bottom).cpu().numpy()]
        refined=points[((votes>=5)&bottom).cpu().numpy()]
        if len(refined)<100:raise RuntimeError('Empty bottom-refined hull')
        stats=[];fig,axs=plt.subplots(3,7,figsize=(21,10),layout='constrained')
        for v,n in enumerate(names):
            full=np.full((1200,1080),255,np.uint8)
            unclip=rasterize_depth(refined,full,Ks[n],Es[n],1.)
            depth=unclip.copy();depth[masks[n]==0]=0
            p=scene/'rendered_depth_maps'/f'{v:08d}.pfm';write_pfm(p,depth)
            assert np.array_equal(depth,read_pfm(p)) and np.isfinite(depth).all()
            p=scene/'masks'/f'{v:08d}.png';p.parent.mkdir(exist_ok=True);cv2.imwrite(str(p),np.uint8(depth>0)*255)
            pred=unclip>0;target=masks[n]>0
            iou=float(np.count_nonzero(pred&target)/max(1,np.count_nonzero(pred|target)))
            rgb=cv2.cvtColor(imgs[n],cv2.COLOR_BGR2RGB);ov=rgb.copy();ov[pred]=(.6*ov[pred]+.4*np.array([0,255,80])).astype(np.uint8)
            axs[0,v].imshow(rgb);axs[0,v].set_title('cam'+n)
            axs[1,v].imshow(ov);axs[1,v].set_title(f'Hull/mask agreement {iou:.2f}',fontsize=9)
            cm=axs[2,v].imshow(np.ma.masked_where(depth==0,depth),cmap='turbo',vmin=100,vmax=320)
            for ax in axs[:,v]:ax.axis('off')
            if index in [0,len(frames)//2,len(frames)-1]:
                strictdepth=rasterize_depth(strict,full,Ks[n],Es[n],1.)
                write_pfm(scene/'strict7_diagnostic_depth'/f'{v:08d}.pfm',strictdepth)
            vals=depth[depth>0]
            stats.append(dict(camera=n,view_id=v,unclipped_mask_agreement_iou=iou,
                              foreground_pixels=len(vals),depth_p2_p98_mm=np.percentile(vals,[2,98]).tolist(),
                              input_mask_provenance='SAM2 geometry-prompted pseudo mask' if n=='A' else 'supplied unverified mask'))
        fig.colorbar(cm,ax=list(axs[2,:]),orientation='horizontal',shrink=.6,label='camera-Z mm; white = invalid/background (no scene depth)')
        fig.suptitle(f'AZL3 frame {frame} | bottom SAM2 + >=5/6 side masks | GEOMETRY CHECK ONLY; not independent GT')
        p=reviews/f'{frame:06d}.png';fig.savefig(p,dpi=90);plt.close(fig)
        img=cv2.imread(str(p))
        if writer is None:writer=cv2.VideoWriter(str(reviews/'multiview_depth_review_slow8fps.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),8,(img.shape[1],img.shape[0]))
        assert writer.isOpened();writer.write(img)
        if index in [0,len(frames)//2,len(frames)-1]:
            cloud=refined[::max(1,len(refined)//100000)];xy,z=project(cloud,Ks['A'],Es['A']);x,y=np.rint(xy).astype(int).T
            good=(z>0)&(x>=0)&(x<1080)&(y>=0)&(y<1200);colors=np.full((len(cloud),3),100,np.uint8)
            colors[good]=imgs['A'][y[good],x[good],::-1];write_ply(scene/'hull_voxels_mm.ply',cloud,colors)
            # Three-way mask overlay, same bottom RGB, no ground truth claim.
            fig,axes=plt.subplots(1,4,figsize=(16,5),layout='constrained')
            selections=[coarse,rasterize_depth(strict,np.full((1200,1080),255,np.uint8),Ks['A'],Es['A'],1.)>0,
                        rasterize_depth(refined,np.full((1200,1080),255,np.uint8),Ks['A'],Es['A'],1.)>0,masks['A']>0]
            for ax,selection,title in zip(axes,selections,['6/6 side only','7/7 including bottom','bottom + 5/6 sides','SAM2 bottom mask (pseudo)']):
                rgb=cv2.cvtColor(imgs['A'],cv2.COLOR_BGR2RGB);rgb[selection]=(.6*rgb[selection]+.4*np.array([0,255,80])).astype(np.uint8)
                ax.imshow(rgb);ax.set_title(title);ax.axis('off')
            fig.savefig(reviews/f'bottom_comparison_{frame:06d}.png',dpi=120);plt.close(fig)
        boundary=float(np.mean(np.any((refined-low<1.01)|(high-refined<1.01),axis=1)))
        records.append(dict(frame=frame,strict7_voxels=len(strict),refined_voxels=len(refined),views=stats,
                            sam2_candidate_score=float(scores[best]),bounds_boundary_fraction=boundary,train_allowed=False))
        write(out/'progress.json',dict(completed=index+1,total=len(frames)));print(f'BOTTOM {index+1}/{len(frames)}',flush=True)
    writer.release()
    write(out/'config.json',{**config,'method':'bottom SAM2 + >=5/6 supplied side masks, 1mm visual hull',
          'bottom_input_mask_available':True,'bottom_depth_source':'bottom-constrained visual hull',
          'bottom_prompt':'coarse hull bounding box + maximum distance-transform interior point; no manual GT',
          'bottom_mask_quality_independently_verified':False,'mask_agreement_is_independent_accuracy':False,
          'comparison_strict_min_votes':7,'production_rule':'bottom mandatory AND >=5/6 side votes',
          'playback_fps':8,'source_fps':52,'train_allowed':False})
    write(out/'report.json',dict(state='complete',frames=records,automated_file_checks_passed=True,
          median_mask_agreement_iou=float(np.median([v['unclipped_mask_agreement_iou'] for r in records for v in r['views']])),
          manual_visual_review_required=True,train_allowed=False))


if __name__=='__main__':main()
