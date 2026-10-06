"""DLC-consistent sparse tail-tip anchors + temporal bottom SAM2 segmentation.

Correct the truncated-hull bounding-box feedback loop. Use only tail tips that
passed three-view robust geometry; preserve provenance and never call them GT.
"""
from pathlib import Path
import sys,json
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from add_dannce_real_multiview_pseudo_depth import project,read_pfm
from export_pymousestudio_rectified import write
from repair_pymousestudio_tail_hull import tail_roi
from sam2.build_sam import build_sam2_video_predictor
cv2.setNumThreads(1)


def main():
    session=ROOT/'dataset/PyMouseStudio/sessions/AZL3';previous=session/'reconstruction_bottom_refined_v1'
    out=session/'tail_dlc_video_masks_v1';out.mkdir(exist_ok=False)
    numeric=out/'tracking_input_jpeg';numeric.mkdir();frames=list(range(780,812))
    records=json.loads((session/'tail_dlc_audit_v1/report.json').read_text())['records']
    records={(r['frame'],r['id']):r for r in records}
    cal=json.loads((session/'calibration.json').read_text())['A'];K=np.array(cal['K']);E=np.array(cal['world_to_camera_mm'])
    images={}
    for idx,frame in enumerate(frames):
        im=cv2.imread(str(session/f'pilot_rgb/{frame:06d}/camA.png'));images[frame]=im
        assert cv2.imwrite(str(numeric/f'{idx:06d}.jpg'),im,[cv2.IMWRITE_JPEG_QUALITY,98])
    predictor=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_s.yaml',str(ROOT/'third_party/checkpoints/sam2.1_hiera_small.pt'),device='cuda')
    anchors=[];masks={}
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        state=predictor.init_state(video_path=str(numeric),offload_video_to_cpu=True,offload_state_to_cpu=False)
        for frame in frames:
            tip=records[frame,30]['xyz_mm']
            if tip is None:continue
            xyz=np.array([records[frame,j]['xyz_mm'] for j in [24,25,30] if records[frame,j]['xyz_mm'] is not None])
            uv,z=project(xyz,K,E)
            assert (z>0).all() and ((uv>=[0,0])&(uv<[1080,1200])).all()
            coarse=read_pfm(session/f'reconstruction_pilot_v2/frame_{frame:06d}/rendered_depth_maps/00000000.pfm')>0
            distance=cv2.distanceTransform(np.uint8(coarse),cv2.DIST_L2,5);yy,xx=np.unravel_index(distance.argmax(),distance.shape)
            points=np.concatenate([[[xx,yy]],uv],0).astype(np.float32)
            # Deliberately does not inherit the truncated frame-local hull box.
            box=np.array([60.,180.,880.,1199.],np.float32)
            predictor.add_new_points_or_box(state,frame_idx=frame-780,obj_id=1,points=points,
                      labels=np.ones(len(points),np.int32),box=box)
            anchors.append(dict(frame=frame,points=points.tolist(),box=box.tolist(),
                                 tail_tip_inlier_reprojection_px=records[frame,30]['median_inlier_reprojection_px']))
        assert len(anchors)==2 and [a['frame'] for a in anchors]==[783,796]
        start=min(a['frame'] for a in anchors)-780
        for reverse in [False,True]:
            for idx,ids,logits in predictor.propagate_in_video(state,start_frame_idx=start,reverse=reverse):
                masks[idx+780]=np.uint8((logits[ids.index(1),0]>0).cpu().numpy())*255
        predictor.reset_state(state)
    assert set(masks)==set(frames)
    summary=[];reviews=out/'review';reviews.mkdir();writer=None
    for frame in frames:
        p=out/'masks'/f'{frame:06d}.png';p.parent.mkdir(exist_ok=True);cv2.imwrite(str(p),masks[frame])
        old=cv2.imread(str(previous/f'frame_{frame:06d}/bottom_sam2_mask.png'),0)
        root=project(np.array(records[frame,24]['xyz_mm'])[None],K,E)[0][0]
        nxt=records[frame,25]['xyz_mm'];nxt=project(np.array(nxt)[None],K,E)[0][0] if nxt is not None else None
        roi,meta=tail_roi(masks[frame],root,nxt)
        p=out/'tail_roi'/f'{frame:06d}.png';p.parent.mkdir(exist_ok=True);cv2.imwrite(str(p),roi)
        # Publish only the tail additions to the original bottom segmentation.
        combined=old|roi;p=out/'tail_augmented_masks'/f'{frame:06d}.png';p.parent.mkdir(exist_ok=True);cv2.imwrite(str(p),combined)
        fig,axes=plt.subplots(1,4,figsize=(16,5),layout='constrained')
        rgb=cv2.cvtColor(images[frame],cv2.COLOR_BGR2RGB)
        for ax,selection,title in zip(axes,[None,old>0,masks[frame]>0,combined>0],
               ['Rectified bottom RGB','Old frame-local box SAM2','DLC tip anchors + video SAM2','Tail-only additions (body preserved)']):
            im=rgb.copy()
            if selection is not None:im[selection]=(.6*im[selection]+.4*np.array([0,255,80])).astype(np.uint8)
            ax.imshow(im);ax.axis('off');ax.set_title(title,fontsize=9)
        if frame in [a['frame'] for a in anchors]:
            pts=np.array(next(a['points'] for a in anchors if a['frame']==frame));axes[2].scatter(pts[:,0],pts[:,1],c='red',s=12)
        fig.suptitle(f'Frame {frame}: predictions, not manual masks; only geometry-checked tips at 783 and 796 are anchors')
        p=reviews/f'{frame:06d}.png';fig.savefig(p,dpi=120);plt.close(fig)
        im=cv2.imread(str(p))
        if writer is None:writer=cv2.VideoWriter(str(reviews/'bottom_tail_video_tracking.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),8,(im.shape[1],im.shape[0]))
        writer.write(im)
        summary.append(dict(frame=frame,tail_roi=meta,added_pixels=int(np.count_nonzero((roi>0)&(old==0))),
                             total_tracked_mask_pixels=int(np.count_nonzero(masks[frame]))))
    writer.release();write(out/'report.json',dict(state='complete',anchors=anchors,frames=summary,
          method='SAM2 video tracking with robust source31 DLC root/tail0/tip prompts',
          full_tail_curve_from_DLC=False,training_supervision_enabled=False,
          inference_image_encoding='JPEG98 temporary input, masks evaluated on original rectified PNG canvas',
          caveat='Temporal predictions are not annotations; visually validate tail tip and glass reflections.'))
    print('COMPLETE',out)


if __name__=='__main__':main()
