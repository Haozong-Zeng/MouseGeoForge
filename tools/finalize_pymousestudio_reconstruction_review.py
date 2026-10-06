"""Publish explicit per-frame paths, diagnostics and honest visual QA limits."""
from pathlib import Path
import sys,json
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from add_dannce_real_multiview_pseudo_depth import read_pfm
from export_pymousestudio_rectified import write


def main():
    session=ROOT/'dataset/PyMouseStudio/sessions/AZL3'
    cal=json.loads((session/'calibration.json').read_text());names=list('ABCDEFG')
    baseline=session/'reconstruction_pilot_v2';refined=session/'reconstruction_bottom_refined_v1'
    br=json.loads((baseline/'report.json').read_text());rr=json.loads((refined/'report.json').read_text())
    for folder,report in [(baseline,br),(refined,rr)]:
        index=[]
        for f in report['frames']:
            frame=f['frame'];views=[]
            for v,n in enumerate(names):
                depthfile=folder/f'frame_{frame:06d}/rendered_depth_maps/{v:08d}.pfm'
                maskfile=folder/f'frame_{frame:06d}/masks/{v:08d}.png'
                d=read_pfm(depthfile);mask=cv2.imread(str(maskfile),0)
                assert d.shape==(1200,1080) and d.dtype==np.float32 and np.isfinite(d).all() and np.min(d)>=0
                assert set(np.unique(mask))<={0,255} and np.array_equal(mask>0,d>0)
                views.append(dict(view_id=v,camera=n,rgb=f'pilot_rgb/{frame:06d}/cam{n}.png',
                    depth=str(depthfile.relative_to(session)),mask=str(maskfile.relative_to(session)),
                    intrinsic=cal[n]['K'],extrinsic_world_to_camera_mm=cal[n]['world_to_camera_mm'],
                    distortion=[0.]*5,size_wh=[1080,1200],depth_unit='mm',depth_semantics='camera_z',
                    depth_is_pseudo_label=True,mask_is_pseudo_label=True,pose_supervision_valid=False,
                    training_supervision_enabled=False))
            index.append(dict(source_frame=frame,time_seconds=frame/52,views=views,
                               path_root='parent session directory',train_allowed=False))
        (folder/'frames.jsonl').write_text(''.join(json.dumps(f)+'\n' for f in index),encoding='utf-8')
        write(folder/'metadata.json',dict(frame_count=32,source_fps=52,source_start=780,source_stop_inclusive=811,
                  duration_sample_count_seconds=32/52,camera_count=7,size_wh=[1080,1200],
                  foreground_only=True,background_depth_valid=False,invalid_depth_value=0,
                  depth_unit='mm',depth_semantics='camera_z',pfm_flip_y=True,
                  world_system='unchanged source calibration, not ground-aligned',
                  frame_paths_root='parent session directory',manual_ground_truth=False,train_allowed=False))
    comparison=[]
    for b,r in zip(br['frames'],rr['frames']):
        frame=b['frame'];target=cv2.imread(str(refined/f'frame_{frame:06d}/bottom_sam2_mask.png'),0)>0
        pred=read_pfm(baseline/f'frame_{frame:06d}/rendered_depth_maps/00000000.pfm')>0
        overlap=float(np.count_nonzero(pred&target)/np.count_nonzero(pred|target))
        comparison.append(dict(frame=frame,side_only_bottom_vs_sam2_iou=overlap,
             refined_bottom_vs_sam2_iou=r['views'][0]['unclipped_mask_agreement_iou'],
             baseline_side_iou_mean=float(np.mean([v['unclipped_hull_vs_supplied_mask_iou'] for v in b['views'][1:]])),
             refined_side_iou_mean=float(np.mean([v['unclipped_mask_agreement_iou'] for v in r['views'][1:]])),
             baseline_voxels=b['strict_voxels'],refined_voxels=r['refined_voxels']))
    metrics={key:float(np.mean([c[key] for c in comparison])) for key in comparison[0] if key!='frame'}
    counts=np.array([f['refined_voxels'] for f in rr['frames']],float)
    review=dict(automated_pfms_checked=448,comparison=metrics,
          max_adjacent_relative_occupancy_change=float(np.max(np.abs(np.diff(counts))/counts[:-1])),
          max_boundary_voxel_fraction=max(f['bounds_boundary_fraction'] for f in rr['frames']),
          metric_warning='Mask consistency with input pseudo masks is NOT independent depth/segmentation accuracy.',
          visually_reviewed_frames=[780,796,808,809,811],visually_reviewed_views=list('ABCDEFG'),
          visual_findings=['Bottom silhouette excess reduced by bottom SAM2 constraint.',
                           'Relaxed side voting recovers more tail, but increases some side-view excess.',
                           'Frame811 still misses distal tail and part of a hind paw; SAM2 mask also misses paw pixels.',
                           'Orbit shows coarse connected body and tail with small disconnected tail-end fragments.',
                           'Only 32-frame (0.615s by sample count) development clip checked; not all sessions.',
                           'Frame808 to809 shows a 15.1 percent occupied-volume jump; temporal stability remains unverified.'],
          approved_use='geometry feasibility and human review ONLY',
          full_dataset_depth_generation_approved=False,training_supervision_enabled=False)
    write(session/'reconstruction_review.json',review)
    fig,axs=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    time=np.arange(32)/52
    for ax,fields,title in zip(axs,
          [('side_only_bottom_vs_sam2_iou','refined_bottom_vs_sam2_iou'),
           ('baseline_side_iou_mean','refined_side_iou_mean'),('baseline_voxels','refined_voxels')],
          ['Bottom: consistency with SAM2 (NOT GT)','Side: consistency with supplied masks','Occupied voxel count (1mm lattice)']):
        for field in fields:ax.plot(time,[c[field] for c in comparison],label=field)
        ax.set_title(title,fontsize=9);ax.set_xlabel('Time from frame780 (s)');ax.legend(fontsize=6);ax.grid(alpha=.2)
    fig.savefig(refined/'review/temporal_diagnostics.png',dpi=130);plt.close(fig)
    # Correct the side-only review's former background-colour caption by
    # regenerating from the unchanged scientific arrays, never painting pixels.
    writer=None
    for b in br['frames']:
        frame=b['frame'];fig,axs=plt.subplots(3,7,figsize=(21,10),layout='constrained')
        for v,n in enumerate(names):
            rgb=cv2.cvtColor(cv2.imread(str(session/f'pilot_rgb/{frame:06d}/cam{n}.png')),cv2.COLOR_BGR2RGB)
            depth=read_pfm(baseline/f'frame_{frame:06d}/rendered_depth_maps/{v:08d}.pfm');mask=depth>0
            ov=rgb.copy();ov[mask]=(.6*ov[mask]+.4*np.array([0,255,80])).astype(np.uint8)
            axs[0,v].imshow(rgb);axs[0,v].set_title('cam'+n)
            axs[1,v].imshow(ov);axs[1,v].set_title('Published depth support',fontsize=9)
            cm=axs[2,v].imshow(np.ma.masked_where(~mask,depth),cmap='turbo',vmin=100,vmax=320)
            for ax in axs[:,v]:ax.axis('off')
        fig.colorbar(cm,ax=list(axs[2,:]),orientation='horizontal',shrink=.6,label='camera-Z mm; WHITE = invalid/background; foreground pseudo-depth only')
        fig.suptitle(f'AZL3 frame{frame} | baseline strict 6/6 side masks | no bottom input mask')
        p=baseline/'review'/f'{frame:06d}.png';fig.savefig(p,dpi=90);plt.close(fig)
        a=cv2.imread(str(p))
        if writer is None:writer=cv2.VideoWriter(str(baseline/'review/multiview_depth_review.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),8,(a.shape[1],a.shape[0]))
        writer.write(a)
    writer.release()
    print(json.dumps(review,indent=2))


if __name__=='__main__':main()
