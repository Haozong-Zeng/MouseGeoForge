"""Verify one-cloud rendering, geometry provenance and all delivered masks."""
from pathlib import Path
import sys,json,subprocess
import numpy as np
import cv2
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from add_dannce_real_multiview_pseudo_depth import read_pfm,rasterize_depth
from export_pymousestudio_rectified import write,digest


def main():
    s=ROOT/'dataset/PyMouseStudio/sessions/AZL3';out=s/'reconstruction_unified_guarded_v1'
    prev=s/'reconstruction_limb_guarded_v1';body=s/'reconstruction_bottom_refined_v1';track=s/'tail_dlc_video_masks_v1'
    cal=json.loads((s/'calibration.json').read_text());report=json.loads((out/'report.json').read_text())
    tail_report={r['frame']:r for r in json.loads((s/'reconstruction_tail_guarded_v2/report.json').read_text())['frames']}
    limb_report={r['frame']:r for r in json.loads((prev/'report.json').read_text())['frames']}
    assert report['state']=='complete' and [r['frame'] for r in report['frames']]==list(range(780,812))
    ratios=[];fcoverage=[];tailcoverage=[];losses=[];outside=0;checked=0;seams=[]
    for r in report['frames']:
        f=r['frame'];scene=out/f'frame_{f:06d}';g=dict(np.load(scene/'geometry.npz'))
        pts=g['points_world_mm'];soft=g['bottom_relaxed']
        assert pts.ndim==2 and pts.shape[1]==3 and np.isfinite(pts).all()
        assert r['body_voxels']==tail_report[f]['old_voxels']
        assert r['tail_added_voxels']==tail_report[f]['added_tail_voxels']
        assert r['limb_added_voxels']==limb_report[f]['accepted_voxels']
        assert int(soft.sum())==r['bottom_relaxed_voxels']
        ratios.append(float(soft.sum()/r['body_voxels']));assert ratios[-1]<=.15
        tail=cv2.imread(str(track/f'tail_roi/{f:06d}.png'),0)
        bottom=cv2.imread(str(body/f'frame_{f:06d}/bottom_sam2_mask.png'),0)|tail|cv2.imread(str(prev/f'frame_{f:06d}/bottom_limb_mask_extension.png'),0)
        for v,n in enumerate('ABCDEFG'):
            depth=read_pfm(scene/f'rendered_depth_maps/{v:08d}.pfm')
            mask=cv2.imread(str(scene/f'masks/{v:08d}.png'),0)
            before=read_pfm(prev/f'frame_{f:06d}/rendered_depth_maps/{v:08d}.pfm')
            change=cv2.imread(str(scene/f'changed_pixels/{v:08d}.png'),0)
            source=bottom if n=='A' else cv2.imread(str(s/f'pilot_input_masks/{f:06d}/cam{n}.png'),0)
            assert depth.shape==(1200,1080) and depth.dtype==np.float32 and np.isfinite(depth).all() and (depth>=0).all()
            assert np.array_equal(mask>0,depth>0) and set(np.unique(mask))<={0,255}
            assert np.array_equal(change>0,abs(depth-before)>.01)
            outside+=int(np.count_nonzero((depth>0)&(source==0)))
            rerender=rasterize_depth(pts,source,np.array(cal[n]['K']),np.array(cal[n]['world_to_camera_mm']),1.)
            assert np.array_equal(rerender,depth),'Depth is not a unified one-cloud render'
            lost=np.count_nonzero((before>0)&(depth==0))/max(1,np.count_nonzero(before));losses.append(dict(frame=f,camera=n,fraction=float(lost)))
            if n=='A':tailcoverage.append(float(np.count_nonzero((tail>0)&(depth>0))/max(1,np.count_nonzero(tail))))
            if n=='F':
                row=r['views'][v];fcoverage.append([row['coverage_before'],row['coverage_unified'],row['coverage_after']])
                seams.append(dict(frame=f,**row['seam_final']))
            checked+=1
    assert outside==0
    for name in ['multiview','camF']:
        movie=out/f'review/{name}_h264.mp4'
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
            '-show_entries','stream=codec_name,nb_read_frames,r_frame_rate','-of','json',str(movie)]))['streams'][0]
        assert probe['codec_name']=='h264' and probe['nb_read_frames']=='32' and probe['r_frame_rate']=='8/1'
    receipt=json.loads((ROOT/'dataset/PyMouseStudio/annotation_delivery.json').read_text())
    assert digest(ROOT/receipt['path'])==receipt['sha256']
    result=dict(passed=True,frames=32,views=7,one_cloud_exact_rerender_checks=checked,depth_unit='mm',depth_semantics='camera_z',
        size_wh=[1080,1200],original_tail_limb_voxel_counts_matched=True,bottom_relaxed_volume_fraction_range=[min(ratios),max(ratios)],
        camF_mean_input_mask_coverage_before_unified_final=np.mean(fcoverage,0).tolist(),
        camF_min_input_mask_coverage_before_unified_final=np.min(fcoverage,0).tolist(),
        coverage_is_independent_accuracy=False,foreground_outside_input_masks=outside,
        tail_roi_mean_coverage=float(np.mean(tailcoverage)),tail_roi_min_coverage=min(tailcoverage),
        largest_old_foreground_loss=max(losses,key=lambda x:x['fraction']),
        camF_seams=seams,annotation_zip_unchanged=True,train_allowed=False)
    write(out/'verification.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
