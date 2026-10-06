"""Check tail-only invariants and native depth products for all 32x7 views."""
from pathlib import Path
import sys,json,subprocess
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from add_dannce_real_multiview_pseudo_depth import read_pfm
from export_pymousestudio_rectified import write,digest


def main():
    s=ROOT/'dataset/PyMouseStudio/sessions/AZL3';out=s/'reconstruction_tail_guarded_v2'
    old=s/'reconstruction_bottom_refined_v1';report=json.loads((out/'report.json').read_text())
    assert report['state']=='complete' and len(report['frames'])==32
    checked=0;outside=0;lost=0
    for frame in range(780,812):
        roi=cv2.imread(str(out/f'frame_{frame:06d}/bottom_tail_roi.png'),0)>0
        for v in range(7):
            a=read_pfm(old/f'frame_{frame:06d}/rendered_depth_maps/{v:08d}.pfm')
            b=read_pfm(out/f'frame_{frame:06d}/rendered_depth_maps/{v:08d}.pfm')
            m=cv2.imread(str(out/f'frame_{frame:06d}/masks/{v:08d}.png'),0)
            changed=cv2.imread(str(out/f'frame_{frame:06d}/tail_changed_pixels/{v:08d}.png'),0)
            assert b.shape==(1200,1080) and b.dtype==np.float32 and np.isfinite(b).all() and (b>=0).all()
            assert np.array_equal(m>0,b>0) and set(np.unique(m))<={0,255}
            assert np.array_equal(changed>0,abs(a-b)>.01)
            lost+=int(np.count_nonzero((a>0)&(b==0)))
            if v==0:outside+=int(np.count_nonzero(a[~roi]!=b[~roi]))
            checked+=1
    assert lost==0 and outside==0
    receipt=json.loads((ROOT/'dataset/PyMouseStudio/annotation_delivery.json').read_text())
    assert digest(ROOT/receipt['path'])==receipt['sha256']
    movie=out/'review/tail_repair_comparison_h264.mp4'
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
           '-show_entries','stream=codec_name,nb_read_frames,r_frame_rate','-of','json',str(movie)]))['streams'][0]
    assert probe['codec_name']=='h264' and probe['nb_read_frames']=='32' and probe['r_frame_rate']=='8/1'
    verification=dict(passed=True,frames=32,views=7,depth_maps_checked=checked,size_wh=[1080,1200],
          depth_unit='mm',depth_semantics='camera_z',lost_old_foreground_pixels=lost,
          changed_bottom_pixels_outside_tail_roi=outside,annotation_zip_unchanged=True,
          anchor_frames=[783,796],tail_middle_points_used=False,
          visually_reviewed_frames=[780,796,811],visually_reviewed_views=list('ABCDEFG'),
          visual_findings=['Tail endpoints restored in inspected frames796 and811.',
                           'Small reflection adhesion/jagged boundaries remain around frame796.',
                           'Unconstrained full SAM2 tracking has glass false positives; only tail ROI additions are used.'],
          optional_sam2_cuda_hole_postprocessing='Skipped: installed sam2._C unavailable; neural inference completed, no dependency changes.',
          mean_input_tail_mask_coverage_before=report['mean_tail_coverage_before'],
          mean_input_tail_mask_coverage_after=report['mean_tail_coverage_after'],
          coverage_is_independent_accuracy=False,training_supervision_enabled=False,
          video_sha256=digest(movie),focused_unittest_count=6)
    write(out/'verification.json',verification);print(json.dumps(verification,indent=2))


if __name__=='__main__':main()
