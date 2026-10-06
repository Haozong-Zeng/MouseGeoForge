"""Independently check delivered limb masks/depths and publish review manifest."""
from pathlib import Path
import json
import subprocess
import sys
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from add_dannce_real_multiview_pseudo_depth import read_pfm
from export_pymousestudio_rectified import write,digest
from repair_pymousestudio_limb_hull import component_metrics


def main():
    s=ROOT/'dataset/PyMouseStudio/sessions/AZL3'
    out=s/'reconstruction_limb_guarded_v1';base=s/'reconstruction_tail_guarded_v2'
    report=json.loads((out/'report.json').read_text())
    assert [r['frame'] for r in report['frames']]==list(range(780,812))
    old_values_changed=0;outside_input=0;bottom_tail_changed=0;added_total=0
    components_before=0;components_after=0;orphan_before=0;orphan_after=0
    indices=[]
    for f in range(780,812):
        scene=out/f'frame_{f:06d}'
        extension=cv2.imread(str(scene/'bottom_limb_mask_extension.png'),0)>0
        roi=cv2.imread(str(scene/'bottom_limb_search_roi.png'),0)>0
        tail=cv2.imread(str(s/f'tail_dlc_video_masks_v1/tail_roi/{f:06d}.png'),0)>0
        assert not (extension&(~roi|tail)).any()
        views=[]
        for v,n in enumerate('ABCDEFG'):
            before=read_pfm(base/f'frame_{f:06d}/rendered_depth_maps/{v:08d}.pfm')
            depth=read_pfm(scene/f'rendered_depth_maps/{v:08d}.pfm')
            mask=cv2.imread(str(scene/f'masks/{v:08d}.png'),0)
            added=cv2.imread(str(scene/f'limb_added_pixels/{v:08d}.png'),0)>0
            assert depth.dtype==np.float32 and depth.shape==(1200,1080)
            assert np.isfinite(depth).all() and (depth>=0).all()
            assert np.array_equal(mask>0,depth>0) and set(np.unique(mask))<={0,255}
            assert np.array_equal(added,(before==0)&(depth>0))
            old_values_changed+=int(np.count_nonzero(depth[before>0]!=before[before>0]))
            source=extension if n=='A' else cv2.imread(str(s/f'pilot_input_masks/{f:06d}/cam{n}.png'),0)>0
            outside_input+=int(np.count_nonzero(added&~source))
            if n=='A': bottom_tail_changed+=int(np.count_nonzero(depth[tail]!=before[tail]))
            added_total+=int(added.sum())
            a,b=component_metrics(before>0),component_metrics(depth>0)
            components_before+=int(a['components_ge20']>1);components_after+=int(b['components_ge20']>1)
            orphan_before+=a['orphan_pixels'];orphan_after+=b['orphan_pixels']
            views.append(dict(camera=n,view_id=v,size_wh=[1080,1200],depth_unit='mm',
                depth=str((scene/f'rendered_depth_maps/{v:08d}.pfm').relative_to(s)),
                mask=str((scene/f'masks/{v:08d}.png').relative_to(s)),
                pseudo_limb_additions=str((scene/f'limb_added_pixels/{v:08d}.png').relative_to(s))))
        indices.append(dict(source_frame=f,time_seconds=f/52,views=views,train_allowed=False))
    assert old_values_changed==outside_input==bottom_tail_changed==0
    receipt=json.loads((ROOT/'dataset/PyMouseStudio/annotation_delivery.json').read_text())
    assert digest(ROOT/receipt['path'])==receipt['sha256']
    movie=out/'review/limb_repair_comparison_h264.mp4'
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
        '-show_entries','stream=codec_name,nb_read_frames,r_frame_rate','-of','json',str(movie)]))['streams'][0]
    assert probe['codec_name']=='h264' and probe['nb_read_frames']=='32' and probe['r_frame_rate']=='8/1'
    (out/'frames.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in indices))
    result=dict(passed=True,frames=32,views=7,depth_maps_checked=224,size_wh=[1080,1200],
        depth_unit='mm',depth_semantics='camera_z',old_nonzero_depth_values_changed=old_values_changed,
        additions_outside_source_mask=outside_input,bottom_tail_roi_pixels_changed=bottom_tail_changed,
        added_foreground_pixels=added_total,images_with_multiple_components_ge20_before=components_before,
        images_with_multiple_components_ge20_after=components_after,orphan_pixels_before=orphan_before,
        orphan_pixels_after=orphan_after,connectivity_metric_is_not_accuracy=True,
        annotation_zip_unchanged=True,training_supervision_enabled=False,
        visually_reviewed_frames=[780,791,796,802,803,808,809,810,811],
        visual_findings=['811 rear leg connection restored in side B/C/D/F/G; visible bottom paw recovered.',
                         '802/803/808 rear-leg gaps reduced without filling inter-leg background.',
                         'Some pre-existing silhouette fragments and head omissions remain; not all masks are a single component.'],
        video_sha256=digest(movie))
    write(out/'verification.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
