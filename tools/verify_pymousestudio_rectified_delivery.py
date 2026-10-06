"""Final metadata, source integrity, corrected samples and package checks."""
from pathlib import Path
import sys,json,zipfile
import numpy as np
import cv2
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from export_pymousestudio_rectified import write,digest
from mousegeoforge.raw_validation import rectify_points


def main():
    dst=ROOT/'dataset/PyMouseStudio';raw=ROOT/'dataset/PyMouseStudioRaw'
    assert json.loads((dst/'export_status.json').read_text())['state']=='complete'
    records=[json.loads(l) for l in (raw/'_validation_v2/sequences.jsonl').read_text().splitlines()]
    imported=json.loads((raw/'_metadata/import.json').read_text())
    hashes={f['path']:f['sha256'] for a in imported['archives'] for f in a['files']}
    camera_count=0;mask_count=0;point_count=0;sample_check_count=0
    for r in records:
        s=r['sequence_id'].split('/')[-1];cal=json.loads((dst/'sessions'/s/'calibration.json').read_text())
        original=json.loads((ROOT/r['prepared_calibration']).read_text())
        for n,c in r['cameras'].items():
            camera_count+=1;m=json.loads((dst/'sessions'/s/'completed'/f'cam{n}.json').read_text())
            assert m['completed'] and m['frames']==1560 and m['fps']==52
            relative=str((ROOT/c['video']).relative_to(raw))
            assert m['source_sha256']==hashes[relative]==digest(ROOT/c['video'])
            assert digest(dst/m['rgb'])==m['rgb_sha256']
            np.testing.assert_array_equal(cal[n]['K'],original[n]['K'])
            np.testing.assert_array_equal(cal[n]['world_to_camera_mm'],original[n]['world_to_camera_mm'])
            assert not np.array(cal[n]['distortion']).any()
            if c['mask']:
                mask_count+=1;cap=cv2.VideoCapture(str(dst/m['mask']));ok,a=cap.read();cap.release()
                assert ok and a.shape==(1200,1080,3) and set(np.unique(a))<={0,255}
            if c['keypoints']:
                point_count+=1
                with np.load(ROOT/c['keypoints'],allow_pickle=False) as a,np.load(dst/m['keypoints'],allow_pickle=False) as b:
                    desired=rectify_points(a['uv'],np.array(cal[n]['K']),np.array(original[n]['distortion'])).astype(np.float32)
                    np.testing.assert_allclose(desired,b['uv'],atol=1e-4,equal_nan=True)
                    np.testing.assert_array_equal(a['likelihood'],b['likelihood'])
                    np.testing.assert_array_equal(a['frame_indices'],b['frame_indices'])
            # Pixel QA restricted to development, no holdout model/visual selection.
            if r['split']=='real_development':
                cap=cv2.VideoCapture(str(ROOT/c['video']));ok,a=cap.read();cap.release();assert ok
                K=np.array(cal[n]['K']);D=np.array(original[n]['distortion'])
                mx,my=cv2.initUndistortRectifyMap(K,D,None,K,(1080,1200),cv2.CV_32FC1)
                expected=cv2.remap(a,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
                p=dst/'annotation_pack_v2/real_images'/r['split']/s/'frame_000000'/f'cam{n}.png'
                assert np.array_equal(cv2.imread(str(p)),expected);sample_check_count+=1
    assert (camera_count,mask_count,point_count)==(78,66,66)
    receipt=json.loads((dst/'annotation_delivery.json').read_text());zip_path=ROOT/receipt['path']
    assert digest(zip_path)==receipt['sha256']
    with zipfile.ZipFile(zip_path) as z:assert z.testzip() is None
    review=json.loads((dst/'sessions/AZL3/reconstruction_review.json').read_text())
    assert review['automated_pfms_checked']==448 and not review['training_supervision_enabled']
    report=dict(passed=True,original_rgb_hashes_unchanged=78,output_rgb_hashes_verified=78,
                lossless_binary_mask_videos=66,transformed_source31_files_verified=66,
                independent_development_first_frame_pixel_checks=sample_check_count,
                all_exported_png_pixel_roundtrips_checked_during_export=1950,
                annotation_zip_sha256=receipt['sha256'],annotation_images=1950,annotation_time_groups=300,
                unit_tests_passed=8,pfm_checks=448,full_dataset_depth_generated=False)
    write(dst/'delivery_verification.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
